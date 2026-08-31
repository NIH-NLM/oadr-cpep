"""
Shared low-level helpers for the oadr-cpep steps: data loading, within-site
scaling, cross-validation, and metrics. No step logic and no plotting live here
(plotting is in plot.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from . import oadr_data as od

# C-peptide unit conversion. Trials state preserved beta-cell function as
# >= 0.2 nmol/L; this package's target is in ng/mL, so the default responder
# threshold is that cutoff converted. Both numbers travel in every payload, so a
# ROC AUC is never silently compared across sites that used different cutoffs.
NMOL_L_TO_NG_ML = 3.02
DEFAULT_RESPONDER_THRESHOLD = 0.2 * NMOL_L_TO_NG_ML      # 0.604 ng/mL


def load_site(site, panel, *, cohort_id=None, tidy=None, aa=None, demo=None,
              cpeptide=None, arms=None, arm_subjects=None, **cohort_kwargs):
    """Load one study + panel -> (frame, feature_names, target).

    Two sources, same return shape. ``cohort_id`` reads a CloudOS Cohort Browser
    selection (concept ids arrive already mapped); otherwise the explicit file
    paths are read, which is what keeps the whole pipeline runnable locally.
    """
    if cohort_id:
        from . import cohort_data as cd      # optional dependency — import only if used
        return cd.load_cohort(cohort_id, panel, **cohort_kwargs)
    return od.load_features(site, panel, tidy=tidy, aa=aa, demo=demo,
                            cpeptide=cpeptide, arms=arms, arm_subjects=arm_subjects)


def read_feature_list(features):
    """Read a feature set -> (feats, source_basename, source_tag).

    Accepts either a CSV with a ``feature`` column or a JSON feature-space
    document (``{"features": [{"concept_id": ..., "feature": ...}, ...]}``) as
    written by ``consensus-features --emit-json``, so the same option serves the
    local and federated paths. ``source_tag`` is the leading token of the
    filename (e.g. SDY524), used to stamp every fit output.
    """
    src = os.path.basename(str(features))
    tag = os.path.splitext(src)[0].split("_")[0]
    if str(features).endswith(".json"):
        with open(features) as fh:
            doc = json.load(fh)
        feats = [f.get("feature") or str(f.get("concept_id")) for f in doc["features"]]
    else:
        feats = list(pd.read_csv(features)["feature"])
    return feats, src, tag


def stem(site, panel, source_tag):
    """The `<site>_from-<src>_panel<X>` filename stem shared by every fit output."""
    return f"{site}_from-{source_tag}_panel{panel.upper()}"


def design_matrix(frame, feats):
    """Reindex the site frame to feats, fill missing with 0 -> float ndarray."""
    return frame.reindex(columns=feats).fillna(0.0).astype(float).values


def kfold(n, seed):
    """5-fold (fewer for tiny studies) shuffled KFold."""
    return KFold(n_splits=min(5, max(2, n // 2)), shuffle=True, random_state=seed)


def cv_predict(build_model, X, y, kf):
    """Out-of-fold predictions, a fresh model per fold, scaled within the fold."""
    pred = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        sc = MinMaxScaler().fit(X[tr])
        m = build_model().fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return pred


def cv_predict_proba(build_model, X, y_bin, skf):
    """Out-of-fold P(responder), a fresh classifier per fold, scaled within the fold.

    The classifier twin of ``cv_predict`` — same fold-local scaling discipline, but
    returning ``predict_proba``'s positive-class column, which is what ROC AUC needs.
    """
    pred = np.full(len(y_bin), np.nan)
    for tr, te in skf.split(X, y_bin):
        sc = MinMaxScaler().fit(X[tr])
        m = build_model().fit(sc.transform(X[tr]), y_bin[tr])
        pred[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return pred


def r2(y, p):
    m = ~np.isnan(p); yy, pp = y[m], p[m]
    rss = float(np.sum((yy - pp) ** 2)); tss = float(np.sum((yy - yy.mean()) ** 2))
    return 1.0 - rss / tss if tss > 0 else float("nan")


def mse(y, p):
    m = ~np.isnan(p)
    return float(np.mean((y[m] - p[m]) ** 2))


def bootstrap_r2_ci(y, p, n_boot, seed):
    m = ~np.isnan(p); yy, pp = y[m], p[m]
    rng = np.random.default_rng(seed); n = len(yy); out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n); ys, ps = yy[idx], pp[idx]
        tss = float(np.sum((ys - ys.mean()) ** 2))
        out.append(1.0 - float(np.sum((ys - ps) ** 2)) / tss if tss > 0 else np.nan)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


# ------------------------------------------------------------------ discrimination
def c_index(y, p):
    """Harrell's C-index on a continuous outcome — the regression analogue of ROC AUC.

    Over every comparable pair of subjects, the fraction the model orders the way
    the truth does (ties in the prediction count a half). Runs 0.5 (no better than
    chance) to 1.0, and reads exactly as ROC AUC does — the probability that a
    randomly chosen higher-outcome subject is predicted higher than a lower one.

    This is NOT ROC AUC: ROC AUC needs a binary label, which a regression on a
    continuous target does not have. Reported as ``c_index`` throughout so the two
    are never confused.
    """
    m = ~np.isnan(p); yy, pp = np.asarray(y)[m], np.asarray(p)[m]
    if len(yy) < 2:
        return float("nan")
    # Pairwise over the upper triangle; the cohorts here are small enough that the
    # O(n^2) form is cheaper than sorting-based counting and far easier to read.
    dy = yy[:, None] - yy[None, :]
    dp = pp[:, None] - pp[None, :]
    comparable = dy > 0                      # each unordered pair counted once
    if not comparable.any():
        return float("nan")
    concordant = (dp[comparable] > 0).sum() + 0.5 * (dp[comparable] == 0).sum()
    return float(concordant / comparable.sum())


def bootstrap_ci(stat, y, p, n_boot, seed):
    """Percentile bootstrap CI for any statistic of (y, pred).

    Resamples subjects with replacement. A resample the statistic is undefined on
    (a single class, no comparable pair) yields NaN and drops out via
    ``np.nanpercentile`` rather than biasing the interval.
    """
    m = ~np.isnan(p); yy, pp = np.asarray(y)[m], np.asarray(p)[m]
    rng = np.random.default_rng(seed); n = len(yy); out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            out.append(stat(yy[idx], pp[idx]))
        except ValueError:
            out.append(np.nan)               # e.g. roc_auc_score on a single class
    if np.all(np.isnan(out)):
        return float("nan"), float("nan")
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def stratified_kfold(y_bin, seed):
    """Stratified folds for the classifier track.

    Plain KFold can hand a small site a fold with one class in it, which makes
    that fold's ROC AUC undefined — at n=10 with 5 responders that is a live risk,
    not a theoretical one. Stratifying keeps both classes in every fold. The split
    count is bounded by the rarer class, since a fold cannot hold more folds than
    it has members of it.
    """
    minority = int(min(np.sum(y_bin == 0), np.sum(y_bin == 1)))
    n_splits = max(2, min(5, minority))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# ------------------------------------------------------------------ preprocessing
def trim_outliers(frame, target, *, lower_pct=5, upper_pct=95, on="target", feats=None):
    """Drop rows outside the [lower, upper] percentile range. Returns (frame, info).

    Off by default at every call site — trimming is an option, never a dictate.
    Percentiles are computed **within this site only** (a pooled percentile would
    leak across the federation) and **once, before cross-validation**, so every
    fold sees the same population.

    ``on`` selects what is trimmed: ``target`` (the default — trimming twelve
    features independently would discard far more rows than intended),
    ``features``, or ``both``.
    """
    n_before = len(frame)
    keep = pd.Series(True, index=frame.index)

    cols = []
    if on in ("target", "both"):
        cols.append(target)
    if on in ("features", "both"):
        cols += [c for c in (feats or []) if c in frame.columns]

    for c in cols:
        v = pd.to_numeric(frame[c], errors="coerce")
        lo, hi = v.quantile(lower_pct / 100.0), v.quantile(upper_pct / 100.0)
        keep &= v.between(lo, hi)

    out = frame.loc[keep].reset_index(drop=True)
    return out, {"enabled": True, "lower_pct": lower_pct, "upper_pct": upper_pct,
                 "applied_to": on, "n_before": n_before, "n_after": len(out),
                 "n_excluded": n_before - len(out)}


NO_TRIM = {"enabled": False}
