"""
Shared low-level helpers for the oadr-cpep steps: data loading, within-site
scaling, cross-validation, and metrics. No step logic and no plotting live here
(plotting is in plot.py).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from . import oadr_data as od


def load_site(site, panel, *, tidy=None, aa=None, demo=None,
              cpeptide=None, arms=None, arm_subjects=None):
    """Load one study + panel from explicit file paths -> (frame, feature_names, target)."""
    return od.load_features(site, panel, tidy=tidy, aa=aa, demo=demo,
                            cpeptide=cpeptide, arms=arms, arm_subjects=arm_subjects)


def read_feature_list(features):
    """Read a feature-list CSV (column 'feature') -> (feats, source_basename, source_tag).

    ``source_tag`` is the leading token of the filename (e.g. SDY524), used to
    stamp every fit output so you can see which feature set it was fit on.
    """
    feats = list(pd.read_csv(features)["feature"])
    src = os.path.basename(str(features))
    tag = os.path.splitext(src)[0].split("_")[0]
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
