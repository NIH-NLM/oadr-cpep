"""
Phase 2 (site): fit the analytical methods on a given feature set.

One function per process — fit_ridge, fit_lasso, fit_rf, fit_logistic — plus
fit_models, the convenience that runs them. Each fits its model on all rows (the
coefficient vector / forest that goes to the aggregator), then evaluates the
site's solo performance by cross-validation, writes a metrics CSV, prints the
scores, draws its graphic (via plot.py), and — with ``--emit-json`` — writes the
exchange payload the aggregator workflow consumes. Every output is stamped with
the feature source (`from-<src>`) so you can see what it was fit on, and with the
``iteration`` so an improved model is never confused with the one it came from.

Four metrics, and they are not interchangeable:

    mse, r2      regression error and explained variance, as before
    c_index      Harrell's concordance — the ranking quality of a continuous
                 prediction. Reads like ROC AUC (0.5 = chance, 1.0 = perfect) but
                 is NOT ROC AUC, which needs a binary label a regression lacks.
    roc_auc      only from fit_logistic, which dichotomises the outcome at the
                 responder threshold and so has a label to sweep.

The regression and logistic coefficients are different scales — betas on log
C-peptide versus log-odds of being a responder — and must never be averaged
together. They travel as separate payloads with distinct ``algorithm`` values.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from . import common_utils as cu
from . import forest as forest_mod
from . import payload as pl
from . import plot
from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")


def _write_linear_vector(path, feats, model, site, panel, n, method, alpha, source, iteration):
    """Write a linear coefficient vector (with __intercept__) + provenance columns."""
    # Regressors give a scalar intercept, classifiers a length-1 array.
    rows = [{"feature": "__intercept__", "coefficient": float(np.ravel(model.intercept_)[0])}]
    rows += [{"feature": f, "coefficient": float(c)} for f, c in zip(feats, np.ravel(model.coef_))]
    vec = pd.DataFrame(rows)
    vec["site"] = site
    vec["panel"] = panel
    vec["n_subjects"] = n
    vec["method"] = method
    vec["alpha"] = alpha
    vec["features_source"] = source
    vec["iteration"] = iteration
    vec.to_csv(path, index=False)


def _write_metrics(path, site, panel, method, feats, n, metrics, source, iteration, extra=None):
    row = {"site": site, "panel": panel, "method": method, "n_subjects": n,
           "n_features": len(feats), "iteration": iteration,
           "features_source": source, "features": ";".join(feats)}
    row.update({k: v for k, v in metrics.items() if not isinstance(v, (list, tuple))})
    for k, v in metrics.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            row[f"{k}_lo"], row[f"{k}_hi"] = v
    if extra:
        row.update(extra)
    pd.DataFrame([row]).to_csv(path, index=False)


def _prepare(site, panel, features, files, trim):
    """Load, optionally trim, and build (frame, feats, X, y, src, tag, trim_info).

    Trimming happens here — once, on the whole site frame, before any fold is cut —
    so every fold sees the same population, and percentiles are this site's own.
    """
    frame, _all, target = cu.load_site(site, panel, **files)
    feats, src, tag = cu.read_feature_list(features)

    trim_info = cu.NO_TRIM
    if trim and trim.get("enabled"):
        frame, trim_info = cu.trim_outliers(
            frame, target, lower_pct=trim["lower_pct"], upper_pct=trim["upper_pct"],
            on=trim["on"], feats=feats)
        logger.info(f"  outlier trim [{trim['lower_pct']}-{trim['upper_pct']}%] on "
                    f"{trim['on']}: {trim_info['n_before']} -> {trim_info['n_after']} rows "
                    f"({trim_info['n_excluded']} excluded)")

    y = frame[target].astype(float).values
    X = cu.design_matrix(frame, feats)
    return frame, feats, X, y, src, tag, trim_info, target


def _regression_metrics(y, pred, n_boot, seed):
    """MSE, R2 and C-index with bootstrap CIs. ROC AUC is null here, with the reason."""
    return {
        "mse": cu.mse(y, pred),
        "r2": cu.r2(y, pred),
        "r2_ci": list(cu.bootstrap_r2_ci(y, pred, n_boot, seed)),
        "c_index": cu.c_index(y, pred),
        "c_index_ci": list(cu.bootstrap_ci(cu.c_index, y, pred, n_boot, seed)),
        "roc_auc": None,
        "roc_auc_reason": "regression on a continuous outcome has no binary label; "
                          "see the logistic payload for ROC AUC",
    }


def _emit_json(path, *, iteration, site, panel, algorithm, feats, coefs, intercept,
               hyperparameters, trim_info, metrics, src, cohort_id=None, forest=None,
               concepts=None):
    """Write the exchange payload for one fitted model."""
    cmap = concepts or {}
    entries = [pl.coefficient(f, c, **cmap.get(f, {})) for f, c in zip(feats, coefs)]
    doc = pl.site_payload(
        iteration=iteration, site=site, panel=panel, algorithm=algorithm,
        coefficients=entries, intercept=intercept, cohort_id=cohort_id,
        hyperparameters=hyperparameters,
        preprocessing={"outlier_trim": trim_info,
                       "feature_scaling": "MinMaxScaler (refit at application site)"},
        metrics=metrics, features_source=src, forest=forest)
    pl.write(doc, path)
    logger.info(f"Wrote {os.path.basename(path)} (iteration {iteration})")


def _files_kwargs(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id):
    return {"tidy": tidy, "aa": aa, "demo": demo, "cpeptide": cpeptide,
            "arms": arms, "arm_subjects": arm_subjects, "cohort_id": cohort_id}


# ------------------------------------------------------------------ linear regressors
def _fit_linear(kind, site, panel, features, files, outdir, alpha, n_boot, seed,
                iteration, emit_json, trim, concepts):
    """Shared body for Ridge and LASSO — identical but for the estimator and label."""
    build = ((lambda: Lasso(alpha=alpha, max_iter=50000)) if kind == "lasso"
             else (lambda: Ridge(alpha=alpha)))
    frame, feats, X, y, src, tag, trim_info, _t = _prepare(site, panel, features, files, trim)
    concepts = concepts or frame.attrs.get("concepts", {})
    p = panel.upper(); stem = cu.stem(site, panel, tag)
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"{site} panel {p}: {kind.upper()}(alpha={alpha}) on {len(feats)} "
                f"features from {src} [iteration {iteration}]: {feats}")

    sc = MinMaxScaler().fit(X)
    m = build().fit(sc.transform(X), y)
    _write_linear_vector(os.path.join(outdir, f"{stem}_{kind}_vector.csv"),
                         feats, m, site, p, len(y), kind, alpha, src, iteration)
    logger.info(f"Wrote {stem}_{kind}_vector.csv")

    pred = cu.cv_predict(build, X, y, cu.kfold(len(y), seed))
    metrics = _regression_metrics(y, pred, n_boot, seed)
    _write_metrics(os.path.join(outdir, f"{stem}_{kind}_fit_metrics.csv"),
                   site, p, kind, feats, len(y), metrics, src, iteration, {"alpha": alpha})
    plot.scatter(y, pred,
                 f"{kind.upper()} — {site} panel {p} (5-fold CV)\n"
                 f"R²={metrics['r2']:+.2f}  MSE={metrics['mse']:.3f}  "
                 f"C={metrics['c_index']:.2f}  features: {src}",
                 os.path.join(outdir, f"{stem}_{kind}_fit"))
    logger.info(f"  {kind}: CV  MSE={metrics['mse']:.3f}  R2={metrics['r2']:+.3f}  "
                f"C-index={metrics['c_index']:.3f}")

    if emit_json:
        _emit_json(os.path.join(outdir, f"{stem}_{kind}_iter{iteration}.json"),
                   iteration=iteration, site=site, panel=p, algorithm=kind,
                   feats=feats, coefs=np.ravel(m.coef_), intercept=float(m.intercept_),
                   hyperparameters={"alpha": alpha, "penalty": "L1" if kind == "lasso" else "L2",
                                    "seed": seed},
                   trim_info=trim_info, metrics=metrics, src=src,
                   cohort_id=files.get("cohort_id"), concepts=concepts)


def fit_ridge(site, panel="B", features=None, *, tidy=None, aa=None, demo=None,
              cpeptide=None, arms=None, arm_subjects=None, cohort_id=None, outdir=".",
              alpha=1.0, n_boot=2000, seed=42, iteration=1, emit_json=False,
              trim=None, concepts=None):
    """Fit Ridge(alpha) on the feature set -> coefficient vector, CV metrics, graphic."""
    _fit_linear("ridge", site, panel, features,
                _files_kwargs(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id),
                outdir, alpha, n_boot, seed, iteration, emit_json, trim, concepts)


def fit_lasso(site, panel="B", features=None, *, tidy=None, aa=None, demo=None,
              cpeptide=None, arms=None, arm_subjects=None, cohort_id=None, outdir=".",
              alpha=0.008, n_boot=2000, seed=42, iteration=1, emit_json=False,
              trim=None, concepts=None):
    """Fit Lasso(alpha) on the feature set -> coefficient vector, CV metrics, graphic."""
    _fit_linear("lasso", site, panel, features,
                _files_kwargs(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id),
                outdir, alpha, n_boot, seed, iteration, emit_json, trim, concepts)


# ------------------------------------------------------------------ random forest
def fit_rf(site, panel="B", features=None, *, tidy=None, aa=None, demo=None,
           cpeptide=None, arms=None, arm_subjects=None, cohort_id=None, outdir=".",
           n_trees=200, n_boot=2000, seed=42, iteration=1, emit_json=False,
           trim=None, concepts=None):
    """Fit a Random Forest on the feature set -> forest, CV metrics, graphic.

    The forest is written twice: the legacy ``.pkl`` for backward compatibility,
    and — with ``--emit-json`` — the tree structure as data, which is what the
    aggregator should consume. A pickle is tied to the exact scikit-learn that
    wrote it; the JSON form is not.
    """
    files = _files_kwargs(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id)
    frame, feats, X, y, src, tag, trim_info, _t = _prepare(site, panel, features, files, trim)
    concepts = concepts or frame.attrs.get("concepts", {})
    p = panel.upper(); stem = cu.stem(site, panel, tag)
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"{site} panel {p}: RandomForest({n_trees} trees) on {len(feats)} "
                f"features from {src} [iteration {iteration}]: {feats}")

    build = lambda: RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2,
                                          n_jobs=1, random_state=seed)
    sc = MinMaxScaler().fit(X)
    rf = build().fit(sc.transform(X), y)
    with open(os.path.join(outdir, f"{stem}_rf.pkl"), "wb") as fh:
        pickle.dump({"forest": rf, "scaler": sc, "features": feats, "site": site,
                     "panel": p, "n_subjects": len(y), "features_source": src,
                     "iteration": iteration}, fh)
    logger.info(f"Wrote {stem}_rf.pkl  ({n_trees} trees on {len(feats)} features)")

    pred = cu.cv_predict(build, X, y, cu.kfold(len(y), seed))
    metrics = _regression_metrics(y, pred, n_boot, seed)
    _write_metrics(os.path.join(outdir, f"{stem}_rf_fit_metrics.csv"),
                   site, p, "rf", feats, len(y), metrics, src, iteration, {"n_trees": n_trees})
    plot.scatter(y, pred,
                 f"RF — {site} panel {p} (5-fold CV)\n"
                 f"R²={metrics['r2']:+.2f}  MSE={metrics['mse']:.3f}  "
                 f"C={metrics['c_index']:.2f}  features: {src}",
                 os.path.join(outdir, f"{stem}_rf_fit"))
    logger.info(f"  rf: CV  MSE={metrics['mse']:.3f}  R2={metrics['r2']:+.3f}  "
                f"C-index={metrics['c_index']:.3f}")

    if emit_json:
        _emit_json(os.path.join(outdir, f"{stem}_rf_iter{iteration}.json"),
                   iteration=iteration, site=site, panel=p, algorithm="rf",
                   feats=feats, coefs=rf.feature_importances_, intercept=0.0,
                   hyperparameters={"n_estimators": n_trees, "min_samples_leaf": 2,
                                    "criterion": rf.criterion, "seed": seed},
                   trim_info=trim_info, metrics=metrics, src=src,
                   cohort_id=cohort_id, concepts=concepts,
                   forest=forest_mod.to_json(rf, sc, feats, site=site, concepts=concepts))


# ------------------------------------------------------------------ logistic (ROC AUC)
def _logistic_builder(penalty, C, seed):
    """A LogisticRegression factory that works either side of the scikit-learn 1.8 change.

    1.8 deprecated ``penalty=`` in favour of ``l1_ratio`` (removal in 1.10), so
    prefer the new spelling — ``l1_ratio=0`` is pure L2, ``1`` is pure L1 — and
    fall back to ``penalty=`` on the older releases that do not accept it. Probing
    once here keeps the deprecation out of every fold's fit.
    """
    solver = "saga" if penalty == "l1" else "lbfgs"
    kwargs = {"C": C, "solver": solver, "max_iter": 50000, "random_state": seed}
    l1_ratio = 1.0 if penalty == "l1" else 0.0
    try:
        LogisticRegression(l1_ratio=l1_ratio, **kwargs).get_params()
        return lambda: LogisticRegression(l1_ratio=l1_ratio, **kwargs)
    except TypeError:                        # scikit-learn < 1.8
        return lambda: LogisticRegression(penalty=penalty, **kwargs)



def fit_logistic(site, panel="B", features=None, *, tidy=None, aa=None, demo=None,
                 cpeptide=None, arms=None, arm_subjects=None, cohort_id=None, outdir=".",
                 penalty="l2", alpha=1.0, n_boot=2000, seed=42, iteration=1,
                 emit_json=False, trim=None, concepts=None,
                 responder_threshold=cu.DEFAULT_RESPONDER_THRESHOLD):
    """Fit a logistic classifier of responder status -> ROC AUC.

    This is the only model in the package with a genuine ROC AUC, because it is
    the only one with a binary label. The outcome is dichotomised at
    ``responder_threshold`` — a **setup-time** parameter, never prompted for —
    against raw C-peptide AUC in ng/mL, defaulting to the conventional preserved
    beta-cell function cutoff of 0.2 nmol/L.

    ``penalty`` mirrors the regressors: ``l2`` is the Ridge analogue, ``l1`` the
    LASSO analogue, and the regularisation maps straight off the same option via
    ``C = 1/alpha``, so the two tracks cannot drift apart.

    Its coefficients are log-odds and must never be averaged with the regression
    betas; the payload's ``algorithm`` is ``logistic_l1``/``logistic_l2`` so the
    aggregator keeps them apart.
    """
    files = _files_kwargs(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id)
    frame, feats, X, y_log, src, tag, trim_info, _t = _prepare(site, panel, features, files, trim)
    concepts = concepts or frame.attrs.get("concepts", {})
    p = panel.upper(); stem = cu.stem(site, panel, tag)
    algorithm = f"logistic_{penalty}"
    os.makedirs(outdir, exist_ok=True)

    # The target is stored log-transformed; the threshold is stated in raw ng/mL,
    # so compare on the raw scale rather than asking the caller to pre-log it.
    y_raw = np.exp(y_log)
    y_bin = (y_raw >= responder_threshold).astype(int)
    n_pos, n_neg = int(y_bin.sum()), int(len(y_bin) - y_bin.sum())
    logger.info(f"{site} panel {p}: {algorithm.upper()} on {len(feats)} features from {src} "
                f"[iteration {iteration}]; responder >= {responder_threshold:.3f} ng/mL "
                f"-> {n_pos} responder / {n_neg} non-responder")

    if n_pos == 0 or n_neg == 0:
        # Reported, not faked: with only one class there is no ROC curve to
        # integrate. Emit the reason and stop rather than invent a number.
        reason = (f"all {len(y_bin)} subjects fall on one side of the responder "
                  f"threshold {responder_threshold:.3f} ng/mL "
                  f"({n_pos} responder / {n_neg} non-responder)")
        logger.warning(f"  {algorithm}: ROC AUC undefined — {reason}")
        metrics = {"roc_auc": None, "roc_auc_reason": reason,
                   "responder_threshold_ng_ml": responder_threshold,
                   "n_responder": n_pos, "n_non_responder": n_neg}
        _write_metrics(os.path.join(outdir, f"{stem}_{algorithm}_fit_metrics.csv"),
                       site, p, algorithm, feats, len(y_bin), metrics, src, iteration)
        return

    C = 1.0 / alpha if alpha else 1.0
    build = _logistic_builder(penalty, C, seed)

    sc = MinMaxScaler().fit(X)
    m = build().fit(sc.transform(X), y_bin)
    _write_linear_vector(os.path.join(outdir, f"{stem}_{algorithm}_vector.csv"),
                         feats, m, site, p, len(y_bin), algorithm, alpha, src, iteration)

    proba = cu.cv_predict_proba(build, X, y_bin, cu.stratified_kfold(y_bin, seed))
    auc = float(roc_auc_score(y_bin, proba))
    metrics = {
        "roc_auc": auc,
        "roc_auc_ci": list(cu.bootstrap_ci(
            lambda a, b: roc_auc_score(a, b), y_bin.astype(float), proba, n_boot, seed)),
        "responder_threshold_ng_ml": responder_threshold,
        "responder_threshold_nmol_l": responder_threshold / cu.NMOL_L_TO_NG_ML,
        # The fraction goes out; the raw counts stay in the local CSV, since the
        # two of them together would give away the cohort size.
        "responder_fraction": n_pos / (n_pos + n_neg),
        "n_responder": n_pos, "n_non_responder": n_neg,
    }
    _write_metrics(os.path.join(outdir, f"{stem}_{algorithm}_fit_metrics.csv"),
                   site, p, algorithm, feats, len(y_bin), metrics, src, iteration,
                   {"alpha": alpha, "penalty": penalty})
    logger.info(f"  {algorithm}: CV  ROC AUC={auc:.3f} "
                f"[{metrics['roc_auc_ci'][0]:.3f}, {metrics['roc_auc_ci'][1]:.3f}]")

    if emit_json:
        _emit_json(os.path.join(outdir, f"{stem}_{algorithm}_iter{iteration}.json"),
                   iteration=iteration, site=site, panel=p, algorithm=algorithm,
                   feats=feats, coefs=np.ravel(m.coef_), intercept=float(m.intercept_[0]),
                   hyperparameters={"alpha": alpha, "C": C, "penalty": penalty.upper(),
                                    "seed": seed,
                                    "responder_threshold_ng_ml": responder_threshold},
                   trim_info=trim_info, metrics=metrics, src=src,
                   cohort_id=cohort_id, concepts=concepts)


# ------------------------------------------------------------------ convenience
def fit_models(site, panel="B", features=None, *, tidy=None, aa=None, demo=None,
               cpeptide=None, arms=None, arm_subjects=None, cohort_id=None, outdir=".",
               ridge_alpha=1.0, lasso_alpha=0.008, n_trees=200, n_boot=2000, seed=42,
               iteration=1, emit_json=False, trim=None, concepts=None,
               with_logistic=False, responder_threshold=cu.DEFAULT_RESPONDER_THRESHOLD):
    """Convenience: run fit_ridge, fit_lasso, fit_rf on the same feature set.

    ``with_logistic`` adds the classifier track, which is what produces ROC AUC.
    """
    common = dict(tidy=tidy, aa=aa, demo=demo, cpeptide=cpeptide, arms=arms,
                  arm_subjects=arm_subjects, cohort_id=cohort_id, outdir=outdir,
                  n_boot=n_boot, seed=seed, iteration=iteration, emit_json=emit_json,
                  trim=trim, concepts=concepts)
    fit_ridge(site, panel, features, alpha=ridge_alpha, **common)
    fit_lasso(site, panel, features, alpha=lasso_alpha, **common)
    fit_rf(site, panel, features, n_trees=n_trees, **common)
    if with_logistic:
        fit_logistic(site, panel, features, penalty="l2", alpha=ridge_alpha,
                     responder_threshold=responder_threshold, **common)
