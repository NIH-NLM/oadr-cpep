"""
Phase 2 (site): fit the analytical methods on a given feature set.

One function per process — fit_ridge, fit_lasso, fit_rf — plus fit_models, the
convenience that runs all three. Each fits its model on all rows (the coefficient
vector / forest that goes to the aggregator), then evaluates the site's solo
performance by 5-fold CV, writes a metrics CSV, prints R²/MSE, and draws its
graphic (via plot.py). Every output is stamped with the feature source
(`from-<src>`) so you can see what it was fit on.
"""
from __future__ import annotations

import os
import pickle

import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from . import common_utils as cu
from . import plot
from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")


def _write_linear_vector(path, feats, model, site, panel, n, method, alpha, source):
    """Write a linear coefficient vector (with __intercept__) + provenance columns."""
    rows = [{"feature": "__intercept__", "coefficient": float(model.intercept_)}]
    rows += [{"feature": f, "coefficient": float(c)} for f, c in zip(feats, model.coef_)]
    vec = pd.DataFrame(rows)
    vec["site"] = site
    vec["panel"] = panel
    vec["n_subjects"] = n
    vec["method"] = method
    vec["alpha"] = alpha
    vec["features_source"] = source
    vec.to_csv(path, index=False)


def _write_metrics(path, site, panel, method, feats, n, mse, r2, ci, source, extra=None):
    row = {"site": site, "panel": panel, "method": method, "n_subjects": n,
           "n_features": len(feats), "mse": mse, "r2": r2, "r2_lo": ci[0], "r2_hi": ci[1],
           "features_source": source, "features": ";".join(feats)}
    if extra:
        row.update(extra)
    pd.DataFrame([row]).to_csv(path, index=False)


def fit_ridge(site, panel="B", features=None, data_root=".", outdir=".",
              alpha=1.0, n_boot=2000, seed=42):
    """Fit Ridge(alpha) on the feature set -> coefficient vector, CV metrics, graphic."""
    frame, _all, target = cu.load_site(site, panel, data_root)
    feats, src, tag = cu.read_feature_list(features)
    p = panel.upper(); stem = cu.stem(site, panel, tag)
    y = frame[target].astype(float).values
    X = cu.design_matrix(frame, feats)
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"{site} panel {p}: Ridge(alpha={alpha}) on {len(feats)} features from {src}: {feats}")

    sc = MinMaxScaler().fit(X)
    m = Ridge(alpha=alpha).fit(sc.transform(X), y)
    _write_linear_vector(os.path.join(outdir, f"{stem}_ridge_vector.csv"),
                         feats, m, site, p, len(y), "ridge", alpha, src)
    logger.info(f"Wrote {stem}_ridge_vector.csv")

    pred = cu.cv_predict(lambda: Ridge(alpha=alpha), X, y, cu.kfold(len(y), seed))
    mse, r2 = cu.mse(y, pred), cu.r2(y, pred); ci = cu.bootstrap_r2_ci(y, pred, n_boot, seed)
    _write_metrics(os.path.join(outdir, f"{stem}_ridge_fit_metrics.csv"),
                   site, p, "ridge", feats, len(y), mse, r2, ci, src, {"alpha": alpha})
    plot.scatter(y, pred, f"RIDGE — {site} panel {p} (5-fold CV)\nR²={r2:+.2f}  MSE={mse:.3f}  features: {src}",
                 os.path.join(outdir, f"{stem}_ridge_fit"))
    logger.info(f"  ridge: 5-fold CV  MSE={mse:.3f}  R2={r2:+.3f}  -> {stem}_ridge_fit.(png|svg|html)")


def fit_lasso(site, panel="B", features=None, data_root=".", outdir=".",
              alpha=0.008, n_boot=2000, seed=42):
    """Fit Lasso(alpha) on the feature set -> coefficient vector, CV metrics, graphic."""
    frame, _all, target = cu.load_site(site, panel, data_root)
    feats, src, tag = cu.read_feature_list(features)
    p = panel.upper(); stem = cu.stem(site, panel, tag)
    y = frame[target].astype(float).values
    X = cu.design_matrix(frame, feats)
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"{site} panel {p}: Lasso(alpha={alpha}) on {len(feats)} features from {src}: {feats}")

    sc = MinMaxScaler().fit(X)
    m = Lasso(alpha=alpha, max_iter=50000).fit(sc.transform(X), y)
    _write_linear_vector(os.path.join(outdir, f"{stem}_lasso_vector.csv"),
                         feats, m, site, p, len(y), "lasso", alpha, src)
    logger.info(f"Wrote {stem}_lasso_vector.csv")

    pred = cu.cv_predict(lambda: Lasso(alpha=alpha, max_iter=50000), X, y, cu.kfold(len(y), seed))
    mse, r2 = cu.mse(y, pred), cu.r2(y, pred); ci = cu.bootstrap_r2_ci(y, pred, n_boot, seed)
    _write_metrics(os.path.join(outdir, f"{stem}_lasso_fit_metrics.csv"),
                   site, p, "lasso", feats, len(y), mse, r2, ci, src, {"alpha": alpha})
    plot.scatter(y, pred, f"LASSO — {site} panel {p} (5-fold CV)\nR²={r2:+.2f}  MSE={mse:.3f}  features: {src}",
                 os.path.join(outdir, f"{stem}_lasso_fit"))
    logger.info(f"  lasso: 5-fold CV  MSE={mse:.3f}  R2={r2:+.3f}  -> {stem}_lasso_fit.(png|svg|html)")


def fit_rf(site, panel="B", features=None, data_root=".", outdir=".",
           n_trees=200, n_boot=2000, seed=42):
    """Fit a Random Forest on the feature set -> forest pickle, CV metrics, graphic."""
    frame, _all, target = cu.load_site(site, panel, data_root)
    feats, src, tag = cu.read_feature_list(features)
    p = panel.upper(); stem = cu.stem(site, panel, tag)
    y = frame[target].astype(float).values
    X = cu.design_matrix(frame, feats)
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"{site} panel {p}: RandomForest({n_trees} trees) on {len(feats)} features from {src}: {feats}")

    sc = MinMaxScaler().fit(X)
    rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2,
                               n_jobs=1, random_state=seed).fit(sc.transform(X), y)
    with open(os.path.join(outdir, f"{stem}_rf.pkl"), "wb") as fh:
        pickle.dump({"forest": rf, "scaler": sc, "features": feats, "site": site,
                     "panel": p, "n_subjects": len(y), "features_source": src}, fh)
    logger.info(f"Wrote {stem}_rf.pkl  ({n_trees} trees on {len(feats)} features)")

    pred = cu.cv_predict(
        lambda: RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2,
                                      n_jobs=1, random_state=seed), X, y, cu.kfold(len(y), seed))
    mse, r2 = cu.mse(y, pred), cu.r2(y, pred); ci = cu.bootstrap_r2_ci(y, pred, n_boot, seed)
    _write_metrics(os.path.join(outdir, f"{stem}_rf_fit_metrics.csv"),
                   site, p, "rf", feats, len(y), mse, r2, ci, src, {"n_trees": n_trees})
    plot.scatter(y, pred, f"RF — {site} panel {p} (5-fold CV)\nR²={r2:+.2f}  MSE={mse:.3f}  features: {src}",
                 os.path.join(outdir, f"{stem}_rf_fit"))
    logger.info(f"  rf: 5-fold CV  MSE={mse:.3f}  R2={r2:+.3f}  -> {stem}_rf_fit.(png|svg|html)")


def fit_models(site, panel="B", features=None, data_root=".", outdir=".",
               ridge_alpha=1.0, lasso_alpha=0.008, n_trees=200, n_boot=2000, seed=42):
    """Convenience: run fit_ridge, fit_lasso, fit_rf on the same feature set."""
    fit_ridge(site, panel, features, data_root, outdir, ridge_alpha, n_boot, seed)
    fit_lasso(site, panel, features, data_root, outdir, lasso_alpha, n_boot, seed)
    fit_rf(site, panel, features, data_root, outdir, n_trees, n_boot, seed)
