"""
Phase 3 (site): this site's own outcome using the federated results.

There is no single global 'aggregated result' — each site produces its own
site-specific outcome, and the federated coefficient vector (and RF union) are
the channel that carries the aggregated information here. For each of Ridge /
LASSO / RF this compares the site's SOLO model (5-fold CV) against the FEDERATED
model (the aggregated vector applied as-is; for RF, the average of the union
forests), with bootstrap 95% CIs. The graphic is drawn by plot.solo_vs_federated.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from . import common_utils as cu
from . import plot
from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")


def _fed_linear(X, y, kf, c_coef, c_int):
    """Apply the aggregated linear vector as-is to each held-out fold."""
    pred = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        sc = MinMaxScaler().fit(X[tr])
        pred[te] = sc.transform(X[te]) @ c_coef + c_int
    return pred


def _fed_rf(frame, forests):
    """Average the union forests, each applied with its own scaler and features."""
    preds = []
    for fd in forests:
        Xi = frame.reindex(columns=fd["features"]).fillna(0.0).astype(float).values
        preds.append(fd["forest"].predict(fd["scaler"].transform(Xi)))
    return np.mean(preds, axis=0)


def _linear_job(method, path):
    vec = pd.read_csv(path)
    m = (method or (vec["method"].iloc[0] if "method" in vec.columns else "ridge")).lower()
    cd = dict(zip(vec["feature"], vec["coefficient"]))
    c_int = float(cd.pop("__intercept__", 0.0))
    feats = [f for f in vec["feature"] if f != "__intercept__"]
    coef = np.array([float(cd[f]) for f in feats])
    return {"kind": "linear", "method": m, "feats": feats, "coef": coef, "intercept": c_int,
            "source": os.path.basename(str(path)),
            "aggregation": str(vec["aggregation"].iloc[0]) if "aggregation" in vec.columns else "",
            "mode": str(vec["mode"].iloc[0]) if "mode" in vec.columns else "",
            "sites": str(vec["sites"].iloc[0]) if "sites" in vec.columns else ""}


def _rf_job(path):
    with open(path, "rb") as fh:
        u = pickle.load(fh)
    forests = u.get("forests", [])
    feats = list(forests[0]["features"]) if forests else []
    n_trees = int(getattr(forests[0]["forest"], "n_estimators", 200)) if forests else 200
    return {"kind": "rf", "method": "rf", "forests": forests, "feats": feats, "n_trees": n_trees,
            "source": os.path.basename(str(path)),
            "aggregation": str(u.get("aggregation", "union")),
            "mode": str(u.get("mode", "")),
            "sites": ";".join(u.get("sites", []))}


def _jobs(panel, coefficients, coefficients_dir, method, from_features):
    """Assemble method jobs from a single vector/pickle or a directory of federated
    results (ridge + lasso vectors + rf union), scoped by panel/from."""
    jobs = []
    if coefficients_dir:
        parts = []
        if from_features:
            parts.append(f"from-{from_features}")
        if panel:
            parts.append(f"panel{panel.upper()}")
        prefix = "federated" + ("_" + "_".join(parts) if parts else "")
        for meth in ("ridge", "lasso"):
            f = sorted(glob.glob(os.path.join(coefficients_dir, f"{prefix}_{meth}_*_vector.csv")))
            if f:
                jobs.append(_linear_job(meth, f[0]))
        rf = sorted(glob.glob(os.path.join(coefficients_dir, f"{prefix}_rf_union.pkl")))
        if rf:
            jobs.append(_rf_job(rf[0]))
    elif coefficients:
        jobs.append(_rf_job(coefficients) if str(coefficients).endswith(".pkl")
                    else _linear_job(method, coefficients))
    return jobs


def apply_coefficients(site, panel="B", coefficients=None, coefficients_dir=None, data_root=".",
                       method=None, from_features=None, ridge_alpha=1.0, lasso_alpha=0.008,
                       n_boot=2000, outdir=".", seed=42):
    """Produce this site's own outcome (solo vs federated) across the methods found."""
    frame, _all, target = cu.load_site(site, panel, data_root)
    y = frame[target].astype(float).values
    n = len(y)
    p = panel.upper()
    os.makedirs(outdir, exist_ok=True)

    jobs = _jobs(p, coefficients, coefficients_dir, method, from_features)
    if not jobs:
        raise SystemExit("No federated results found. Pass --coefficients <vector.csv|rf_union.pkl> "
                         "or --coefficients-dir <dir> (scope with --panel/--from).")

    kf = cu.kfold(n, seed)
    results = []
    for job in jobs:
        mname = job["method"]
        X = cu.design_matrix(frame, job["feats"])
        if job["kind"] == "linear":
            build = ((lambda: Lasso(alpha=lasso_alpha, max_iter=50000)) if mname == "lasso"
                     else (lambda: Ridge(alpha=ridge_alpha)))
            solo = cu.cv_predict(build, X, y, kf)
            fed = _fed_linear(X, y, kf, job["coef"], job["intercept"])
        else:
            nt = job["n_trees"]
            solo = cu.cv_predict(lambda: RandomForestRegressor(n_estimators=nt, min_samples_leaf=2,
                                                               n_jobs=1, random_state=seed), X, y, kf)
            fed = _fed_rf(frame, job["forests"])
        r2s = cu.r2(y, solo); cis = cu.bootstrap_r2_ci(y, solo, n_boot, seed)
        r2f = cu.r2(y, fed);  cif = cu.bootstrap_r2_ci(y, fed, n_boot, seed)
        results.append({"method": mname, "solo": solo, "fed": fed, "r2_solo": r2s, "ci_solo": cis,
                        "r2_fed": r2f, "ci_fed": cif, "n_features": len(job["feats"]),
                        "source": job["source"], "aggregation": job["aggregation"],
                        "mode": job["mode"], "sites": job["sites"]})
        logger.info(f"{site} {mname}: solo R2={r2s:+.3f}  federated R2={r2f:+.3f}  "
                    f"({'improves' if r2f > r2s else 'no gain'})  [{job['mode']}: {job['sites']}]")

    pd.DataFrame([{"site": site, "panel": p, "method": r["method"], "n_subjects": n,
                   "n_features": r["n_features"],
                   "r2_solo": r["r2_solo"], "r2_solo_lo": r["ci_solo"][0], "r2_solo_hi": r["ci_solo"][1],
                   "r2_federated": r["r2_fed"], "r2_fed_lo": r["ci_fed"][0], "r2_fed_hi": r["ci_fed"][1],
                   "coefficients_source": r["source"], "aggregation": r["aggregation"],
                   "mode": r["mode"], "aggregated_sites": r["sites"]} for r in results]).to_csv(
        os.path.join(outdir, f"{site}_panel{p}_federated_metrics.csv"), index=False)

    pred_cols = {"y_true": y}
    for r in results:
        pred_cols[f"{r['method']}_solo"] = r["solo"]
        pred_cols[f"{r['method']}_federated"] = r["fed"]
    pd.DataFrame(pred_cols).to_csv(
        os.path.join(outdir, f"{site}_panel{p}_federated_predictions.csv"), index=False)

    plot.solo_vs_federated(site, p, y, results,
                           os.path.join(outdir, f"{site}_panel{p}_federated"),
                           sites_label=results[0]["sites"])
    logger.info(f"Wrote {site}_panel{p}_federated_metrics.csv and {site}_panel{p}_federated.(png|svg|html)")
