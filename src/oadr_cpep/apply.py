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

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from . import common_utils as cu
from . import forest as forest_mod
from . import payload as pl
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


def _without_own(items, site, key, what):
    """Drop this site's own contribution from a federated forest before scoring.

    A forest trained on this site's rows would be predicting subjects it has
    already seen, which is not an out-of-sample comparison against the solo
    model. Provenance rides along on every tree/forest, so the site's own share
    is dropped here. If nothing else remains (a union of one), there is no
    honest federated prediction to make and the leak is reported rather than
    hidden.
    """
    kept = [it for it in items if str(it.get(key, "")) != str(site)]
    if not kept:
        logger.warning(f"{site}: the federated forest holds only {site}'s own {what} — "
                       f"its federated score is NOT out-of-sample")
        return items
    if len(kept) < len(items):
        logger.info(f"{site}: held out this site's own {what} from the federated forest "
                    f"({len(items)} -> {len(kept)}) so the comparison is out-of-sample")
    return kept


def _fed_rf(frame, forests, site=None):
    """Average the union forests, each applied with its own scaler and features."""
    if site is not None:
        forests = _without_own(forests, site, "site", "forest")
    preds = []
    for fd in forests:
        Xi = frame.reindex(columns=fd["features"]).fillna(0.0).astype(float).values
        preds.append(fd["forest"].predict(fd["scaler"].transform(Xi)))
    return np.mean(preds, axis=0)


def _fed_rf_json(doc, frame, site=None):
    """Traverse a JSON forest, minus any trees this site itself contributed."""
    if site is None:
        return forest_mod.predict(doc, frame)
    trees = _without_own(doc.get("trees", []), site, "site", "trees")
    return forest_mod.predict({**doc, "trees": trees}, frame)


def _json_job(path):
    """A federated JSON payload -> a job, for either a linear model or a forest.

    This is the aggregator's own output coming back to the site for the next
    epoch. A forest arrives as trees rather than as a pickle, so it is applied by
    traversing them — no scikit-learn object is reconstructed and no scaler is
    needed, because the thresholds are already in the site's own units.
    """
    doc = pl.read(path)
    common = {"source": os.path.basename(str(path)),
              "aggregation": str(doc.get("aggregation", {}).get("rule", "")),
              "mode": doc.get("stage", ""),
              "iteration": int(doc.get("iteration", 1)),
              "sites": ";".join(doc.get("cohorts", []))}
    if doc.get("forest"):
        return {"kind": "rf_json", "method": doc.get("algorithm", "rf"),
                "forest": doc["forest"], "feats": list(doc["forest"].get("features", [])),
                "n_trees": int(doc["forest"].get("n_trees", 0)), **common}
    feats, coef, intercept = pl.as_vector(doc)
    return {"kind": "linear", "method": doc.get("algorithm", "ridge"),
            "feats": feats, "coef": coef, "intercept": intercept, **common}


def _linear_job(method, path):
    if str(path).endswith(".json"):
        return _json_job(path)
    vec = pd.read_csv(path)
    m = (method or (vec["method"].iloc[0] if "method" in vec.columns else "ridge")).lower()
    cd = dict(zip(vec["feature"], vec["coefficient"]))
    c_int = float(cd.pop("__intercept__", 0.0))
    feats = [f for f in vec["feature"] if f != "__intercept__"]
    coef = np.array([float(cd[f]) for f in feats])
    return {"kind": "linear", "method": m, "feats": feats, "coef": coef, "intercept": c_int,
            "source": os.path.basename(str(path)),
            "iteration": int(vec["iteration"].iloc[0]) if "iteration" in vec.columns else 1,
            "aggregation": str(vec["aggregation"].iloc[0]) if "aggregation" in vec.columns else "",
            "mode": str(vec["mode"].iloc[0]) if "mode" in vec.columns else "",
            "sites": str(vec["sites"].iloc[0]) if "sites" in vec.columns else ""}


def _rf_job(path):
    if str(path).endswith(".json"):
        return _json_job(path)
    with open(path, "rb") as fh:
        u = pickle.load(fh)
    forests = u.get("forests", [])
    feats = list(forests[0]["features"]) if forests else []
    n_trees = int(getattr(forests[0]["forest"], "n_estimators", 200)) if forests else 200
    return {"kind": "rf", "method": "rf", "forests": forests, "feats": feats, "n_trees": n_trees,
            "source": os.path.basename(str(path)),
            "iteration": int(u.get("iteration", 1)),
            "aggregation": str(u.get("aggregation", "union")),
            "mode": str(u.get("mode", "")),
            "sites": ";".join(u.get("sites", []))}


def apply_coefficients(site, panel="B", *, ridge_vector=None, lasso_vector=None, rf_union=None,
                       cohort_id=None, tidy=None, aa=None, demo=None, cpeptide=None,
                       arms=None, arm_subjects=None,
                       ridge_alpha=1.0, lasso_alpha=0.008, n_boot=2000, outdir=".", seed=42):
    """Produce this site's own outcome (solo vs federated) from explicit federated
    artifact files (--ridge-vector / --lasso-vector / --rf-union)."""
    frame, _all, target = cu.load_site(site, panel, cohort_id=cohort_id, tidy=tidy, aa=aa,
                                       demo=demo, cpeptide=cpeptide, arms=arms,
                                       arm_subjects=arm_subjects)
    y = frame[target].astype(float).values
    n = len(y)
    p = panel.upper()
    os.makedirs(outdir, exist_ok=True)

    jobs = []
    if ridge_vector:
        jobs.append(_linear_job("ridge", ridge_vector))
    if lasso_vector:
        jobs.append(_linear_job("lasso", lasso_vector))
    if rf_union:
        jobs.append(_rf_job(rf_union))
    if not jobs:
        raise SystemExit("No federated results given. Pass at least one of "
                         "--ridge-vector / --lasso-vector / --rf-union.")

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
            nt = max(1, job["n_trees"])
            solo = cu.cv_predict(lambda: RandomForestRegressor(n_estimators=nt, min_samples_leaf=2,
                                                               n_jobs=1, random_state=seed), X, y, kf)
            # A JSON forest is traversed as data; a pickled union still needs sklearn.
            # Either way the site's own trees are held out (see _without_own).
            fed = (_fed_rf_json(job["forest"], frame, site) if job["kind"] == "rf_json"
                   else _fed_rf(frame, job["forests"], site))
        r2s = cu.r2(y, solo); cis = cu.bootstrap_r2_ci(y, solo, n_boot, seed)
        r2f = cu.r2(y, fed);  cif = cu.bootstrap_r2_ci(y, fed, n_boot, seed)
        results.append({"method": mname, "solo": solo, "fed": fed, "r2_solo": r2s, "ci_solo": cis,
                        "r2_fed": r2f, "ci_fed": cif, "n_features": len(job["feats"]),
                        "c_solo": cu.c_index(y, solo), "c_fed": cu.c_index(y, fed),
                        "mse_solo": cu.mse(y, solo), "mse_fed": cu.mse(y, fed),
                        "iteration": job.get("iteration", 1),
                        "source": job["source"], "aggregation": job["aggregation"],
                        "mode": job["mode"], "sites": job["sites"]})
        logger.info(f"{site} {mname} [iteration {job.get('iteration', 1)}]: "
                    f"solo R2={r2s:+.3f} MSE={results[-1]['mse_solo']:.3f}  "
                    f"federated R2={r2f:+.3f} MSE={results[-1]['mse_fed']:.3f}  "
                    f"({'improves' if r2f > r2s else 'no gain'})  [{job['mode']}: {job['sites']}]")

    pd.DataFrame([{"site": site, "panel": p, "method": r["method"], "n_subjects": n,
                   "n_features": r["n_features"], "iteration": r["iteration"],
                   "mse_solo": r["mse_solo"], "mse_federated": r["mse_fed"],
                   "c_index_solo": r["c_solo"], "c_index_federated": r["c_fed"],
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
