"""
Coordinator (aggregator) steps of the federated pipeline.

  consensus_features : Phase 1 — tally the per-site feature selections and keep
                       the features chosen by >= a threshold number of sites.
  aggregate_vectors  : Phase 2 — combine the per-site coefficient vectors
                       (FedAvg weighted by n_subjects, or median / mean) and
                       build a union-of-forests ensemble from the site forests.

Only site-level model parameters (feature lists, coefficient vectors, trained
forests) are read — never subject-level data.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np
import pandas as pd

from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")


def consensus_features(input_dir, min_sites=None, outdir="."):
    """Phase 1: keep the features selected by >= a threshold number of sites.

    Args:
        input_dir: directory (searched recursively) holding the per-site
            ``*_selected_features.csv`` files.
        min_sites: keep a feature selected by at least this many sites; defaults
            to a simple majority (``n_sites // 2 + 1``).
        outdir: output directory for ``consensus_features.csv`` and
            ``feature_selection_tally.csv``.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*_selected_features.csv"),
                             recursive=True))
    if not files:
        raise SystemExit(f"No *_selected_features.csv under {input_dir}")
    counts, sites = {}, []
    for f in files:
        d = pd.read_csv(f)
        sites.append(d["site"].iloc[0] if "site" in d.columns else os.path.basename(f))
        for feat in d.loc[d["selected"] == 1, "feature"]:
            counts[feat] = counts.get(feat, 0) + 1
    n = len(files)
    thr = min_sites if min_sites is not None else (n // 2 + 1)
    consensus = sorted(f for f, c in counts.items() if c >= thr)
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame({"feature": consensus}).to_csv(
        os.path.join(outdir, "consensus_features.csv"), index=False)
    tally = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]),
                         columns=["feature", "n_sites_selected"])
    tally["kept"] = (tally["n_sites_selected"] >= thr).astype(int)
    tally.to_csv(os.path.join(outdir, "feature_selection_tally.csv"), index=False)
    logger.info(f"{n} sites {sites}, threshold {thr}")
    logger.info(f"consensus features ({len(consensus)}): {consensus}")


def aggregate_vectors(input_dir, method="fedavg", outdir="."):
    """Phase 2: combine the per-site coefficient vectors and forests.

    Args:
        input_dir: directory (searched recursively) holding the per-site
            ``*_ridge_vector.csv`` / ``*_lasso_vector.csv`` and ``*_rf.pkl``.
        method: vector combine rule — ``fedavg`` (weighted by ``n_subjects``),
            ``median``, or ``mean``.
        outdir: output directory for ``federated_<method_name>_<method>_vector.csv``
            and ``federated_rf_union.pkl``.
    """
    os.makedirs(outdir, exist_ok=True)
    for meth in ("ridge", "lasso"):
        files = sorted(glob.glob(os.path.join(input_dir, "**", f"*_{meth}_vector.csv"),
                                 recursive=True))
        if not files:
            continue
        series, sizes = [], []
        for f in files:
            d = pd.read_csv(f).set_index("feature")
            series.append(d["coefficient"])
            sizes.append(int(d["n_subjects"].iloc[0]) if "n_subjects" in d.columns else 1)
        allfeats = sorted(set().union(*[set(s.index) for s in series]))
        M = np.array([[s.get(f, 0.0) for f in allfeats] for s in series])
        sizes = np.array(sizes)
        if method == "fedavg":
            agg = np.average(M, axis=0, weights=sizes)
        elif method == "median":
            agg = np.median(M, axis=0)
        else:
            agg = M.mean(axis=0)
        out = pd.DataFrame({"feature": allfeats, "coefficient": agg})
        out["method"] = meth
        out["aggregation"] = method
        out["n_sites"] = len(files)
        out.to_csv(os.path.join(outdir, f"federated_{meth}_{method}_vector.csv"), index=False)
        logger.info(f"Aggregated {len(files)} {meth} vectors by {method}")

    rf_files = sorted(glob.glob(os.path.join(input_dir, "**", "*_rf.pkl"), recursive=True))
    if rf_files:
        forests = []
        for f in rf_files:
            with open(f, "rb") as fh:
                forests.append(pickle.load(fh))
        with open(os.path.join(outdir, "federated_rf_union.pkl"), "wb") as fh:
            pickle.dump({"forests": forests, "aggregation": "union"}, fh)
        logger.info(f"Union of {len(rf_files)} forests -> federated_rf_union.pkl")
