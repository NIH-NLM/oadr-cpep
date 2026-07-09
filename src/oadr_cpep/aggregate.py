"""
Aggregator steps: build the consensus feature set and combine the per-site
coefficient vectors / forests.

  consensus_features : Phase 1 — multi-site tally, or one site's selection (--from-site).
  aggregate_vectors  : Phase 2 — FedAvg / median / mean of the vectors + union of forests.
                       Detects solo vs federated; --panel and --from scope which
                       vectors combine so panels and feature sources never mix.

Only site-level parameters (feature lists, coefficient vectors, forests) are read.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np
import pandas as pd

from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")


# --------------------------------------------------------------- consensus (Phase 1)
def consensus_features(input_dir, min_sites=None, outdir=".", panel=None, from_site=None):
    """Tally per-site ``*_selected_features.csv`` into a consensus feature set.

    Args:
        input_dir: directory (searched recursively) with the selected-features files.
        min_sites: keep a feature selected by >= this many sites (default: majority).
        outdir: output directory.
        panel: restrict to one panel (A|B); avoids mixing panels in one dir.
        from_site: use this site's selection AS the consensus (single-site, bespoke).
    """
    tag = f"panel{panel.upper()}" if panel else ""
    sel_glob = f"*_{tag}_selected_features.csv" if panel else "*_selected_features.csv"
    cons_name = f"consensus_{tag}_features.csv" if panel else "consensus_features.csv"
    tally_name = f"feature_selection_tally_{tag}.csv" if panel else "feature_selection_tally.csv"

    files = sorted(glob.glob(os.path.join(input_dir, "**", sel_glob), recursive=True))
    if not files:
        raise SystemExit(f"No {sel_glob} under {input_dir}")
    os.makedirs(outdir, exist_ok=True)

    if from_site:
        match = [f for f in files if os.path.basename(f).startswith(f"{from_site}_")]
        if not match:
            raise SystemExit(f"No selected-features file for site {from_site!r} under {input_dir}")
        d = pd.read_csv(match[0])
        chosen = d.loc[d["selected"] == 1, "feature"] if "selected" in d.columns else d["feature"]
        consensus = sorted(chosen)
        pd.DataFrame({"feature": consensus}).to_csv(os.path.join(outdir, cons_name), index=False)
        logger.info(f"consensus from site {from_site} (single-site, bespoke) "
                    f"({len(consensus)}) -> {cons_name}: {consensus}")
        return

    counts, sites = {}, []
    for f in files:
        d = pd.read_csv(f)
        sites.append(d["site"].iloc[0] if "site" in d.columns else os.path.basename(f))
        chosen = d.loc[d["selected"] == 1, "feature"] if "selected" in d.columns else d["feature"]
        for feat in chosen:
            counts[feat] = counts.get(feat, 0) + 1
    n = len(files)
    thr = min_sites if min_sites is not None else (n // 2 + 1)
    consensus = sorted(f for f, c in counts.items() if c >= thr)
    pd.DataFrame({"feature": consensus}).to_csv(os.path.join(outdir, cons_name), index=False)
    tally = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]),
                         columns=["feature", "n_sites_selected"])
    tally["kept"] = (tally["n_sites_selected"] >= thr).astype(int)
    tally.to_csv(os.path.join(outdir, tally_name), index=False)
    logger.info(f"{n} sites {sites}, panel {panel.upper() if panel else 'all'}, threshold {thr}")
    logger.info(f"consensus features ({len(consensus)}) -> {cons_name}: {consensus}")


# --------------------------------------------------------------- aggregate (Phase 2)
def _scope(panel, from_features):
    parts = []
    if from_features:
        parts.append(f"from-{from_features}")
    if panel:
        parts.append(f"panel{panel.upper()}")
    return "_".join(parts)


def _find(input_dir, tail, panel, from_features):
    """Glob per-site files ending in ``tail`` (e.g. 'ridge_vector.csv'), scoped by
    feature source and/or panel."""
    if from_features and panel:
        pat = f"*_from-{from_features}_panel{panel.upper()}_{tail}"
    elif panel:
        pat = f"*_panel{panel.upper()}_{tail}"
    elif from_features:
        pat = f"*_from-{from_features}_*_{tail}"
    else:
        pat = f"*_{tail}"
    return sorted(glob.glob(os.path.join(input_dir, "**", pat), recursive=True))


def aggregate_vectors(input_dir, method="fedavg", outdir=".", panel=None, from_features=None):
    """Combine the site coefficient vectors (FedAvg / median / mean) and forests.

    Args:
        input_dir: directory (searched recursively) with per-site vectors / forests.
        method: vector combine rule — ``fedavg`` (weighted by n_subjects), ``median``, ``mean``.
        outdir: output directory.
        panel: restrict to one panel (A|B).
        from_features: restrict to vectors fit on one feature source (e.g. SDY524).
    """
    scope = _scope(panel, from_features)
    fed_prefix = "federated" + (f"_{scope}" if scope else "")
    os.makedirs(outdir, exist_ok=True)

    for meth in ("ridge", "lasso"):
        files = _find(input_dir, f"{meth}_vector.csv", panel, from_features)
        if not files:
            continue
        series, sizes, contrib = [], [], []
        for f in files:
            d = pd.read_csv(f)
            contrib.append(str(d["site"].iloc[0]) if "site" in d.columns else os.path.basename(f))
            d = d.set_index("feature")
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
        mode = "solo" if len(set(contrib)) == 1 else "federated"
        out = pd.DataFrame({"feature": allfeats, "coefficient": agg})
        out["method"] = meth
        out["aggregation"] = method
        if panel:
            out["panel"] = panel.upper()
        if from_features:
            out["features_source_site"] = from_features
        out["n_sites"] = len(set(contrib))
        out["sites"] = ";".join(sorted(set(contrib)))
        out["mode"] = mode
        out_name = f"{fed_prefix}_{meth}_{method}_vector.csv"
        out.to_csv(os.path.join(outdir, out_name), index=False)
        logger.info(f"Aggregated {len(files)} {meth} vector(s) [{mode}] by {method} "
                    f"from {sorted(set(contrib))} -> {out_name}")

    rf_files = _find(input_dir, "rf.pkl", panel, from_features)
    if rf_files:
        forests, rf_sites = [], []
        for f in rf_files:
            with open(f, "rb") as fh:
                fd = pickle.load(fh)
            forests.append(fd)
            rf_sites.append(str(fd.get("site", os.path.basename(f))))
        mode = "solo" if len(set(rf_sites)) == 1 else "federated"
        rf_name = f"{fed_prefix}_rf_union.pkl"
        with open(os.path.join(outdir, rf_name), "wb") as fh:
            pickle.dump({"forests": forests, "aggregation": "union", "mode": mode,
                         "sites": sorted(set(rf_sites))}, fh)
        logger.info(f"Union of {len(rf_files)} forest(s) [{mode}] "
                    f"from {sorted(set(rf_sites))} -> {rf_name}")
