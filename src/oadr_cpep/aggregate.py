"""
Aggregator steps: build the consensus feature set and combine the per-site
coefficient vectors / forests.

  consensus_features : Phase 1 — multi-site tally, or one site's selection (--from-site).
  aggregate_vectors  : Phase 2 — FedAvg / median / mean of the vectors + union of forests.

Both take their inputs as EXPLICIT files (no directory, no glob). Each vector /
selection file carries its own provenance columns (panel, features_source, site),
so the output naming and the solo-vs-federated mode are derived from the files
themselves — panels / feature sources are never mixed (that is an error).

Only site-level parameters (feature lists, coefficient vectors, forests) are read.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")


# --------------------------------------------------------------- consensus (Phase 1)
def consensus_features(features, min_sites=None, from_site=None, outdir="."):
    """Tally the given per-site selected-features CSVs into a consensus feature set.

    Args:
        features: list of explicit ``*_selected_features.csv`` file paths.
        min_sites: keep a feature selected by >= this many sites (default: majority).
        from_site: use this site's selection AS the consensus (single-site, bespoke).
        outdir: output directory.
    """
    files = [str(f) for f in features]
    if not files:
        raise SystemExit("No --features files given.")
    os.makedirs(outdir, exist_ok=True)

    # derive the panel tag from the files themselves (no flags, no assumptions)
    panels = set()
    for f in files:
        d = pd.read_csv(f)
        if "panel" in d.columns and len(d):
            panels.add(str(d["panel"].iloc[0]).upper())
    if len(panels) > 1:
        raise SystemExit(f"input files mix panels {sorted(panels)} — pass one panel's selections")
    tag = f"panel{next(iter(panels))}" if panels else ""
    cons_name = f"consensus_{tag}_features.csv" if tag else "consensus_features.csv"
    tally_name = f"feature_selection_tally_{tag}.csv" if tag else "feature_selection_tally.csv"

    def _chosen(d):
        return d.loc[d["selected"] == 1, "feature"] if "selected" in d.columns else d["feature"]

    if from_site:
        match = [f for f in files if os.path.basename(f).startswith(f"{from_site}_")]
        if not match:
            raise SystemExit(f"No selected-features file for site {from_site!r} in the given files")
        consensus = sorted(_chosen(pd.read_csv(match[0])))
        pd.DataFrame({"feature": consensus}).to_csv(os.path.join(outdir, cons_name), index=False)
        logger.info(f"consensus from site {from_site} (single-site, bespoke) "
                    f"({len(consensus)}) -> {cons_name}: {consensus}")
        return

    counts, sites = {}, []
    for f in files:
        d = pd.read_csv(f)
        sites.append(d["site"].iloc[0] if "site" in d.columns else os.path.basename(f))
        for feat in _chosen(d):
            counts[feat] = counts.get(feat, 0) + 1
    n = len(files)
    thr = min_sites if min_sites is not None else (n // 2 + 1)
    consensus = sorted(f for f, c in counts.items() if c >= thr)
    pd.DataFrame({"feature": consensus}).to_csv(os.path.join(outdir, cons_name), index=False)
    tally = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]),
                         columns=["feature", "n_sites_selected"])
    tally["kept"] = (tally["n_sites_selected"] >= thr).astype(int)
    tally.to_csv(os.path.join(outdir, tally_name), index=False)
    logger.info(f"{n} sites {sites}, panel {next(iter(panels)) if panels else 'all'}, threshold {thr}")
    logger.info(f"consensus features ({len(consensus)}) -> {cons_name}: {consensus}")


# --------------------------------------------------------------- aggregate (Phase 2)
def _src_tag(features_source):
    """The leading token of a features-source filename (e.g. SDY524, consensus)."""
    return str(features_source).split("_")[0] if features_source else ""


def aggregate_vectors(vectors, method="fedavg", outdir="."):
    """Combine the given per-site coefficient vectors / forests.

    Args:
        vectors: list of explicit per-site files — coefficient vector CSVs and/or
            RF ``.pkl`` forests (each is dispatched by type). Panel and feature
            source are read from the files and must be consistent.
        method: vector combine rule — ``fedavg`` (weighted by n_subjects), ``median``, ``mean``.
        outdir: output directory.
    """
    files = [str(f) for f in vectors]
    if not files:
        raise SystemExit("No --vector files given (pass the per-site vectors / forests).")
    os.makedirs(outdir, exist_ok=True)

    panels, srcs = set(), set()

    # linear coefficient vectors, grouped by their own method column
    frames = {}
    for f in (x for x in files if x.endswith(".csv")):
        d = pd.read_csv(f)
        m = str(d["method"].iloc[0]).lower() if "method" in d.columns else "ridge"
        frames.setdefault(m, []).append((f, d))
        if "panel" in d.columns and len(d):
            panels.add(str(d["panel"].iloc[0]).upper())
        if "features_source" in d.columns and len(d):
            srcs.add(_src_tag(d["features_source"].iloc[0]))

    # RF forests
    forest_dicts = []
    for f in (x for x in files if x.endswith(".pkl")):
        with open(f, "rb") as fh:
            fd = pickle.load(fh)
        forest_dicts.append((f, fd))
        if fd.get("panel"):
            panels.add(str(fd["panel"]).upper())
        if fd.get("features_source"):
            srcs.add(_src_tag(fd["features_source"]))

    if len(panels) > 1:
        raise SystemExit(f"input vectors mix panels {sorted(panels)} — pass one panel's vectors")
    if len(srcs) > 1:
        raise SystemExit(f"input vectors mix feature sources {sorted(srcs)} — pass vectors fit on one source")
    scope = []
    if srcs:
        scope.append(f"from-{next(iter(srcs))}")
    if panels:
        scope.append(f"panel{next(iter(panels))}")
    fed_prefix = "federated" + ("_" + "_".join(scope) if scope else "")

    for meth, group in frames.items():
        series, sizes, contrib = [], [], []
        for f, d in group:
            contrib.append(str(d["site"].iloc[0]) if "site" in d.columns else os.path.basename(f))
            di = d.set_index("feature")
            series.append(di["coefficient"])
            sizes.append(int(di["n_subjects"].iloc[0]) if "n_subjects" in di.columns else 1)
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
        if panels:
            out["panel"] = next(iter(panels))
        if srcs:
            out["features_source_site"] = next(iter(srcs))
        out["n_sites"] = len(set(contrib))
        out["sites"] = ";".join(sorted(set(contrib)))
        out["mode"] = mode
        out_name = f"{fed_prefix}_{meth}_{method}_vector.csv"
        out.to_csv(os.path.join(outdir, out_name), index=False)
        logger.info(f"Aggregated {len(group)} {meth} vector(s) [{mode}] by {method} "
                    f"from {sorted(set(contrib))} -> {out_name}")

    if forest_dicts:
        forests = [fd for _f, fd in forest_dicts]
        rf_sites = [str(fd.get("site", os.path.basename(f))) for f, fd in forest_dicts]
        mode = "solo" if len(set(rf_sites)) == 1 else "federated"
        rf_name = f"{fed_prefix}_rf_union.pkl"
        with open(os.path.join(outdir, rf_name), "wb") as fh:
            pickle.dump({"forests": forests, "aggregation": "union", "mode": mode,
                         "sites": sorted(set(rf_sites))}, fh)
        logger.info(f"Union of {len(forest_dicts)} forest(s) [{mode}] "
                    f"from {sorted(set(rf_sites))} -> {rf_name}")
