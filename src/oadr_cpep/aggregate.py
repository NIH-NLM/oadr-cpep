"""
Coordinator (aggregator) steps of the federated pipeline.

  consensus_features : Phase 1 — tally the per-site feature selections and keep
                       the features chosen by >= a threshold number of sites.
  aggregate_vectors  : Phase 2 — combine the per-site coefficient vectors
                       (FedAvg weighted by n_subjects, or median / mean) and
                       build a union-of-forests ensemble from the site forests.

Both take an optional ``panel`` (A|B): when given, only files for that panel are
considered and the outputs are panel-tagged, so Panel A and Panel B runs can
share one directory without mixing. Only site-level model parameters (feature
lists, coefficient vectors, trained forests) are read — never subject-level data.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np
import pandas as pd

from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")


def _panel_tag(panel):
    """Return the ``panel<X>`` infix (e.g. 'panelB') or '' when panel is None."""
    return f"panel{panel.upper()}" if panel else ""


def consensus_features(input_dir, min_sites=None, outdir=".", panel=None, from_site=None):
    """Phase 1: build the feature set to fit on.

    By default this keeps the features selected by >= a threshold number of
    sites. When ``from_site`` is given it instead uses that one site's selection
    AS the consensus — a bespoke, single-site choice, honest when the result is
    driven by one dominant study rather than a genuine multi-site agreement.

    Args:
        input_dir: directory (searched recursively) holding the per-site
            ``*_selected_features.csv`` files.
        min_sites: keep a feature selected by at least this many sites; defaults
            to a simple majority (``n_sites // 2 + 1``). Ignored with ``from_site``.
        outdir: output directory.
        panel: restrict to one panel (``A`` or ``B``); avoids mixing panels when
            both are in ``input_dir``. Outputs are panel-tagged when given.
        from_site: use this site's selection as the consensus (single-site).
    """
    tag = _panel_tag(panel)
    sel_glob = f"*_{tag}_selected_features.csv" if panel else "*_selected_features.csv"
    cons_name = f"consensus_{tag}_features.csv" if panel else "consensus_features.csv"
    tally_name = f"feature_selection_tally_{tag}.csv" if panel else "feature_selection_tally.csv"

    files = sorted(glob.glob(os.path.join(input_dir, "**", sel_glob), recursive=True))
    if not files:
        raise SystemExit(f"No {sel_glob} under {input_dir}")
    os.makedirs(outdir, exist_ok=True)

    if from_site:
        # Bespoke: one site's selection IS the consensus (not a multi-site tally).
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
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame({"feature": consensus}).to_csv(os.path.join(outdir, cons_name), index=False)
    tally = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]),
                         columns=["feature", "n_sites_selected"])
    tally["kept"] = (tally["n_sites_selected"] >= thr).astype(int)
    tally.to_csv(os.path.join(outdir, tally_name), index=False)
    logger.info(f"{n} sites {sites}, panel {panel.upper() if panel else 'all'}, threshold {thr}")
    logger.info(f"consensus features ({len(consensus)}) -> {cons_name}: {consensus}")


def aggregate_vectors(input_dir, method="fedavg", outdir=".", panel=None):
    """Phase 2: combine the per-site coefficient vectors and forests.

    Args:
        input_dir: directory (searched recursively) holding the per-site
            ``*_ridge_vector.csv`` / ``*_lasso_vector.csv`` and ``*_rf.pkl``.
        method: vector combine rule — ``fedavg`` (weighted by ``n_subjects``),
            ``median``, or ``mean``.
        outdir: output directory.
        panel: restrict to one panel (``A`` or ``B``); outputs are panel-tagged
            when given.
    """
    tag = _panel_tag(panel)
    fed_infix = f"{tag}_" if panel else ""            # federated_panelB_ridge_...  vs  federated_ridge_...
    os.makedirs(outdir, exist_ok=True)

    for meth in ("ridge", "lasso"):
        vec_glob = f"*_{tag}_{meth}_vector.csv" if panel else f"*_{meth}_vector.csv"
        files = sorted(glob.glob(os.path.join(input_dir, "**", vec_glob), recursive=True))
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
        if panel:
            out["panel"] = panel.upper()
        out["n_sites"] = len(files)
        out_name = f"federated_{fed_infix}{meth}_{method}_vector.csv"
        out.to_csv(os.path.join(outdir, out_name), index=False)
        logger.info(f"Aggregated {len(files)} {meth} vectors by {method} -> {out_name}")

    rf_glob = f"*_{tag}_rf.pkl" if panel else "*_rf.pkl"
    rf_files = sorted(glob.glob(os.path.join(input_dir, "**", rf_glob), recursive=True))
    if rf_files:
        forests = []
        for f in rf_files:
            with open(f, "rb") as fh:
                forests.append(pickle.load(fh))
        rf_name = f"federated_{fed_infix}rf_union.pkl"
        with open(os.path.join(outdir, rf_name), "wb") as fh:
            pickle.dump({"forests": forests, "aggregation": "union"}, fh)
        logger.info(f"Union of {len(rf_files)} forests -> {rf_name}")
