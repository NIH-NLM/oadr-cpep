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

from . import forest as forest_mod
from . import payload as pl
from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")


# --------------------------------------------------------------- consensus
def consensus_features(features, outdir=".", emit_json=False, iteration=1):
    """The consensus feature set = the **intersection** of the per-site selections —
    the features every site selected. ``features`` is a list of explicit
    ``*_selected_features.csv`` file paths.
    """
    files = [str(f) for f in features]
    if not files:
        raise SystemExit("No --features files given.")
    os.makedirs(outdir, exist_ok=True)

    def _chosen(d):
        return d.loc[d["selected"] == 1, "feature"] if "selected" in d.columns else d["feature"]

    panels, sites, sets, concepts = set(), [], [], {}
    for f in files:
        d = pd.read_csv(f)
        if "panel" in d.columns and len(d):
            panels.add(str(d["panel"].iloc[0]).upper())
        sites.append(str(d["site"].iloc[0]) if "site" in d.columns else os.path.basename(f))
        sets.append(set(_chosen(d)))
        if "concept_id" in d.columns:
            concepts.update({r.feature: int(r.concept_id) for r in d.itertuples()
                             if pd.notna(r.concept_id)})
    if len(panels) > 1:
        raise SystemExit(f"input files mix panels {sorted(panels)} — pass one panel's selections")
    tag = f"panel{next(iter(panels))}" if panels else ""
    cons_name = f"consensus_{tag}_features.csv" if tag else "consensus_features.csv"

    consensus = sorted(set.intersection(*sets)) if sets else []
    pd.DataFrame({"feature": consensus}).to_csv(os.path.join(outdir, cons_name), index=False)
    logger.info(f"consensus = intersection of {len(files)} sites {sites}: "
                f"{len(consensus)} features -> {cons_name}: {consensus}")

    if emit_json:
        # The feature space the coordinator sends out, carrying each feature's
        # concept id where the sites reported one.
        entries = [{"feature": f, **({"concept_id": concepts[f]} if f in concepts else {})}
                   for f in consensus]
        panel = next(iter(panels)) if panels else ""
        name = f"consensus_panel{panel}_features_iter{iteration}.json"
        pl.write(pl.feature_space_payload(iteration=iteration, panel=panel,
                                          features=entries, sites=sites),
                 os.path.join(outdir, name))
        logger.info(f"  feature space -> {name} (iteration {iteration})")


# --------------------------------------------------------------- aggregate (JSON)
def _combine(values, weights, method):
    """Combine one coefficient across the sites that carried it."""
    v = np.asarray(values, dtype=float)
    if method == "median":
        return float(np.median(v))
    if method == "fedavg" and weights is not None:
        return float(np.average(v, weights=np.asarray(weights, dtype=float)))
    return float(v.mean())


def aggregate_payloads(files, method="fedavg", outdir=".", iteration=None):
    """Combine site payloads into one federated payload per algorithm.

    Coefficients are matched on ``payload.entry_key`` — the concept id where the
    data came in mapped, the feature name where it did not — so two sites whose
    columns are named differently still combine the same variable. A coefficient
    only some sites carried is averaged over those sites, with
    ``n_sites_contributing`` recording how many.

    Algorithms are never mixed: ridge, lasso, rf and the logistic models each
    aggregate against their own kind, because their coefficients are different
    quantities (betas on log C-peptide versus log-odds of responder status).

    The output ``iteration`` is one past the inputs' — this is the next round.

    On weighting: FedAvg weights are subject counts, and subject counts stay at
    the site. Unless a payload volunteers ``n_subjects``, the combination is an
    unweighted mean and says so in the output, rather than silently claiming to
    be weighted.
    """
    docs = [(f, pl.read(f)) for f in files]
    panels = {d.get("panel", "") for _f, d in docs}
    if len(panels) > 1:
        raise SystemExit(f"input payloads mix panels {sorted(panels)} — pass one panel's payloads")
    panel = next(iter(panels), "")

    in_iters = {int(d.get("iteration", 1)) for _f, d in docs}
    out_iter = iteration if iteration is not None else max(in_iters) + 1

    by_algorithm = {}
    for f, d in docs:
        by_algorithm.setdefault(d.get("algorithm", "ridge"), []).append((f, d))

    for algorithm, group in by_algorithm.items():
        weights = [d.get("n_subjects") for _f, d in group]
        weighted = method == "fedavg" and all(w for w in weights)
        if method == "fedavg" and not weighted:
            logger.warning(f"  {algorithm}: no n_subjects in the payloads (counts stay at "
                           f"the site), so fedavg is an unweighted mean")

        # gather every coefficient by its identity across the group
        seen, order = {}, []
        for _f, d in group:
            for e in d.get("coefficients", []):
                k = pl.entry_key(e)
                if k not in seen:
                    seen[k] = {"entry": e, "values": [], "weights": [], "sites": []}
                    order.append(k)
                seen[k]["values"].append(float(e.get("coefficient", 0.0)))
                seen[k]["weights"].append(d.get("n_subjects") or 1)
                seen[k]["sites"].append(d.get("site"))

        coefficients = []
        for k in order:
            rec = seen[k]
            e = rec["entry"]
            coefficients.append(pl.coefficient(
                e.get("feature"), _combine(rec["values"],
                                           rec["weights"] if weighted else None, method),
                concept_id=e.get("concept_id"), domain_id=e.get("domain_id"),
                concept_name=e.get("concept_name"),
                n_sites_contributing=len({s for s in rec["sites"] if s})))

        intercepts = [float(d.get("intercept", 0.0)) for _f, d in group]
        w = [d.get("n_subjects") or 1 for _f, d in group] if weighted else None

        forests = [d["forest"] for _f, d in group if d.get("forest")]
        input_models = [{"site": d.get("site"), "cohort_id": d.get("cohort_id"),
                         "iteration": int(d.get("iteration", 1)),
                         "source": os.path.basename(str(f)),
                         "n_coefficients": len(d.get("coefficients", []))}
                        for f, d in group]

        doc = pl.aggregate_payload(
            iteration=out_iter, panel=panel, algorithm=algorithm,
            coefficients=coefficients, intercept=_combine(intercepts, w, method),
            rule=method if weighted else ("median" if method == "median" else "mean"),
            input_models=input_models,
            hyperparameters={"weighted_by_n_subjects": weighted},
            forest=forest_mod.union(forests) if forests else None)

        name = f"federated_panel{panel}_{algorithm}_iter{out_iter}.json"
        pl.write(doc, os.path.join(outdir, name))
        logger.info(f"Aggregated {len(group)} {algorithm} payload(s) by "
                    f"{doc['aggregation']['rule']} from {doc['cohorts']} "
                    f"-> {name} (iteration {max(in_iters)} -> {out_iter})"
                    + (f", {doc['forest']['n_trees']} trees" if doc.get("forest") else ""))


# --------------------------------------------------------------- aggregate (CSV/pkl)
def _src_tag(features_source):
    """The leading token of a features-source filename (e.g. SDY524, consensus)."""
    return str(features_source).split("_")[0] if features_source else ""


def aggregate_vectors(vectors, method="fedavg", outdir=".", iteration=None):
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

    # JSON payloads are the federated exchange format; CSV/pkl remain for local runs.
    json_files = [f for f in files if f.endswith(".json")]
    if json_files:
        if len(json_files) != len(files):
            raise SystemExit("mixed inputs: pass either JSON payloads or CSV/pkl vectors, "
                             "not both — they carry different provenance")
        return aggregate_payloads(files, method=method, outdir=outdir, iteration=iteration)

    panels, srcs = set(), set()

    # linear coefficient vectors, grouped by their own method column
    frames = {}
    for f in (x for x in files if x.endswith(".csv")):
        d = pd.read_csv(f)
        if "feature" not in d.columns or "coefficient" not in d.columns:
            continue   # not a coefficient vector (e.g. a *_fit_metrics.csv) — skip
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
