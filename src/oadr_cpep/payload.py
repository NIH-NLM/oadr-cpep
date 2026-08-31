"""
The JSON envelope that carries a model between the site and aggregator workflows.

One format for every hop of the federated round trip:

    site fit  --(site_fit)-->  aggregator  --(aggregation)-->  site apply --> site fit ...

Every document carries an ``iteration``, so a model improved over successive epochs is
never confused with the one it came from. An aggregation additionally names every input
model and every cohort behind it, because "which studies produced this vector" is the
question a federated result has to be able to answer.

**Coefficients are keyed by concept, not by column string.** Lifebit's Cohort Browser
returns data already mapped to OMOP, so a coefficient carries the ``concept_id`` and
``domain_id`` it was fit on and the aggregator matches on those — two sites whose source
data disagree still combine the same clinical entity. Reading local CSVs there are no
concept ids to carry, so an entry falls back to its ``feature`` name and matching falls
back with it. That keeps local runs honest rather than fabricating concept ids that were
never mapped; ``entry_key`` is the single place the choice is made.

**No patient counts cross the boundary.** FedAvg weights are subject counts, so they stay
at the site and never enter a document. Every payload asserts this in
``contains_patient_counts``.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

SCHEMA_VERSION = "1.0"

STAGE_SITE_FIT = "site_fit"
STAGE_AGGREGATION = "aggregation"
STAGE_FEATURE_SPACE = "feature_space"


def _plain(v):
    """numpy scalars -> JSON-native, NaN -> null (a metric that is undefined is null)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


# Keys whose values are subject counts. They are stripped from every outgoing
# document rather than trusted not to be added: a count is disclosure, and two of
# them (responders and non-responders) reconstruct the cohort size exactly. They
# stay in the local metrics CSV, which does not leave the site.
_COUNT_KEY = re.compile(r"(^|_)(n|num|count)($|_)|n_subjects|n_responder|"
                        r"n_non_responder|n_node_samples|patient", re.I)

# ...except these, which are counts of things, not of people.
_COUNT_KEY_ALLOWED = {"n_sites", "n_sites_contributing", "n_trees", "n_estimators",
                      "n_estimators_per_site", "n_coefficients", "n_nodes", "n_leaves",
                      "n_boot", "n_splits"}


def strip_counts(value):
    """Recursively drop subject counts from a document destined to leave the site."""
    if isinstance(value, dict):
        return {k: strip_counts(v) for k, v in value.items()
                if k in _COUNT_KEY_ALLOWED or not _COUNT_KEY.search(k)}
    if isinstance(value, list):
        return [strip_counts(v) for v in value]
    return value


def entry_key(entry) -> str:
    """The identity a coefficient is matched on across sites.

    ``concept_id`` when the data came in mapped; the feature name when it did not.
    Every match — aggregation, application — goes through here, so the two paths
    cannot diverge on what counts as "the same variable".
    """
    cid = entry.get("concept_id")
    if cid not in (None, "", 0):
        return f"concept:{cid}"
    return f"feature:{entry.get('feature', '')}"


def coefficient(feature, value, *, concept_id=None, domain_id=None,
                concept_name=None, **extra):
    """One coefficient entry, carrying whatever identity its source provided."""
    e = {"feature": feature, "coefficient": _plain(value)}
    if concept_id not in (None, "", 0):
        e["concept_id"] = int(concept_id)
        e["domain_id"] = domain_id or ""
        e["concept_name"] = concept_name or feature
    e.update({k: _plain(v) for k, v in extra.items()})
    return e


def outcome_block(source_value="C_Peptide_AUC_4Hrs", transform="log",
                  concept_id=None, domain_id="Measurement"):
    """What was predicted, and on what scale."""
    o = {"source_value": source_value, "transform": transform}
    if concept_id not in (None, "", 0):
        o["concept_id"] = int(concept_id)
        o["domain_id"] = domain_id
    return o


def site_payload(*, iteration, site, panel, algorithm, coefficients, intercept=0.0,
                 cohort_id=None, outcome=None, hyperparameters=None, preprocessing=None,
                 metrics=None, features_source=None, forest=None):
    """A model as fit at one site."""
    doc = {
        "schema_version": SCHEMA_VERSION,
        "iteration": int(iteration),
        "stage": STAGE_SITE_FIT,
        "site": site,
        "panel": str(panel).upper(),
        "algorithm": algorithm,
        "outcome": outcome or outcome_block(),
        "intercept": _plain(intercept),
        "coefficients": [_plain(c) for c in coefficients],
        "hyperparameters": strip_counts(_plain(hyperparameters or {})),
        "preprocessing": strip_counts(_plain(preprocessing or {})),
        "metrics": strip_counts(_plain(metrics or {})),
        "contains_patient_counts": False,
    }
    if cohort_id:
        doc["cohort_id"] = cohort_id
    if features_source:
        doc["features_source"] = features_source
    if forest is not None:
        doc["forest"] = _plain(forest)
    return doc


def aggregate_payload(*, iteration, panel, algorithm, coefficients, intercept=0.0,
                      rule="fedavg", input_models=None, cohorts=None, outcome=None,
                      hyperparameters=None, forest=None):
    """The combined model. ``iteration`` is the round this output belongs to.

    ``input_models`` and ``cohorts`` are the provenance: which site documents went
    in, at which iteration, and which studies stand behind them.
    """
    inputs = input_models or []
    return {
        "schema_version": SCHEMA_VERSION,
        "iteration": int(iteration),
        "stage": STAGE_AGGREGATION,
        "panel": str(panel).upper(),
        "algorithm": algorithm,
        "aggregation": {"rule": rule, "n_sites": len({m.get("site") for m in inputs})},
        "input_models": _plain(inputs),
        "cohorts": sorted({str(m.get("site")) for m in inputs if m.get("site")}
                          | set(cohorts or [])),
        "outcome": outcome or outcome_block(),
        "intercept": _plain(intercept),
        "coefficients": [_plain(c) for c in coefficients],
        "hyperparameters": strip_counts(_plain(hyperparameters or {})),
        "contains_patient_counts": False,
        **({"forest": _plain(forest)} if forest is not None else {}),
    }


def feature_space_payload(*, iteration, panel, features, sites=None, rule="intersection"):
    """The consensus feature set the coordinator sends out."""
    return {
        "schema_version": SCHEMA_VERSION,
        "iteration": int(iteration),
        "stage": STAGE_FEATURE_SPACE,
        "panel": str(panel).upper(),
        "selection_rule": rule,
        "cohorts": sorted(sites or []),
        "features": [_plain(f) for f in features],
        "contains_patient_counts": False,
    }


# ------------------------------------------------------------------ io
def write(doc, path):
    """Write a payload. Returns the path, for logging."""
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return path


def read(path):
    """Read a payload, rejecting anything that is not one."""
    with open(path) as fh:
        doc = json.load(fh)
    if "coefficients" not in doc and "features" not in doc:
        raise SystemExit(f"{os.path.basename(str(path))}: not an oadr-cpep payload "
                         f"(no 'coefficients' or 'features')")
    return doc


def as_vector(doc):
    """A payload -> (features, coefficients, intercept), for applying it to local data.

    Returns feature *names*, because that is what the local design matrix is keyed
    by. A document whose entries carry only concept ids yields those ids as names —
    which is correct when the local data came from the same mapped cohort, and is a
    visible mismatch rather than a silent wrong answer when it did not.
    """
    feats, coefs = [], []
    for e in doc.get("coefficients", []):
        if e.get("value_type") == "intercept":
            continue
        feats.append(e.get("feature") or str(e.get("concept_id")))
        coefs.append(float(e.get("coefficient", 0.0)))
    return feats, np.array(coefs, dtype=float), float(doc.get("intercept", 0.0))
