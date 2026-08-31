"""The JSON exchange: payload shape, disclosure, forest fidelity, and the round trip."""

import json
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from oadr_cpep import aggregate, common_utils as cu, forest, payload as pl


# --------------------------------------------------------------------- fixtures
def _site_doc(site, coefs, *, iteration=1, algorithm="ridge", concept_ids=None):
    entries = [pl.coefficient(f, v, **({"concept_id": concept_ids[f],
                                        "domain_id": "Measurement",
                                        "concept_name": f} if concept_ids else {}))
               for f, v in coefs.items()]
    return pl.site_payload(iteration=iteration, site=site, panel="B",
                           algorithm=algorithm, coefficients=entries, intercept=-1.0)


@pytest.fixture
def forest_fixture():
    rng = np.random.default_rng(0)
    feats = ["bmi", "weight_kg", "GAD65"]
    X = pd.DataFrame(rng.normal(50, 15, size=(60, 3)), columns=feats)
    y = X["bmi"] * 0.03 - X["GAD65"] * 0.01 + rng.normal(0, 0.1, 60)
    sc = MinMaxScaler().fit(X.values)
    rf = RandomForestRegressor(n_estimators=8, min_samples_leaf=2,
                               random_state=1).fit(sc.transform(X.values), y)
    return rf, sc, feats, X


# --------------------------------------------------------------------- disclosure
def test_payload_carries_no_patient_counts():
    """Subject counts stay at the site — a payload must not leak one, at any depth."""
    doc = _site_doc("SDY524", {"weight_kg": 0.8, "GAD65": -0.1})
    assert doc.pop("contains_patient_counts") is False    # the assertion itself, not a count
    assert not re.search(r"n_subjects|n_node_samples|patient_count|n_responder",
                         json.dumps(doc))


def test_counts_are_stripped_from_metrics_not_merely_omitted():
    """The logistic fit computes responder counts for its local CSV; two of them give
    away the cohort size exactly, so they must not survive into a payload."""
    doc = pl.site_payload(
        iteration=1, site="SDY524", panel="B", algorithm="logistic_l2",
        coefficients=[pl.coefficient("weight_kg", 0.5)],
        metrics={"roc_auc": 0.81, "responder_fraction": 0.36,
                 "n_responder": 26, "n_non_responder": 46})
    assert "n_responder" not in doc["metrics"] and "n_non_responder" not in doc["metrics"]
    assert doc["metrics"]["roc_auc"] == 0.81            # the science survives
    assert doc["metrics"]["responder_fraction"] == 0.36  # balance without the size


def test_strip_counts_keeps_counts_of_things_that_are_not_people():
    kept = pl.strip_counts({"n_sites": 2, "n_trees": 40, "n_estimators": 200,
                            "n_subjects": 83, "n_responder": 26})
    assert kept == {"n_sites": 2, "n_trees": 40, "n_estimators": 200}


def test_forest_nodes_carry_no_sample_counts(forest_fixture):
    rf, sc, feats, _X = forest_fixture
    doc = forest.to_json(rf, sc, feats, site="SDY524")
    assert not any("n_node_samples" in n for t in doc["trees"] for n in t["nodes"])


# --------------------------------------------------------------------- forest fidelity
def test_json_forest_reproduces_sklearn_exactly(forest_fixture):
    """The tree form must be a faithful replacement for the pickle, not an approximation."""
    rf, sc, feats, X = forest_fixture
    doc = forest.to_json(rf, sc, feats, site="SDY524")
    assert np.allclose(rf.predict(sc.transform(X.values)), forest.predict(doc, X), atol=1e-12)


def test_forest_thresholds_are_in_original_units(forest_fixture):
    """A split must read in clinical units, so a partner can compare it to their data."""
    rf, sc, feats, X = forest_fixture
    doc = forest.to_json(rf, sc, feats, site="SDY524")
    splits = [n for t in doc["trees"] for n in t["nodes"] if not n["is_leaf"]]
    assert splits and all(X[n["split_feature"]].min() <= n["threshold"] <= X[n["split_feature"]].max()
                          for n in splits)


def test_forest_union_keeps_every_tree_and_its_origin(forest_fixture):
    rf, sc, feats, _X = forest_fixture
    a = forest.to_json(rf, sc, feats, site="SDY524")
    b = forest.to_json(rf, sc, feats, site="SDY569")
    u = forest.union([a, b])
    assert u["n_trees"] == a["n_trees"] + b["n_trees"]
    assert {t["site"] for t in u["trees"]} == {"SDY524", "SDY569"}
    assert len({t["tree_id"] for t in u["trees"]}) == u["n_trees"]   # renumbered, unique


# --------------------------------------------------------------------- aggregation
def test_aggregation_increments_iteration_and_records_provenance(tmp_path):
    for site in ("SDY524", "SDY569"):
        pl.write(_site_doc(site, {"weight_kg": 0.8, "GAD65": -0.1}, iteration=3),
                 tmp_path / f"{site}.json")
    aggregate.aggregate_payloads([str(tmp_path / "SDY524.json"), str(tmp_path / "SDY569.json")],
                                 method="fedavg", outdir=str(tmp_path))

    out = json.load(open(tmp_path / "federated_panelB_ridge_iter4.json"))
    assert out["iteration"] == 4                       # one past the inputs'
    assert out["cohorts"] == ["SDY524", "SDY569"]
    assert {m["site"] for m in out["input_models"]} == {"SDY524", "SDY569"}
    assert all(m["iteration"] == 3 for m in out["input_models"])


def test_aggregation_averages_and_counts_contributors(tmp_path):
    pl.write(_site_doc("SDY524", {"weight_kg": 1.0, "bmi": 0.5}), tmp_path / "a.json")
    pl.write(_site_doc("SDY569", {"weight_kg": 0.0}), tmp_path / "b.json")
    aggregate.aggregate_payloads([str(tmp_path / "a.json"), str(tmp_path / "b.json")],
                                 outdir=str(tmp_path))
    out = json.load(open(tmp_path / "federated_panelB_ridge_iter2.json"))
    by = {c["feature"]: c for c in out["coefficients"]}
    assert by["weight_kg"]["coefficient"] == pytest.approx(0.5)      # mean of 1.0 and 0.0
    assert by["weight_kg"]["n_sites_contributing"] == 2
    assert by["bmi"]["n_sites_contributing"] == 1                    # only SDY524 had it


def test_algorithms_are_never_mixed(tmp_path):
    """Logistic betas are log-odds; averaging them with regression betas is meaningless."""
    pl.write(_site_doc("SDY524", {"weight_kg": 1.0}, algorithm="ridge"), tmp_path / "r.json")
    pl.write(_site_doc("SDY524", {"weight_kg": 9.0}, algorithm="logistic_l2"), tmp_path / "g.json")
    aggregate.aggregate_payloads([str(tmp_path / "r.json"), str(tmp_path / "g.json")],
                                 outdir=str(tmp_path))
    ridge = json.load(open(tmp_path / "federated_panelB_ridge_iter2.json"))
    logit = json.load(open(tmp_path / "federated_panelB_logistic_l2_iter2.json"))
    assert ridge["coefficients"][0]["coefficient"] == pytest.approx(1.0)
    assert logit["coefficients"][0]["coefficient"] == pytest.approx(9.0)


def test_coefficients_match_on_concept_id_not_column_name(tmp_path):
    """Two sites naming the same concept differently must still combine."""
    pl.write(_site_doc("SDY524", {"weight_kg": 1.0}, concept_ids={"weight_kg": 3025315}),
             tmp_path / "a.json")
    pl.write(_site_doc("SDY569", {"body_weight": 3.0}, concept_ids={"body_weight": 3025315}),
             tmp_path / "b.json")
    aggregate.aggregate_payloads([str(tmp_path / "a.json"), str(tmp_path / "b.json")],
                                 outdir=str(tmp_path))
    out = json.load(open(tmp_path / "federated_panelB_ridge_iter2.json"))
    assert len(out["coefficients"]) == 1
    assert out["coefficients"][0]["coefficient"] == pytest.approx(2.0)
    assert out["coefficients"][0]["concept_id"] == 3025315


def test_entry_key_prefers_concept_over_feature():
    assert pl.entry_key({"concept_id": 3025315, "feature": "weight_kg"}) == "concept:3025315"
    assert pl.entry_key({"feature": "weight_kg"}) == "feature:weight_kg"


# --------------------------------------------------------------------- metrics
def test_c_index_is_rank_based_not_roc_auc():
    y = np.array([1.0, 2, 3, 4, 5])
    assert cu.c_index(y, y) == 1.0                       # perfect ordering
    assert cu.c_index(y, -y) == 0.0                      # exactly reversed
    assert cu.c_index(y, np.array([1.0, 2, 3, 5, 4])) == pytest.approx(0.9)


def test_trim_outliers_is_off_unless_asked():
    assert cu.NO_TRIM == {"enabled": False}


def test_trim_outliers_drops_the_tails_only():
    df = pd.DataFrame({"log_auc": list(range(100)), "x": list(range(100))})
    out, info = cu.trim_outliers(df, "log_auc", lower_pct=5, upper_pct=95)
    assert info["n_excluded"] == len(df) - len(out) > 0
    assert out["log_auc"].min() >= 4 and out["log_auc"].max() <= 95


def test_stratified_kfold_keeps_both_classes_in_every_fold():
    """At n=10 with 5 responders, plain KFold can hand a fold one class; this must not."""
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    X = np.arange(20, dtype=float).reshape(10, 2)
    for _tr, te in cu.stratified_kfold(y, 42).split(X, y):
        assert len(set(y[te])) == 2
