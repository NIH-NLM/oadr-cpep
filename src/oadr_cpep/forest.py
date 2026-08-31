"""
Random Forests as data, not as pickles.

A pickled scikit-learn forest is tied to the exact library that wrote it — the
forests already in this repo were written under scikit-learn 1.7.2 and warn when
read under 1.8.0 — and it is opaque to any partner who will not execute a
stranger's pickle. Neither is acceptable for something that has to move between
workflows.

So a forest travels as its trees: one row per node, with the split feature, the
threshold, and the leaf value. Two properties make that enough.

**Thresholds are stored in original units.** The forest is fit on MinMax-scaled
features, but MinMax scaling is monotonic, so ``x_scaled <= t_scaled`` exactly
when ``x_raw <= t_raw``. Back-transforming each threshold once, at write time,
means the receiving site traverses its own untouched data and needs neither the
scaler nor the same feature ordering. It also makes a split readable — a weight
split reads as 63.4 kg rather than 0.41.

**Splits are named, not indexed.** Each node names its feature (and its
``concept_id`` where the data came in mapped), so a union of trees from sites
that fit different feature sets still traverses correctly.

No node carries a sample count. ``n_node_samples`` would be a patient count, and
patient counts do not cross the boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _threshold_original(scaler, j, scaled):
    """The split point in the site's own units: undo the MinMaxScaler on one feature."""
    s = float(scaler.scale_[j])
    if s == 0 or not np.isfinite(s):
        return float(scaled)
    return (float(scaled) - float(scaler.min_[j])) / s


def tree_to_nodes(tree, feats, scaler, concepts=None):
    """One scikit-learn tree -> a list of node dicts."""
    concepts = concepts or {}
    idx = {name: j for j, name in enumerate(feats)}
    t = tree.tree_
    parent = {0: None}
    nodes = []
    for n in range(t.node_count):
        left, right = int(t.children_left[n]), int(t.children_right[n])
        is_leaf = left == -1
        if not is_leaf:
            parent[left] = parent[right] = n
        node = {"node_id": n, "parent_node_id": parent.get(n),
                "is_leaf": bool(is_leaf),
                "left_child_node_id": None if is_leaf else left,
                "right_child_node_id": None if is_leaf else right,
                "predicted_value": float(t.value[n].ravel()[0])}
        if not is_leaf:
            feature = feats[t.feature[n]]
            node["split_feature"] = feature
            node["threshold"] = _threshold_original(scaler, idx[feature],
                                                    float(t.threshold[n]))
            cid = concepts.get(feature, {}).get("concept_id")
            if cid not in (None, "", 0):
                node["split_concept_id"] = int(cid)
                node["split_domain_id"] = concepts[feature].get("domain_id", "")
        nodes.append(node)
    return nodes


def to_json(rf, scaler, feats, *, site=None, concepts=None):
    """A fitted RandomForestRegressor -> the portable forest document."""
    return {
        "n_trees": len(rf.estimators_),
        "features": list(feats),
        "threshold_units": "original",
        "trees": [{"tree_id": i, "site": site,
                   "nodes": tree_to_nodes(est, feats, scaler, concepts)}
                  for i, est in enumerate(rf.estimators_)],
    }


def union(docs):
    """Combine several forest documents into one, keeping each tree's origin.

    The federated Random Forest is the union of the site forests — every tree
    kept, none averaged — so a prediction is the mean over all of them. Tree ids
    are renumbered across the union; ``site`` on each tree preserves provenance.
    """
    trees, feats = [], []
    for doc in docs:
        for t in doc.get("trees", []):
            trees.append({**t, "tree_id": len(trees)})
        for f in doc.get("features", []):
            if f not in feats:
                feats.append(f)
    return {"n_trees": len(trees), "features": feats,
            "threshold_units": "original", "trees": trees}


def _predict_tree(nodes, frame):
    """Traverse one tree for every row, in numpy. Returns a prediction per row.

    Rows walk the tree together: at each step the still-active rows at a node are
    split by that node's test and pushed to its children. A feature the local
    frame does not have is treated as 0.0, matching ``design_matrix``.
    """
    by_id = {n["node_id"]: n for n in nodes}
    out = np.full(len(frame), np.nan)
    # (node_id, row positions currently at that node)
    stack = [(0, np.arange(len(frame)))]
    while stack:
        nid, rows = stack.pop()
        if len(rows) == 0:
            continue
        node = by_id[nid]
        if node["is_leaf"]:
            out[rows] = node["predicted_value"]
            continue
        col = node["split_feature"]
        v = (pd.to_numeric(frame[col], errors="coerce").fillna(0.0).values
             if col in frame.columns else np.zeros(len(frame)))
        go_left = v[rows] <= node["threshold"]
        stack.append((node["left_child_node_id"], rows[go_left]))
        stack.append((node["right_child_node_id"], rows[~go_left]))
    return out


def predict(doc, frame):
    """Mean prediction over every tree in the document, on the frame's own units.

    No scikit-learn estimator is reconstructed and no scaler is needed: thresholds
    are already in original units, so the local data is traversed as it stands.
    """
    trees = doc.get("trees", [])
    if not trees:
        raise SystemExit("forest document contains no trees")
    preds = np.array([_predict_tree(t["nodes"], frame) for t in trees])
    return preds.mean(axis=0)
