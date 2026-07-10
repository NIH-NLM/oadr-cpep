"""Phase 1 (site): LASSO feature selection."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler

from . import common_utils as cu
from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")


def select_features(site, panel="B", *, tidy=None, aa=None, demo=None, cpeptide=None,
                    arms=None, arm_subjects=None, outdir=".", seed=42):
    """LASSO selects features on this site's own data (alpha chosen by CV). Writes:

      <site>_panel<X>_lasso_selection.csv    full LASSO result — every candidate
                                             feature with coefficient, ``selected``
                                             (0/1), and the CV-chosen ``alpha``.
      <site>_panel<X>_selected_features.csv  only the selected features (feeds fit).
    """
    frame, feats, target = cu.load_site(site, panel, tidy=tidy, aa=aa, demo=demo,
                                        cpeptide=cpeptide, arms=arms, arm_subjects=arm_subjects)
    X = frame[feats].astype(float).values
    y = frame[target].astype(float).values

    sc = MinMaxScaler().fit(X)                     # scale within this site only
    cv = max(2, min(5, len(y) // 4))
    m = LassoCV(cv=cv, random_state=seed, max_iter=50000).fit(sc.transform(X), y)
    os.makedirs(outdir, exist_ok=True)
    p = panel.upper()

    full = pd.DataFrame({"feature": feats, "coefficient": m.coef_,
                         "selected": (np.abs(m.coef_) > 1e-8).astype(int)})
    full["site"] = site
    full["panel"] = p
    full["n_subjects"] = len(y)
    full["alpha"] = float(m.alpha_)
    full.to_csv(os.path.join(outdir, f"{site}_panel{p}_lasso_selection.csv"), index=False)

    sel = full.loc[full["selected"] == 1, ["feature", "coefficient"]].copy()
    sel["site"] = site
    sel["panel"] = p
    sel["n_subjects"] = len(y)
    sel.to_csv(os.path.join(outdir, f"{site}_panel{p}_selected_features.csv"), index=False)

    kept = list(sel["feature"])
    logger.info(f"{site} panel {p}: N={len(y)}, "
                f"selected {len(kept)}/{len(feats)} at alpha={m.alpha_:.4f}: {kept}")
    logger.info(f"  full LASSO -> {site}_panel{p}_lasso_selection.csv")
    logger.info(f"  selected   -> {site}_panel{p}_selected_features.csv  (feeds fit)")
