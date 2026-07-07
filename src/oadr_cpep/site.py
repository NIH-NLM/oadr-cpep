"""
Per-site (institution) steps of the federated pipeline.

  select_features    : Phase 1 — LASSO selects features on this site's own data.
  fit_models         : Phase 2 — Ridge / LASSO / Random Forest on the consensus
                       features.
  apply_coefficients : Phase 3 — incorporate the aggregator's central federated
                       vector (solo-vs-federated CV, bootstrap 95% CI, scatter).

Data is read through :mod:`oadr_cpep.oadr_data`. Only model parameters and
scalar performance summaries leave the site — never subject-level data.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from . import oadr_data as od
from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")


def _load(site, panel, data_root):
    """Load this study + panel via oadr_data, rooted at ``data_root``."""
    od._DATA = Path(data_root)
    return od.load_features(site, panel)          # (frame, feature_names, target)


def select_features(site, panel="B", data_root=".", outdir=".", seed=42):
    """Phase 1: LASSO selects features on this site's own data.

    Writes two files:

      * ``<site>_panel<X>_lasso_selection.csv`` — the full LASSO result: every
        candidate feature with its coefficient and a ``selected`` (0/1) flag.
      * ``<site>_panel<X>_selected_features.csv`` — only the selected features
        (feature + coefficient). This is the clean list that feeds ``fit-models``
        and ``consensus-features``.

    Args:
        site: study id, e.g. ``SDY524``.
        panel: feature panel, ``A`` (legacy 9) or ``B`` (extended 12).
        data_root: directory holding the (flat) data files.
        outdir: output directory.
        seed: random seed.
    """
    frame, feats, target = _load(site, panel, data_root)
    X = frame[feats].astype(float).values
    y = frame[target].astype(float).values

    sc = MinMaxScaler().fit(X)                     # scale within this site only
    cv = max(2, min(5, len(y) // 4))
    m = LassoCV(cv=cv, random_state=seed, max_iter=50000).fit(sc.transform(X), y)
    os.makedirs(outdir, exist_ok=True)
    p = panel.upper()

    # Full LASSO feature-selection result (all candidate features).
    full = pd.DataFrame({"feature": feats, "coefficient": m.coef_,
                         "selected": (np.abs(m.coef_) > 1e-8).astype(int)})
    full["site"] = site
    full["panel"] = p
    full["n_subjects"] = len(y)
    full["alpha"] = float(m.alpha_)
    full.to_csv(os.path.join(outdir, f"{site}_panel{p}_lasso_selection.csv"), index=False)

    # Just the selected features — the input to fit-models / consensus.
    sel = full.loc[full["selected"] == 1, ["feature", "coefficient"]].copy()
    sel["site"] = site
    sel["panel"] = p
    sel["n_subjects"] = len(y)
    sel.to_csv(os.path.join(outdir, f"{site}_panel{p}_selected_features.csv"), index=False)

    kept = list(sel["feature"])
    logger.info(f"{site} panel {p}: N={len(y)}, "
                f"selected {len(kept)}/{len(feats)} at alpha={m.alpha_:.4f}: {kept}")
    logger.info(f"  full LASSO -> {site}_panel{p}_lasso_selection.csv")
    logger.info(f"  selected   -> {site}_panel{p}_selected_features.csv  (feeds fit-models)")


def fit_models(site, panel="B", features=None, data_root=".", outdir=".",
               ridge_alpha=1.0, lasso_alpha=0.008, n_trees=200, seed=42):
    """Phase 2: fit Ridge / LASSO / Random Forest on the consensus features.

    Args:
        site: study id.
        panel: feature panel ``A`` or ``B``.
        features: path to the consensus feature list CSV (column ``feature``).
        data_root: directory holding the (flat) data files.
        outdir: output directory.
        ridge_alpha: Ridge L2 penalty.
        lasso_alpha: LASSO L1 penalty.
        n_trees: number of Random Forest trees.
        seed: random seed.
    """
    frame, _all, target = _load(site, panel, data_root)
    feats = list(pd.read_csv(features)["feature"])
    features_source = os.path.basename(str(features))
    logger.info(f"{site} panel {panel.upper()}: fitting on {len(feats)} features "
                f"from {features_source}: {feats}")
    y = frame[target].astype(float).values
    X = frame.reindex(columns=feats).fillna(0.0).astype(float).values
    sc = MinMaxScaler().fit(X)
    Xs = sc.transform(X)
    os.makedirs(outdir, exist_ok=True)

    for name, model in [("ridge", Ridge(alpha=ridge_alpha)),
                        ("lasso", Lasso(alpha=lasso_alpha, max_iter=50000))]:
        m = model.fit(Xs, y)
        rows = [{"feature": "__intercept__", "coefficient": float(m.intercept_)}]
        rows += [{"feature": f, "coefficient": float(c)} for f, c in zip(feats, m.coef_)]
        vec = pd.DataFrame(rows)
        vec["site"] = site
        vec["panel"] = panel.upper()
        vec["n_subjects"] = len(y)
        vec["method"] = name
        vec["features_source"] = features_source
        vec.to_csv(os.path.join(outdir, f"{site}_panel{panel.upper()}_{name}_vector.csv"), index=False)
        logger.info(f"Wrote {site}_panel{panel.upper()}_{name}_vector.csv")

    rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2,
                               n_jobs=1, random_state=seed).fit(Xs, y)
    with open(os.path.join(outdir, f"{site}_panel{panel.upper()}_rf.pkl"), "wb") as fh:
        pickle.dump({"forest": rf, "scaler": sc, "features": feats,
                     "site": site, "panel": panel.upper(), "n_subjects": len(y)}, fh)
    logger.info(f"Wrote {site}_panel{panel.upper()}_rf.pkl  ({n_trees} trees on {len(feats)} features)")

    # Fit report: per-method 5-fold CV MSE / R2 + observed-vs-predicted graphic.
    p = panel.upper()
    kf = KFold(n_splits=min(5, max(2, len(y) // 2)), shuffle=True, random_state=seed)
    builders = [
        ("ridge", lambda: Ridge(alpha=ridge_alpha)),
        ("lasso", lambda: Lasso(alpha=lasso_alpha, max_iter=50000)),
        ("rf",    lambda: RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2,
                                                n_jobs=1, random_state=seed)),
    ]
    metrics, preds = [], {}
    for name, build in builders:
        pred = np.full(len(y), np.nan)
        for tr, te in kf.split(X):
            scf = MinMaxScaler().fit(X[tr])
            model = build().fit(scf.transform(X[tr]), y[tr])
            pred[te] = model.predict(scf.transform(X[te]))
        mse = float(np.nanmean((y - pred) ** 2))
        r2 = _r2(y, pred)
        metrics.append({"site": site, "panel": p, "method": name,
                        "n_subjects": len(y), "n_features": len(feats),
                        "mse": mse, "r2": r2,
                        "features_source": features_source, "features": ";".join(feats)})
        preds[name] = pred
        logger.info(f"  {name}: 5-fold CV  MSE={mse:.3f}  R2={r2:+.3f}")
    pd.DataFrame(metrics).to_csv(os.path.join(outdir, f"{site}_panel{p}_fit_metrics.csv"), index=False)
    _fit_report(site, p, y, preds, {m["method"]: m for m in metrics}, outdir, features_source)
    logger.info(f"Wrote {site}_panel{p}_fit_metrics.csv and {site}_panel{p}_fit.(png|svg|html)")


def _fit_report(site, panel, y, preds, mby, outdir, features_source=""):
    """Observed-vs-predicted scatter per method (5-fold CV) as PNG, SVG, and an
    interactive HTML. PNG/SVG via matplotlib; HTML via plotly's self-contained
    write_html (no kaleido/chromium needed). ``features_source`` is shown in the
    title so the graphic records which feature set it was fit on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [m for m in ("ridge", "lasso", "rf") if m in preds]
    lo = float(min(y.min(), *[np.nanmin(preds[m]) for m in methods]))
    hi = float(max(y.max(), *[np.nanmax(preds[m]) for m in methods]))
    base = os.path.join(outdir, f"{site}_panel{panel}_fit")
    src = f" — features: {features_source}" if features_source else ""
    suptitle = f"{site} panel {panel} — fit (5-fold CV){src}"

    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.6),
                             constrained_layout=True, squeeze=False)
    for ax, m in zip(axes[0], methods):
        ax.scatter(y, preds[m], c="#1f77b4", s=55, edgecolor="white")
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)
        ax.set_title(f"{m.upper()}  R²={mby[m]['r2']:+.2f}  MSE={mby[m]['mse']:.3f}", fontweight="bold")
        ax.set_xlabel("Observed log(C-peptide AUC)")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(base + ".png", dpi=220)
    fig.savefig(base + ".svg")
    plt.close(fig)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        titles = [f"{m.upper()}  R²={mby[m]['r2']:+.2f}  MSE={mby[m]['mse']:.3f}" for m in methods]
        pfig = make_subplots(rows=1, cols=len(methods), subplot_titles=titles)
        for i, m in enumerate(methods, start=1):
            pfig.add_trace(go.Scatter(x=y, y=preds[m], mode="markers", showlegend=False,
                                      marker=dict(size=8, line=dict(width=1, color="white"))),
                           row=1, col=i)
            pfig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                                      line=dict(dash="dash", color="black")), row=1, col=i)
            pfig.update_xaxes(title_text="Observed", row=1, col=i)
            pfig.update_yaxes(title_text="Predicted", row=1, col=i)
        pfig.update_layout(title_text=suptitle, width=420 * len(methods), height=460)
        pfig.write_html(base + ".html")
    except Exception as e:                     # plotly optional; PNG/SVG still produced
        logger.info(f"  (interactive HTML skipped: {e})")


def _r2(y, p):
    m = ~np.isnan(p); yy, pp = y[m], p[m]
    rss = float(np.sum((yy - pp) ** 2)); tss = float(np.sum((yy - yy.mean()) ** 2))
    return 1.0 - rss / tss if tss > 0 else float("nan")


def _boot_r2_ci(y, p, n_boot, seed):
    m = ~np.isnan(p); yy, pp = y[m], p[m]
    rng = np.random.default_rng(seed); n = len(yy); out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n); ys, ps = yy[idx], pp[idx]
        tss = float(np.sum((ys - ys.mean()) ** 2))
        out.append(1.0 - float(np.sum((ys - ps) ** 2)) / tss if tss > 0 else np.nan)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def apply_coefficients(site, panel="B", coefficients=None, data_root=".",
                       method=None, ridge_alpha=1.0, lasso_alpha=0.008,
                       n_boot=2000, outdir=".", seed=42):
    """Phase 3: incorporate the central federated coefficients.

    Reproduces the Stage-2 notebook evaluation from this site's own view: a
    5-fold CV comparing the site's SOLO model against the FEDERATED model, with
    bootstrap 95% CIs on R² and an observed-vs-predicted scatter. The federated
    arm applies the aggregator's central FedAvg vector as-is — the central
    average already includes this site, so it is not re-blended (that would
    double-count this site). Features are MinMax scaled within this site only,
    per fold. Subject-level predictions stay local; only the performance summary
    leaves.

    Args:
        site: study id.
        panel: feature panel ``A`` or ``B``.
        coefficients: path to the central federated coefficient vector CSV.
        data_root: directory holding the (flat) data files.
        method: ``ridge`` or ``lasso`` (default: read from the vector).
        ridge_alpha: Ridge L2 penalty for the solo model.
        lasso_alpha: LASSO L1 penalty for the solo model.
        n_boot: bootstrap resamples for the R² 95% CI.
        outdir: output directory.
        seed: random seed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame, _all, target = _load(site, panel, data_root)
    vec = pd.read_csv(coefficients)
    method = (method or (vec["method"].iloc[0] if "method" in vec.columns else "ridge")).lower()
    cd = dict(zip(vec["feature"], vec["coefficient"]))
    c_int = float(cd.pop("__intercept__", 0.0))
    feats = [f for f in vec["feature"] if f != "__intercept__"]
    c_coef = np.array([float(cd[f]) for f in feats])

    y = frame[target].astype(float).values
    X = frame.reindex(columns=feats).fillna(0.0).astype(float).values
    n = len(y)

    def model_fn():
        if method == "lasso":
            return Lasso(alpha=lasso_alpha, max_iter=50000)
        return Ridge(alpha=ridge_alpha)

    kf = KFold(n_splits=min(5, max(2, n // 2)), shuffle=True, random_state=seed)
    solo = np.full(n, np.nan)
    fed = np.full(n, np.nan)
    for tr, te in kf.split(X):
        sc = MinMaxScaler().fit(X[tr])                 # scale within this site only
        m = model_fn().fit(sc.transform(X[tr]), y[tr])
        Xte = sc.transform(X[te])
        solo[te] = m.predict(Xte)
        fed[te] = Xte @ c_coef + c_int                 # central federated vector as-is

    r2_s = _r2(y, solo); ci_s = _boot_r2_ci(y, solo, n_boot, seed)
    r2_f = _r2(y, fed);  ci_f = _boot_r2_ci(y, fed, n_boot, seed)

    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    lo = float(min(y.min(), np.nanmin(solo), np.nanmin(fed)))
    hi = float(max(y.max(), np.nanmax(solo), np.nanmax(fed)))
    panels = [(solo, f"{site} alone", r2_s, ci_s),
              (fed, f"{site} + federated", r2_f, ci_f)]
    for ax, (pred, title, r2v, ci) in zip(axes, panels):
        ax.scatter(y, pred, c="#1f77b4", s=60, edgecolor="white")
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)
        ax.set_xlabel("Observed log(C-peptide AUC)")
        ax.set_ylabel("Predicted log(C-peptide AUC)")
        ax.set_title(f"{title}\nR²={r2v:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]", fontweight="bold")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{method.upper()} — {site} solo vs federated",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(outdir, f"{site}_panel{panel.upper()}_{method}_federated.png"), dpi=220)
    fig.savefig(os.path.join(outdir, f"{site}_panel{panel.upper()}_{method}_federated.pdf"), dpi=300)
    plt.close(fig)

    # Subject-level predictions stay local (site's own use).
    pd.DataFrame({"y_true": y, "solo_pred": solo, "federated_pred": fed}).to_csv(
        os.path.join(outdir, f"{site}_panel{panel.upper()}_{method}_federated_predictions.csv"), index=False)
    # Scalar performance summary is what is meant to leave the site.
    pd.DataFrame([{"site": site, "panel": panel.upper(), "method": method, "n_subjects": n, "n_features": len(feats),
                   "r2_solo": r2_s, "r2_solo_lo": ci_s[0], "r2_solo_hi": ci_s[1],
                   "r2_federated": r2_f, "r2_fed_lo": ci_f[0], "r2_fed_hi": ci_f[1]}]).to_csv(
        os.path.join(outdir, f"{site}_panel{panel.upper()}_{method}_federated_performance.csv"), index=False)
    logger.info(f"{site} {method}, N={n}:")
    logger.info(f"  solo      R2={r2_s:+.3f}  95% CI [{ci_s[0]:+.2f}, {ci_s[1]:+.2f}]")
    logger.info(f"  federated R2={r2_f:+.3f}  95% CI [{ci_f[0]:+.2f}, {ci_f[1]:+.2f}]  "
                f"({'improves' if r2_f > r2_s else 'no gain'})")
