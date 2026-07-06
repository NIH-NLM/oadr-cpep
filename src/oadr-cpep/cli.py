#!/usr/bin/env python3
"""oadr-cpep-cli entry point — the per-site (institution) subcommands.

  select-features   : Phase 1 — LASSO selects features on this site's own data.
  fit-models        : Phase 2 — Ridge/LASSO/RF on the consensus features.
  apply-coefficients: Phase 3 — incorporate the aggregator's central federated
                      vector: solo-vs-federated 5-fold CV, bootstrap 95% CI, and
                      an observed-vs-predicted scatter, from this site's view.

Data is read through the embedded oadr_data loader (the same one the
oadr-autoantibody notebooks use): each phase takes --site (the study, e.g.
SDY524), --panel (A|B), and --data-root (the ImmPort-derived data/ tree). Only
model parameters (feature lists, coefficient vectors, trained forests) and
scalar performance summaries leave the site — never subject-level data. The
aggregator-side commands live in the aggregator workflow repo.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

try:                                   # installed as a package (container)
    from . import oadr_data as od
except ImportError:                    # run as a script (local dev)
    import oadr_data as od


def _load(args):
    """Load this study+panel through oadr_data, rooted at --data-root."""
    od._DATA = Path(args.data_root)
    return od.load_features(args.site, args.panel)   # (frame, feature_names, target)


# ------------------------------------------------------------------ site: select
def select_features(args):
    """Phase 1 (site): LASSO selects features on this site's own data."""
    frame, feats, target = _load(args)
    X = frame[feats].astype(float).values
    y = frame[target].astype(float).values

    sc = MinMaxScaler().fit(X)                     # scale within this site only
    cv = max(2, min(5, len(y) // 4))
    m = LassoCV(cv=cv, random_state=args.seed, max_iter=50000).fit(sc.transform(X), y)

    out = pd.DataFrame({"feature": feats, "coefficient": m.coef_,
                        "selected": (np.abs(m.coef_) > 1e-8).astype(int)})
    out["site"] = args.site
    out["panel"] = args.panel.upper()
    out["n_subjects"] = len(y)
    out["alpha"] = float(m.alpha_)
    path = args.out or f"{args.site}_selected_features.csv"
    out.to_csv(path, index=False)
    kept = [f for f, c in zip(feats, m.coef_) if abs(c) > 1e-8]
    print(f"{args.site} panel {args.panel.upper()}: N={len(y)}, "
          f"selected {len(kept)}/{len(feats)} at alpha={m.alpha_:.4f}: {kept}")


# ------------------------------------------------------------------ site: fit
def fit_models(args):
    """Phase 2 (site): fit Ridge/LASSO/RF on the consensus features."""
    frame, _all, target = _load(args)
    feats = list(pd.read_csv(args.features)["feature"])
    y = frame[target].astype(float).values
    X = frame.reindex(columns=feats).fillna(0.0).astype(float).values
    sc = MinMaxScaler().fit(X)
    Xs = sc.transform(X)
    os.makedirs(args.outdir, exist_ok=True)

    for name, model in [("ridge", Ridge(alpha=args.ridge_alpha)),
                        ("lasso", Lasso(alpha=args.lasso_alpha, max_iter=50000))]:
        m = model.fit(Xs, y)
        rows = [{"feature": "__intercept__", "coefficient": float(m.intercept_)}]
        rows += [{"feature": f, "coefficient": float(c)} for f, c in zip(feats, m.coef_)]
        vec = pd.DataFrame(rows)
        vec["site"] = args.site
        vec["n_subjects"] = len(y)
        vec["method"] = name
        vec.to_csv(os.path.join(args.outdir, f"{args.site}_{name}_vector.csv"), index=False)
        print(f"Wrote {args.site}_{name}_vector.csv")

    rf = RandomForestRegressor(n_estimators=args.n_trees, min_samples_leaf=2,
                               n_jobs=1, random_state=args.seed).fit(Xs, y)
    with open(os.path.join(args.outdir, f"{args.site}_rf.pkl"), "wb") as fh:
        pickle.dump({"forest": rf, "scaler": sc, "features": feats,
                     "site": args.site, "n_subjects": len(y)}, fh)
    print(f"Wrote {args.site}_rf.pkl  ({args.n_trees} trees on {len(feats)} features)")


# ------------------------------------------------------------------ site: apply
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


def apply_coefficients(args):
    """Phase 3 (site): incorporate the central federated coefficients.

    Reproduces the Stage-2 notebook evaluation from this site's own view:
    5-fold CV comparing the site's SOLO model against the FEDERATED model, with
    bootstrap 95% CIs on R² and an observed-vs-predicted scatter.

    The federated arm applies the aggregator's central FedAvg vector as-is. The
    central average already includes this site's contribution, so it is not
    re-blended with the site's own coefficients (that would double-count this
    site). Features are MinMax scaled within this site only, per fold. Subject-
    level predictions stay local; only the performance summary leaves.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame, _all, target = _load(args)
    vec = pd.read_csv(args.coefficients)
    method = (args.method or (vec["method"].iloc[0] if "method" in vec.columns else "ridge")).lower()
    cd = dict(zip(vec["feature"], vec["coefficient"]))
    c_int = float(cd.pop("__intercept__", 0.0))
    feats = [f for f in vec["feature"] if f != "__intercept__"]
    c_coef = np.array([float(cd[f]) for f in feats])

    y = frame[target].astype(float).values
    X = frame.reindex(columns=feats).fillna(0.0).astype(float).values
    n = len(y)

    def model_fn():
        if method == "lasso":
            return Lasso(alpha=args.lasso_alpha, max_iter=50000)
        return Ridge(alpha=args.ridge_alpha)

    kf = KFold(n_splits=min(5, max(2, n // 2)), shuffle=True, random_state=args.seed)
    solo = np.full(n, np.nan)
    fed = np.full(n, np.nan)
    for tr, te in kf.split(X):
        sc = MinMaxScaler().fit(X[tr])                 # scale within this site only
        m = model_fn().fit(sc.transform(X[tr]), y[tr])
        Xte = sc.transform(X[te])
        solo[te] = m.predict(Xte)
        fed[te] = Xte @ c_coef + c_int                 # central federated vector as-is

    r2_s = _r2(y, solo); ci_s = _boot_r2_ci(y, solo, args.n_boot, args.seed)
    r2_f = _r2(y, fed);  ci_f = _boot_r2_ci(y, fed, args.n_boot, args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    lo = float(min(y.min(), np.nanmin(solo), np.nanmin(fed)))
    hi = float(max(y.max(), np.nanmax(solo), np.nanmax(fed)))
    panels = [(solo, f"{args.site} alone", r2_s, ci_s),
              (fed, f"{args.site} + federated", r2_f, ci_f)]
    for ax, (pred, title, r2v, ci) in zip(axes, panels):
        ax.scatter(y, pred, c="#1f77b4", s=60, edgecolor="white")
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)
        ax.set_xlabel("Observed log(C-peptide AUC)")
        ax.set_ylabel("Predicted log(C-peptide AUC)")
        ax.set_title(f"{title}\nR²={r2v:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]", fontweight="bold")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{method.upper()} — {args.site} solo vs federated",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(args.outdir, f"{args.site}_{method}_federated.png"), dpi=220)
    fig.savefig(os.path.join(args.outdir, f"{args.site}_{method}_federated.pdf"), dpi=300)
    plt.close(fig)

    # Subject-level predictions stay local (site's own use).
    pd.DataFrame({"y_true": y, "solo_pred": solo, "federated_pred": fed}).to_csv(
        os.path.join(args.outdir, f"{args.site}_{method}_federated_predictions.csv"), index=False)
    # Scalar performance summary is what is meant to leave the site.
    pd.DataFrame([{"site": args.site, "method": method, "n_subjects": n, "n_features": len(feats),
                   "r2_solo": r2_s, "r2_solo_lo": ci_s[0], "r2_solo_hi": ci_s[1],
                   "r2_federated": r2_f, "r2_fed_lo": ci_f[0], "r2_fed_hi": ci_f[1]}]).to_csv(
        os.path.join(args.outdir, f"{args.site}_{method}_federated_performance.csv"), index=False)
    print(f"{args.site} {method}, N={n}:")
    print(f"  solo      R²={r2_s:+.3f}  95% CI [{ci_s[0]:+.2f}, {ci_s[1]:+.2f}]")
    print(f"  federated R²={r2_f:+.3f}  95% CI [{ci_f[0]:+.2f}, {ci_f[1]:+.2f}]  "
          f"({'improves' if r2_f > r2_s else 'no gain'})")


def _add_data_args(a):
    """Shared inputs: --site (study), --panel A|B, --data-root (flat data files)."""
    a.add_argument("--site", required=True, help="study / institution id, e.g. SDY524")
    a.add_argument("--panel", default="B", help="feature panel: A (legacy 9) or B (extended 12)")
    a.add_argument("--data-root", default=".",
                   help="dir holding the (flat) data files; default: the work dir where Nextflow stages them")


def main(argv=None):
    p = argparse.ArgumentParser(prog="oadr-cpep-cli",
                                description="Federated prediction of residual beta-cell function (site steps)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("select-features", help="Phase 1 (site) LASSO feature selection")
    _add_data_args(a)
    a.add_argument("--out", default=None); a.add_argument("--seed", type=int, default=42)
    a.set_defaults(func=select_features)

    a = sub.add_parser("fit-models", help="Phase 2 (site) fit Ridge/LASSO/RF on consensus features")
    _add_data_args(a)
    a.add_argument("--features", required=True)
    a.add_argument("--outdir", default="."); a.add_argument("--ridge-alpha", type=float, default=1.0)
    a.add_argument("--lasso-alpha", type=float, default=0.008); a.add_argument("--n-trees", type=int, default=200)
    a.add_argument("--seed", type=int, default=42); a.set_defaults(func=fit_models)

    a = sub.add_parser("apply-coefficients",
                       help="Phase 3 (site) incorporate the central federated coefficients (solo vs federated)")
    _add_data_args(a)
    a.add_argument("--coefficients", required=True)
    a.add_argument("--method", default=None, help="ridge|lasso (default: read from the vector)")
    a.add_argument("--ridge-alpha", type=float, default=1.0)
    a.add_argument("--lasso-alpha", type=float, default=0.008)
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--outdir", default="."); a.add_argument("--seed", type=int, default=42)
    a.set_defaults(func=apply_coefficients)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
