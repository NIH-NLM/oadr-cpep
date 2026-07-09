"""
Plotting for oadr-cpep. All graphics live here (matplotlib for PNG/SVG, plotly
for the self-contained interactive HTML) so the compute modules stay free of
rendering code.
"""
from __future__ import annotations

import os

import numpy as np


def scatter(y, pred, title, base):
    """One observed-vs-predicted scatter -> <base>.png, .svg, and interactive .html."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo = float(min(y.min(), np.nanmin(pred)))
    hi = float(max(y.max(), np.nanmax(pred)))
    fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    ax.scatter(y, pred, c="#1f77b4", s=55, edgecolor="white")
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Observed log(C-peptide AUC)")
    ax.set_ylabel("Predicted")
    ax.grid(alpha=0.25)
    fig.savefig(base + ".png", dpi=220)
    fig.savefig(base + ".svg")
    plt.close(fig)
    try:
        import plotly.graph_objects as go
        pf = go.Figure()
        pf.add_trace(go.Scatter(x=y, y=pred, mode="markers", showlegend=False,
                                marker=dict(size=8, line=dict(width=1, color="white"))))
        pf.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                                line=dict(dash="dash", color="black")))
        pf.update_layout(title_text=title, xaxis_title="Observed", yaxis_title="Predicted",
                         width=560, height=480)
        pf.write_html(base + ".html")
    except Exception:                     # plotly optional; PNG/SVG still produced
        pass


def solo_vs_federated(site, panel, y, results, base, sites_label=""):
    """Per-method solo-vs-federated grid (rows = methods, cols = solo|federated)
    -> <base>.png, .svg, and interactive .html. ``results`` is the list built by
    apply_coefficients (each has method, solo, fed, r2_solo, ci_solo, r2_fed, ci_fed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nm = len(results)
    allp = [y] + [r["solo"] for r in results] + [r["fed"] for r in results]
    lo = float(min(np.nanmin(a) for a in allp))
    hi = float(max(np.nanmax(a) for a in allp))
    prov = f" [of {sites_label}]" if sites_label else ""
    suptitle = f"{site} panel {panel} — solo vs federated{prov}"

    fig, axes = plt.subplots(nm, 2, figsize=(10, 4.4 * nm), constrained_layout=True, squeeze=False)
    for row, r in enumerate(results):
        for col, (pred, lab, r2v, ci) in enumerate([
                (r["solo"], f"{site} alone", r["r2_solo"], r["ci_solo"]),
                (r["fed"], f"{site} + federated", r["r2_fed"], r["ci_fed"])]):
            ax = axes[row][col]
            ax.scatter(y, pred, c="#1f77b4", s=55, edgecolor="white")
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)
            ax.set_title(f"{r['method'].upper()} — {lab}\nR²={r2v:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]",
                         fontweight="bold", fontsize=10)
            ax.set_xlabel("Observed"); ax.set_ylabel("Predicted"); ax.grid(alpha=0.25)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(base + ".png", dpi=220)
    fig.savefig(base + ".svg")
    plt.close(fig)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        titles = []
        for r in results:
            titles += [f"{r['method'].upper()} — alone  R²={r['r2_solo']:+.2f}",
                       f"{r['method'].upper()} — +federated  R²={r['r2_fed']:+.2f}"]
        pfig = make_subplots(rows=nm, cols=2, subplot_titles=titles)
        for row, r in enumerate(results, start=1):
            for col, pred in enumerate([r["solo"], r["fed"]], start=1):
                pfig.add_trace(go.Scatter(x=y, y=pred, mode="markers", showlegend=False,
                                          marker=dict(size=7, line=dict(width=1, color="white"))),
                               row=row, col=col)
                pfig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                                          line=dict(dash="dash", color="black")), row=row, col=col)
        pfig.update_layout(title_text=suptitle, width=900, height=380 * nm)
        pfig.write_html(base + ".html")
    except Exception:
        pass
