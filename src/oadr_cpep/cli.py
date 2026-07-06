"""
Command-line interface for oadr-cpep.

A single typer app drives both federated workflows — the per-site steps
(select-features, fit-models, apply-coefficients) and the coordinator steps
(consensus-features, aggregate-vectors).
"""
# src/oadr_cpep/cli.py

import typer
from pathlib import Path
from typing import Optional

from . import site as site_mod
from . import aggregate as agg_mod
from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")
app = typer.Typer(
    add_completion=False,
    help="Federated prediction of residual beta-cell function (C-peptide AUC) in Type 1 Diabetes.",
)


# ---------------------------------------------------------------- site: Phase 1
@app.command("select-features")
def select_features_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel: A (legacy 9) or B (extended 12)"),
    data_root: Path = typer.Option(".", help="Dir with the (flat) data files; default: the work dir"),
    out: Optional[Path] = typer.Option(None, help="Output CSV path (default: <site>_selected_features.csv)"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 1 (site): LASSO selects features on this site's own data."""
    site_mod.select_features(
        site=site, panel=panel, data_root=str(data_root),
        out=str(out) if out else None, seed=seed,
    )


# ---------------------------------------------------------------- site: Phase 2
@app.command("fit-models")
def fit_models_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel: A (legacy 9) or B (extended 12)"),
    features: Path = typer.Option(..., help="Consensus feature list CSV (column 'feature')"),
    data_root: Path = typer.Option(".", help="Dir with the (flat) data files"),
    outdir: Path = typer.Option(".", help="Output directory"),
    ridge_alpha: float = typer.Option(1.0, help="Ridge L2 penalty"),
    lasso_alpha: float = typer.Option(0.008, help="LASSO L1 penalty"),
    n_trees: int = typer.Option(200, help="Random Forest trees"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 2 (site): fit Ridge / LASSO / Random Forest on the consensus features."""
    site_mod.fit_models(
        site=site, panel=panel, features=str(features), data_root=str(data_root),
        outdir=str(outdir), ridge_alpha=ridge_alpha, lasso_alpha=lasso_alpha,
        n_trees=n_trees, seed=seed,
    )


# ---------------------------------------------------------------- site: Phase 3
@app.command("apply-coefficients")
def apply_coefficients_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel: A (legacy 9) or B (extended 12)"),
    coefficients: Path = typer.Option(..., help="Central federated coefficient vector CSV"),
    data_root: Path = typer.Option(".", help="Dir with the (flat) data files"),
    method: Optional[str] = typer.Option(None, help="ridge|lasso (default: read from the vector)"),
    ridge_alpha: float = typer.Option(1.0, help="Ridge L2 penalty for the solo model"),
    lasso_alpha: float = typer.Option(0.008, help="LASSO L1 penalty for the solo model"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    outdir: Path = typer.Option(".", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 3 (site): incorporate the central federated coefficients (solo vs federated)."""
    site_mod.apply_coefficients(
        site=site, panel=panel, coefficients=str(coefficients), data_root=str(data_root),
        method=method, ridge_alpha=ridge_alpha, lasso_alpha=lasso_alpha,
        n_boot=n_boot, outdir=str(outdir), seed=seed,
    )


# ------------------------------------------------------------ aggregator: Phase 1
@app.command("consensus-features")
def consensus_features_command(
    input_dir: Path = typer.Option(..., help="Dir (searched recursively) with per-site *_selected_features.csv"),
    min_sites: Optional[int] = typer.Option(None, help="Keep features chosen by >= this many sites (default: majority)"),
    outdir: Path = typer.Option(".", help="Output directory"),
):
    """Phase 1 (aggregator): tally the site selections into a consensus set."""
    agg_mod.consensus_features(
        input_dir=str(input_dir),
        min_sites=min_sites,
        outdir=str(outdir),
    )


# ------------------------------------------------------------ aggregator: Phase 2
@app.command("aggregate-vectors")
def aggregate_vectors_command(
    input_dir: Path = typer.Option(..., help="Dir (searched recursively) with per-site vectors / forests"),
    method: str = typer.Option("fedavg", help="Combine rule: fedavg | median | mean"),
    outdir: Path = typer.Option(".", help="Output directory"),
):
    """Phase 2 (aggregator): combine the site coefficient vectors / forests."""
    agg_mod.aggregate_vectors(
        input_dir=str(input_dir),
        method=method,
        outdir=str(outdir),
    )


def main():
    app()


if __name__ == "__main__":
    main()
