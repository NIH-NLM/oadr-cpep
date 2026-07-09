"""
Command-line interface for oadr-cpep.

A single typer app, one thin wrapper per single-function step:

  site       : select-features
               fit-ridge, fit-lasso, fit-rf   (single-method fits)
               fit-models                      (convenience: runs all three)
               apply-coefficients              (this site's solo-vs-federated outcome)
  aggregator : consensus-features, aggregate-vectors
"""
# src/oadr_cpep/cli.py

import typer
from pathlib import Path
from typing import Optional

from .select import select_features as _select_features
from .fit import (fit_ridge as _fit_ridge, fit_lasso as _fit_lasso,
                  fit_rf as _fit_rf, fit_models as _fit_models)
from .apply import apply_coefficients as _apply_coefficients
from .aggregate import (consensus_features as _consensus_features,
                        aggregate_vectors as _aggregate_vectors)

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
    outdir: Path = typer.Option(".", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 1 (site): LASSO selects features on this site's own data (alpha chosen by CV)."""
    _select_features(site=site, panel=panel, data_root=str(data_root), outdir=str(outdir), seed=seed)


# ---------------------------------------------------------------- site: Phase 2 (per method)
@app.command("fit-ridge")
def fit_ridge_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature-list CSV (column 'feature') to fit on"),
    data_root: Path = typer.Option(".", help="Dir with the (flat) data files"),
    outdir: Path = typer.Option(".", help="Output directory"),
    alpha: float = typer.Option(1.0, help="Ridge L2 penalty"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 2 (site): fit Ridge on a given feature set (vector + CV metrics + graphic)."""
    _fit_ridge(site=site, panel=panel, features=str(features), data_root=str(data_root),
               outdir=str(outdir), alpha=alpha, n_boot=n_boot, seed=seed)


@app.command("fit-lasso")
def fit_lasso_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature-list CSV (column 'feature') to fit on"),
    data_root: Path = typer.Option(".", help="Dir with the (flat) data files"),
    outdir: Path = typer.Option(".", help="Output directory"),
    alpha: float = typer.Option(0.008, help="LASSO L1 penalty"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 2 (site): fit LASSO on a given feature set (vector + CV metrics + graphic)."""
    _fit_lasso(site=site, panel=panel, features=str(features), data_root=str(data_root),
               outdir=str(outdir), alpha=alpha, n_boot=n_boot, seed=seed)


@app.command("fit-rf")
def fit_rf_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature-list CSV (column 'feature') to fit on"),
    data_root: Path = typer.Option(".", help="Dir with the (flat) data files"),
    outdir: Path = typer.Option(".", help="Output directory"),
    n_trees: int = typer.Option(200, help="Random Forest trees"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 2 (site): fit a Random Forest on a given feature set (forest + CV metrics + graphic)."""
    _fit_rf(site=site, panel=panel, features=str(features), data_root=str(data_root),
            outdir=str(outdir), n_trees=n_trees, n_boot=n_boot, seed=seed)


@app.command("fit-models")
def fit_models_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature-list CSV (column 'feature') to fit on"),
    data_root: Path = typer.Option(".", help="Dir with the (flat) data files"),
    outdir: Path = typer.Option(".", help="Output directory"),
    ridge_alpha: float = typer.Option(1.0, help="Ridge L2 penalty"),
    lasso_alpha: float = typer.Option(0.008, help="LASSO L1 penalty"),
    n_trees: int = typer.Option(200, help="Random Forest trees"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 2 (site): convenience — runs fit-ridge, fit-lasso, fit-rf on the same feature set."""
    _fit_models(site=site, panel=panel, features=str(features), data_root=str(data_root),
                outdir=str(outdir), ridge_alpha=ridge_alpha, lasso_alpha=lasso_alpha,
                n_trees=n_trees, n_boot=n_boot, seed=seed)


# ---------------------------------------------------------------- site: Phase 3
@app.command("apply-coefficients")
def apply_coefficients_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524 — the site whose outcome this is"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    coefficients_dir: Optional[Path] = typer.Option(None, help="Dir of federated_ results — runs all methods (ridge/lasso/rf)"),
    coefficients: Optional[Path] = typer.Option(None, help="A single federated vector CSV (or rf_union.pkl)"),
    from_features: Optional[str] = typer.Option(None, "--from", help="Feature source to select (e.g. SDY524)"),
    data_root: Path = typer.Option(".", help="Dir with the (flat) data files"),
    method: Optional[str] = typer.Option(None, help="ridge|lasso for a single --coefficients vector"),
    ridge_alpha: float = typer.Option(1.0, help="Ridge L2 penalty for the solo model"),
    lasso_alpha: float = typer.Option(0.008, help="LASSO L1 penalty for the solo model"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    outdir: Path = typer.Option(".", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 3 (site): this site's own outcome using the federated results (solo vs federated)."""
    _apply_coefficients(
        site=site, panel=panel,
        coefficients=str(coefficients) if coefficients else None,
        coefficients_dir=str(coefficients_dir) if coefficients_dir else None,
        from_features=from_features, data_root=str(data_root),
        method=method, ridge_alpha=ridge_alpha, lasso_alpha=lasso_alpha,
        n_boot=n_boot, outdir=str(outdir), seed=seed,
    )


# ------------------------------------------------------------ aggregator: Phase 1
@app.command("consensus-features")
def consensus_features_command(
    input_dir: Path = typer.Option(..., help="Dir (searched recursively) with per-site *_selected_features.csv"),
    min_sites: Optional[int] = typer.Option(None, help="Keep features chosen by >= this many sites (default: majority)"),
    from_site: Optional[str] = typer.Option(None, help="Use ONE site's selection as the consensus (single-site, bespoke)"),
    panel: Optional[str] = typer.Option(None, help="Restrict to one panel (A|B)"),
    outdir: Path = typer.Option(".", help="Output directory"),
):
    """Phase 1 (aggregator): build the consensus feature set (multi-site tally, or --from-site)."""
    _consensus_features(input_dir=str(input_dir), min_sites=min_sites,
                        from_site=from_site, panel=panel, outdir=str(outdir))


# ------------------------------------------------------------ aggregator: Phase 2
@app.command("aggregate-vectors")
def aggregate_vectors_command(
    input_dir: Path = typer.Option(..., help="Dir (searched recursively) with per-site vectors / forests"),
    method: str = typer.Option("fedavg", help="Combine rule: fedavg | median | mean"),
    from_features: Optional[str] = typer.Option(None, "--from", help="Restrict to vectors fit on one feature source (e.g. SDY524)"),
    panel: Optional[str] = typer.Option(None, help="Restrict to one panel (A|B)"),
    outdir: Path = typer.Option(".", help="Output directory"),
):
    """Phase 2 (aggregator): combine the site coefficient vectors / forests (solo vs federated)."""
    _aggregate_vectors(input_dir=str(input_dir), method=method,
                       from_features=from_features, panel=panel, outdir=str(outdir))


def main():
    app()


if __name__ == "__main__":
    main()
