"""
Command-line interface for oadr-cpep.

A single typer app, one thin wrapper per single-function step:

  site       : select-features
               fit-ridge, fit-lasso, fit-rf   (single-method fits)
               fit-models                      (convenience: runs all three)
               apply-coefficients              (this site's solo-vs-federated outcome)
  aggregator : consensus-features, aggregate-vectors

Every step takes its inputs as EXPLICIT files — no directories, no globs, nothing
resolved by name. The site steps read a study's data files by panel:
  Panel A : --tidy, --cpeptide
  Panel B : --aa, --demo, --cpeptide, --arms, --arm-subjects  (arms optional)
"""
# src/oadr_cpep/cli.py

import typer
from pathlib import Path
from typing import List, Optional

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


def _files(tidy, aa, demo, cpeptide, arms, arm_subjects):
    """Collect the explicit data-file options (Path -> str, dropping any not given)
    into the loader kwargs. Panel A uses tidy+cpeptide; Panel B uses
    aa+demo+cpeptide (+arms/arm-subjects); the loader validates what it needs."""
    m = {"tidy": tidy, "aa": aa, "demo": demo, "cpeptide": cpeptide,
         "arms": arms, "arm_subjects": arm_subjects}
    return {k: str(v) for k, v in m.items() if v is not None}


# --- shared data-file options (declared per command; typer needs them inline) ---
_TIDY = typer.Option(None, "--tidy", help="Panel A features: SDY<n>_tidy.csv")
_AA = typer.Option(None, "--aa", help="Panel B autoantibodies + anthropometrics: aa_<n>.csv")
_DEMO = typer.Option(None, "--demo", help="Panel B demographics: demo_<n>.csv")
_CPEP = typer.Option(None, "--cpeptide", help="C-peptide AUC target: SDY<n>_cpeptide_auc_tidy.csv")
_ARMS = typer.Option(None, "--arms", help="Panel B treatment arms (optional): SDY<n>_arm_or_cohort.txt")
_ARMSUBJ = typer.Option(None, "--arm-subjects", help="Panel B arm->subject map (optional): SDY<n>_arm_2_subject.txt")


# ---------------------------------------------------------------- site: Phase 1
@app.command("select-features")
def select_features_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel: A (legacy 9) or B (extended 12)"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    outdir: Path = typer.Option(".", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 1 (site): LASSO selects features on this site's own data (alpha chosen by CV)."""
    _select_features(site=site, panel=panel, outdir=str(outdir), seed=seed,
                     **_files(tidy, aa, demo, cpeptide, arms, arm_subjects))


# ---------------------------------------------------------------- site: Phase 2 (per method)
@app.command("fit-ridge")
def fit_ridge_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature-list CSV (column 'feature') to fit on"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    outdir: Path = typer.Option(".", help="Output directory"),
    alpha: float = typer.Option(1.0, help="Ridge L2 penalty"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 2 (site): fit Ridge on a given feature set (vector + CV metrics + graphic)."""
    _fit_ridge(site=site, panel=panel, features=str(features), outdir=str(outdir),
               alpha=alpha, n_boot=n_boot, seed=seed,
               **_files(tidy, aa, demo, cpeptide, arms, arm_subjects))


@app.command("fit-lasso")
def fit_lasso_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature-list CSV (column 'feature') to fit on"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    outdir: Path = typer.Option(".", help="Output directory"),
    alpha: float = typer.Option(0.008, help="LASSO L1 penalty"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 2 (site): fit LASSO on a given feature set (vector + CV metrics + graphic)."""
    _fit_lasso(site=site, panel=panel, features=str(features), outdir=str(outdir),
               alpha=alpha, n_boot=n_boot, seed=seed,
               **_files(tidy, aa, demo, cpeptide, arms, arm_subjects))


@app.command("fit-rf")
def fit_rf_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature-list CSV (column 'feature') to fit on"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    outdir: Path = typer.Option(".", help="Output directory"),
    n_trees: int = typer.Option(200, help="Random Forest trees"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 2 (site): fit a Random Forest on a given feature set (forest + CV metrics + graphic)."""
    _fit_rf(site=site, panel=panel, features=str(features), outdir=str(outdir),
            n_trees=n_trees, n_boot=n_boot, seed=seed,
            **_files(tidy, aa, demo, cpeptide, arms, arm_subjects))


@app.command("fit-models")
def fit_models_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature-list CSV (column 'feature') to fit on"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    outdir: Path = typer.Option(".", help="Output directory"),
    ridge_alpha: float = typer.Option(1.0, help="Ridge L2 penalty"),
    lasso_alpha: float = typer.Option(0.008, help="LASSO L1 penalty"),
    n_trees: int = typer.Option(200, help="Random Forest trees"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 2 (site): convenience — runs fit-ridge, fit-lasso, fit-rf on the same feature set."""
    _fit_models(site=site, panel=panel, features=str(features), outdir=str(outdir),
                ridge_alpha=ridge_alpha, lasso_alpha=lasso_alpha,
                n_trees=n_trees, n_boot=n_boot, seed=seed,
                **_files(tidy, aa, demo, cpeptide, arms, arm_subjects))


# ---------------------------------------------------------------- site: Phase 3
@app.command("apply-coefficients")
def apply_coefficients_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524 — the site whose outcome this is"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    ridge_vector: Optional[Path] = typer.Option(None, "--ridge-vector", help="Federated ridge vector CSV"),
    lasso_vector: Optional[Path] = typer.Option(None, "--lasso-vector", help="Federated lasso vector CSV"),
    rf_union: Optional[Path] = typer.Option(None, "--rf-union", help="Federated RF union pickle"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    ridge_alpha: float = typer.Option(1.0, help="Ridge L2 penalty for the solo model"),
    lasso_alpha: float = typer.Option(0.008, help="LASSO L1 penalty for the solo model"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the R² 95% CI"),
    outdir: Path = typer.Option(".", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 3 (site): this site's own outcome using the federated results (solo vs federated)."""
    _apply_coefficients(
        site=site, panel=panel,
        ridge_vector=str(ridge_vector) if ridge_vector else None,
        lasso_vector=str(lasso_vector) if lasso_vector else None,
        rf_union=str(rf_union) if rf_union else None,
        ridge_alpha=ridge_alpha, lasso_alpha=lasso_alpha,
        n_boot=n_boot, outdir=str(outdir), seed=seed,
        **_files(tidy, aa, demo, cpeptide, arms, arm_subjects),
    )


# ------------------------------------------------------------ aggregator: Phase 1
@app.command("consensus-features")
def consensus_features_command(
    features: List[Path] = typer.Option(..., "--features", help="Per-site selected-features CSV (repeat --features for each site)"),
    min_sites: Optional[int] = typer.Option(None, help="Keep features chosen by >= this many sites (default: majority)"),
    from_site: Optional[str] = typer.Option(None, help="Use ONE site's selection as the consensus (single-site, bespoke)"),
    outdir: Path = typer.Option(".", help="Output directory"),
):
    """Phase 1 (aggregator): build the consensus feature set (multi-site tally, or --from-site)."""
    _consensus_features(features=[str(f) for f in features], min_sites=min_sites,
                        from_site=from_site, outdir=str(outdir))


# ------------------------------------------------------------ aggregator: Phase 2
@app.command("aggregate-vectors")
def aggregate_vectors_command(
    vector: List[Path] = typer.Option(..., "--vector", help="Per-site coefficient vector CSV or RF .pkl (repeat --vector for each)"),
    method: str = typer.Option("fedavg", help="Combine rule: fedavg | median | mean"),
    outdir: Path = typer.Option(".", help="Output directory"),
):
    """Phase 2 (aggregator): combine the site coefficient vectors / forests (solo vs federated)."""
    _aggregate_vectors(vectors=[str(f) for f in vector], method=method, outdir=str(outdir))


def main():
    app()


if __name__ == "__main__":
    main()
