"""
Command-line interface for oadr-cpep.

A single typer app, one thin wrapper per single-function step:

  site       : select-features
               fit-ridge, fit-lasso, fit-rf   (single-method fits)
               fit-logistic                    (responder classifier -> ROC AUC)
               fit-models                      (convenience: runs all of them)
               apply-coefficients              (this site's solo-vs-federated outcome)
  aggregator : consensus-features, aggregate-vectors

Inputs are either explicit files or a Cohort Browser selection (--cohort-id).
Results move between the site and aggregator workflows as JSON payloads
(--emit-json), each stamped with its --iteration so successive rounds stay
distinct. Every parameter is set up front; nothing prompts.

Every step takes its inputs as EXPLICIT files — no directories, no globs, nothing
resolved by name — and writes its outputs to the current working directory (no
output-dir option; under Nextflow that's the process work dir, published by
publishDir). The site steps read a study's data files by panel:
  Panel A : --tidy, --cpeptide
  Panel B : --aa, --demo, --cpeptide, --arms, --arm-subjects  (arms optional)
"""
# src/oadr_cpep/cli.py

import typer
from pathlib import Path
from typing import List, Optional

from .select import select_features as _select_features
from .fit import (fit_ridge as _fit_ridge, fit_lasso as _fit_lasso,
                  fit_rf as _fit_rf, fit_logistic as _fit_logistic,
                  fit_models as _fit_models)
from .common_utils import DEFAULT_RESPONDER_THRESHOLD
from .apply import apply_coefficients as _apply_coefficients
from .aggregate import (consensus_features as _consensus_features,
                        aggregate_vectors as _aggregate_vectors)

app = typer.Typer(
    add_completion=False,
    help="Federated prediction of residual beta-cell function (C-peptide AUC) in Type 1 Diabetes.",
)


def _sql_text(text, path, flag):
    """Resolve a query given inline or as a file -> SQL string (never both).

    Resolving here means everything below the CLI only ever sees SQL text, so the
    inline/file choice stays a command-line concern.
    """
    if text and path:
        raise typer.BadParameter(f"pass {flag} or {flag}-file, not both")
    if path:
        try:
            return Path(path).read_text()
        except OSError as e:
            raise typer.BadParameter(f"cannot read {flag}-file {path}: {e}") from e
    return text


def _files(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id=None,
           cohort_sql=None, cohort_sql_file=None, cohort_target_sql=None,
           cohort_target_sql_file=None, max_rows=None, cb_profile=""):
    """Collect the input options (Path -> str, dropping any not given) into the
    loader kwargs. Panel A uses tidy+cpeptide; Panel B uses aa+demo+cpeptide
    (+arms/arm-subjects); --cohort-id reads the Cohort Browser instead. The
    loader validates what it needs."""
    m = {"tidy": tidy, "aa": aa, "demo": demo, "cpeptide": cpeptide,
         "arms": arms, "arm_subjects": arm_subjects}
    out = {k: str(v) for k, v in m.items() if v is not None}
    sql = _sql_text(cohort_sql, cohort_sql_file, "--cohort-sql")
    target_sql = _sql_text(cohort_target_sql, cohort_target_sql_file, "--cohort-target-sql")
    if cohort_id:
        if any(k != "cpeptide" for k in out):
            raise typer.BadParameter(
                "--cohort-id reads the cohort itself; do not also pass "
                "--tidy/--aa/--demo/--arms (only --cpeptide may accompany it, as "
                "a local target)")
        out["cohort_id"] = cohort_id
        # the public API's names; cu.cohort_kwargs() renames them to the ones
        # cohort_data.load_cohort expects (sql / target_sql / profilename).
        out["cohort_sql"] = sql
        out["cohort_target_sql"] = target_sql
        out["cb_profile"] = cb_profile
        out["max_rows"] = max_rows
    elif not out:
        raise typer.BadParameter(
            "no input given: pass the panel's data files, or --cohort-id to read "
            "a Cohort Browser selection")
    elif sql or target_sql or max_rows is not None or cb_profile:
        raise typer.BadParameter(
            "--cohort-sql / --cohort-target-sql / --max-rows / --cb-profile apply "
            "only with --cohort-id")
    return out


def _trim(trim_outliers, trim_lower, trim_upper, trim_on):
    """The outlier-trimming settings. Off unless explicitly asked for."""
    if not trim_outliers:
        return {"enabled": False}
    return {"enabled": True, "lower_pct": trim_lower, "upper_pct": trim_upper, "on": trim_on}


# --- shared data-file options (declared per command; typer needs them inline) ---
_TIDY = typer.Option(None, "--tidy", help="Panel A features: SDY<n>_tidy.csv")
_AA = typer.Option(None, "--aa", help="Panel B autoantibodies + anthropometrics: aa_<n>.csv")
_DEMO = typer.Option(None, "--demo", help="Panel B demographics: demo_<n>.csv")
_CPEP = typer.Option(None, "--cpeptide", help="C-peptide AUC target: SDY<n>_cpeptide_auc_tidy.csv")
_ARMS = typer.Option(None, "--arms", help="Panel B treatment arms (optional): SDY<n>_arm_or_cohort.txt")
_ARMSUBJ = typer.Option(None, "--arm-subjects", help="Panel B arm->subject map (optional): SDY<n>_arm_2_subject.txt")
_COHORT = typer.Option(None, "--cohort-id", help="CloudOS Cohort Browser cohort id (from the cohort's URL) — reads the cohort instead of files")
_COHORT_SQL = typer.Option(None, "--cohort-sql", help="Measurement query for --cohort-id: selects person_id, measurement_concept_id, value_as_number (concept_name/domain_id used when present). Run describe-cohort first")
_COHORT_SQL_FILE = typer.Option(None, "--cohort-sql-file", help="Same as --cohort-sql but read from a .sql file (what a Nextflow process stages)")
_COHORT_TGT = typer.Option(None, "--cohort-target-sql", help="Query returning person_id and the C-peptide AUC target; or use --cpeptide for a local target file")
_COHORT_TGT_FILE = typer.Option(None, "--cohort-target-sql-file", help="Same as --cohort-target-sql but read from a .sql file")
_MAX_ROWS = typer.Option(None, "--max-rows", help="Refuse a cohort query matching more rows than this (query() submits one async task per 1000-row page)")
_CB_PROFILE = typer.Option("", "--cb-profile", help="cloudos-cb-py credential profile name (default: the configured default profile)")
_ITER = typer.Option(1, "--iteration", help="Federated round this fit belongs to; stamped on every output")
_EMIT = typer.Option(False, "--emit-json", help="Also write the JSON exchange payload the aggregator consumes")
_TRIM = typer.Option(False, "--trim-outliers", help="Drop rows outside the percentile range before fitting (default: keep all)")
_TRIML = typer.Option(5.0, "--trim-lower", help="Lower percentile for --trim-outliers")
_TRIMU = typer.Option(95.0, "--trim-upper", help="Upper percentile for --trim-outliers")
_TRIMON = typer.Option("target", "--trim-on", help="What --trim-outliers applies to: target | features | both")


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
    cohort_id: Optional[str] = _COHORT,
    cohort_sql: Optional[str] = _COHORT_SQL,
    cohort_sql_file: Optional[Path] = _COHORT_SQL_FILE,
    cohort_target_sql: Optional[str] = _COHORT_TGT,
    cohort_target_sql_file: Optional[Path] = _COHORT_TGT_FILE,
    max_rows: Optional[int] = _MAX_ROWS,
    cb_profile: str = _CB_PROFILE,
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 1 (site): LASSO selects features on this site's own data (alpha chosen by CV)."""
    _select_features(site=site, panel=panel, seed=seed,
                     **_files(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id, cohort_sql, cohort_sql_file, cohort_target_sql, cohort_target_sql_file, max_rows, cb_profile))


# ---------------------------------------------------------------- site: Phase 2 (per method)
@app.command("fit-ridge")
def fit_ridge_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature set to fit on: CSV (column 'feature') or feature-space JSON"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    cohort_id: Optional[str] = _COHORT,
    cohort_sql: Optional[str] = _COHORT_SQL,
    cohort_sql_file: Optional[Path] = _COHORT_SQL_FILE,
    cohort_target_sql: Optional[str] = _COHORT_TGT,
    cohort_target_sql_file: Optional[Path] = _COHORT_TGT_FILE,
    max_rows: Optional[int] = _MAX_ROWS,
    cb_profile: str = _CB_PROFILE,
    alpha: float = typer.Option(1.0, help="Ridge L2 penalty"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the 95% CIs"),
    seed: int = typer.Option(42, help="Random seed"),
    iteration: int = _ITER,
    emit_json: bool = _EMIT,
    trim_outliers: bool = _TRIM,
    trim_lower: float = _TRIML,
    trim_upper: float = _TRIMU,
    trim_on: str = _TRIMON,
):
    """Phase 2 (site): fit Ridge on a given feature set (vector + CV metrics + graphic)."""
    _fit_ridge(site=site, panel=panel, features=str(features),
               alpha=alpha, n_boot=n_boot, seed=seed, iteration=iteration,
               emit_json=emit_json, trim=_trim(trim_outliers, trim_lower, trim_upper, trim_on),
               **_files(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id, cohort_sql, cohort_sql_file, cohort_target_sql, cohort_target_sql_file, max_rows, cb_profile))


@app.command("fit-lasso")
def fit_lasso_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature set to fit on: CSV (column 'feature') or feature-space JSON"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    cohort_id: Optional[str] = _COHORT,
    cohort_sql: Optional[str] = _COHORT_SQL,
    cohort_sql_file: Optional[Path] = _COHORT_SQL_FILE,
    cohort_target_sql: Optional[str] = _COHORT_TGT,
    cohort_target_sql_file: Optional[Path] = _COHORT_TGT_FILE,
    max_rows: Optional[int] = _MAX_ROWS,
    cb_profile: str = _CB_PROFILE,
    alpha: float = typer.Option(0.008, help="LASSO L1 penalty"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the 95% CIs"),
    seed: int = typer.Option(42, help="Random seed"),
    iteration: int = _ITER,
    emit_json: bool = _EMIT,
    trim_outliers: bool = _TRIM,
    trim_lower: float = _TRIML,
    trim_upper: float = _TRIMU,
    trim_on: str = _TRIMON,
):
    """Phase 2 (site): fit LASSO on a given feature set (vector + CV metrics + graphic)."""
    _fit_lasso(site=site, panel=panel, features=str(features),
               alpha=alpha, n_boot=n_boot, seed=seed, iteration=iteration,
               emit_json=emit_json, trim=_trim(trim_outliers, trim_lower, trim_upper, trim_on),
               **_files(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id, cohort_sql, cohort_sql_file, cohort_target_sql, cohort_target_sql_file, max_rows, cb_profile))


@app.command("fit-rf")
def fit_rf_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature set to fit on: CSV (column 'feature') or feature-space JSON"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    cohort_id: Optional[str] = _COHORT,
    cohort_sql: Optional[str] = _COHORT_SQL,
    cohort_sql_file: Optional[Path] = _COHORT_SQL_FILE,
    cohort_target_sql: Optional[str] = _COHORT_TGT,
    cohort_target_sql_file: Optional[Path] = _COHORT_TGT_FILE,
    max_rows: Optional[int] = _MAX_ROWS,
    cb_profile: str = _CB_PROFILE,
    n_trees: int = typer.Option(200, help="Random Forest trees"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the 95% CIs"),
    seed: int = typer.Option(42, help="Random seed"),
    iteration: int = _ITER,
    emit_json: bool = _EMIT,
    trim_outliers: bool = _TRIM,
    trim_lower: float = _TRIML,
    trim_upper: float = _TRIMU,
    trim_on: str = _TRIMON,
):
    """Phase 2 (site): fit a Random Forest (forest + CV metrics + graphic). --emit-json writes the trees as data."""
    _fit_rf(site=site, panel=panel, features=str(features),
            n_trees=n_trees, n_boot=n_boot, seed=seed, iteration=iteration,
            emit_json=emit_json, trim=_trim(trim_outliers, trim_lower, trim_upper, trim_on),
            **_files(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id, cohort_sql, cohort_sql_file, cohort_target_sql, cohort_target_sql_file, max_rows, cb_profile))


@app.command("fit-logistic")
def fit_logistic_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature set to fit on: CSV (column 'feature') or feature-space JSON"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    cohort_id: Optional[str] = _COHORT,
    cohort_sql: Optional[str] = _COHORT_SQL,
    cohort_sql_file: Optional[Path] = _COHORT_SQL_FILE,
    cohort_target_sql: Optional[str] = _COHORT_TGT,
    cohort_target_sql_file: Optional[Path] = _COHORT_TGT_FILE,
    max_rows: Optional[int] = _MAX_ROWS,
    cb_profile: str = _CB_PROFILE,
    penalty: str = typer.Option("l2", help="l2 (Ridge analogue) or l1 (LASSO analogue)"),
    alpha: float = typer.Option(1.0, help="Penalty strength; the classifier uses C = 1/alpha"),
    responder_threshold: float = typer.Option(
        DEFAULT_RESPONDER_THRESHOLD, "--responder-threshold",
        help="Responder cutoff on raw C-peptide AUC in ng/mL (default 0.604 = 0.2 nmol/L). "
             "Set at configuration time; it defines the label ROC AUC is computed against"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the 95% CIs"),
    seed: int = typer.Option(42, help="Random seed"),
    iteration: int = _ITER,
    emit_json: bool = _EMIT,
    trim_outliers: bool = _TRIM,
    trim_lower: float = _TRIML,
    trim_upper: float = _TRIMU,
    trim_on: str = _TRIMON,
):
    """Phase 2 (site): fit a responder classifier -> ROC AUC (the only model here that has one)."""
    _fit_logistic(site=site, panel=panel, features=str(features),
                  penalty=penalty, alpha=alpha, responder_threshold=responder_threshold,
                  n_boot=n_boot, seed=seed, iteration=iteration, emit_json=emit_json,
                  trim=_trim(trim_outliers, trim_lower, trim_upper, trim_on),
                  **_files(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id, cohort_sql, cohort_sql_file, cohort_target_sql, cohort_target_sql_file, max_rows, cb_profile))


@app.command("fit-models")
def fit_models_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    features: Path = typer.Option(..., help="Feature set to fit on: CSV (column 'feature') or feature-space JSON"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    cohort_id: Optional[str] = _COHORT,
    cohort_sql: Optional[str] = _COHORT_SQL,
    cohort_sql_file: Optional[Path] = _COHORT_SQL_FILE,
    cohort_target_sql: Optional[str] = _COHORT_TGT,
    cohort_target_sql_file: Optional[Path] = _COHORT_TGT_FILE,
    max_rows: Optional[int] = _MAX_ROWS,
    cb_profile: str = _CB_PROFILE,
    ridge_alpha: float = typer.Option(1.0, help="Ridge L2 penalty"),
    lasso_alpha: float = typer.Option(0.008, help="LASSO L1 penalty"),
    n_trees: int = typer.Option(200, help="Random Forest trees"),
    with_logistic: bool = typer.Option(False, "--with-logistic", help="Also fit the responder classifier (adds ROC AUC)"),
    responder_threshold: float = typer.Option(
        DEFAULT_RESPONDER_THRESHOLD, "--responder-threshold",
        help="Responder cutoff on raw C-peptide AUC in ng/mL, for --with-logistic"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the 95% CIs"),
    seed: int = typer.Option(42, help="Random seed"),
    iteration: int = _ITER,
    emit_json: bool = _EMIT,
    trim_outliers: bool = _TRIM,
    trim_lower: float = _TRIML,
    trim_upper: float = _TRIMU,
    trim_on: str = _TRIMON,
):
    """Phase 2 (site): convenience — runs fit-ridge, fit-lasso, fit-rf (and fit-logistic) on one feature set."""
    _fit_models(site=site, panel=panel, features=str(features),
                ridge_alpha=ridge_alpha, lasso_alpha=lasso_alpha, n_trees=n_trees,
                with_logistic=with_logistic, responder_threshold=responder_threshold,
                n_boot=n_boot, seed=seed, iteration=iteration, emit_json=emit_json,
                trim=_trim(trim_outliers, trim_lower, trim_upper, trim_on),
                **_files(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id, cohort_sql, cohort_sql_file, cohort_target_sql, cohort_target_sql_file, max_rows, cb_profile))


# ---------------------------------------------------------------- site: Phase 3
@app.command("apply-coefficients")
def apply_coefficients_command(
    site: str = typer.Option(..., help="Study id, e.g. SDY524 — the site whose outcome this is"),
    panel: str = typer.Option("B", help="Feature panel A|B"),
    ridge_vector: Optional[Path] = typer.Option(None, "--ridge-vector", help="Federated ridge result: vector CSV or payload JSON"),
    lasso_vector: Optional[Path] = typer.Option(None, "--lasso-vector", help="Federated lasso result: vector CSV or payload JSON"),
    rf_union: Optional[Path] = typer.Option(None, "--rf-union", help="Federated RF result: union pickle or payload JSON"),
    tidy: Optional[Path] = _TIDY,
    aa: Optional[Path] = _AA,
    demo: Optional[Path] = _DEMO,
    cpeptide: Optional[Path] = _CPEP,
    arms: Optional[Path] = _ARMS,
    arm_subjects: Optional[Path] = _ARMSUBJ,
    cohort_id: Optional[str] = _COHORT,
    cohort_sql: Optional[str] = _COHORT_SQL,
    cohort_sql_file: Optional[Path] = _COHORT_SQL_FILE,
    cohort_target_sql: Optional[str] = _COHORT_TGT,
    cohort_target_sql_file: Optional[Path] = _COHORT_TGT_FILE,
    max_rows: Optional[int] = _MAX_ROWS,
    cb_profile: str = _CB_PROFILE,
    ridge_alpha: float = typer.Option(1.0, help="Ridge L2 penalty for the solo model"),
    lasso_alpha: float = typer.Option(0.008, help="LASSO L1 penalty for the solo model"),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for the 95% CIs"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Phase 3 (site): this site's own outcome using the federated results (solo vs federated)."""
    _apply_coefficients(
        site=site, panel=panel,
        ridge_vector=str(ridge_vector) if ridge_vector else None,
        lasso_vector=str(lasso_vector) if lasso_vector else None,
        rf_union=str(rf_union) if rf_union else None,
        ridge_alpha=ridge_alpha, lasso_alpha=lasso_alpha, n_boot=n_boot, seed=seed,
        **_files(tidy, aa, demo, cpeptide, arms, arm_subjects, cohort_id, cohort_sql, cohort_sql_file, cohort_target_sql, cohort_target_sql_file, max_rows, cb_profile),
    )


# ------------------------------------------------------------ aggregator: Phase 1
@app.command("consensus-features")
def consensus_features_command(
    features: List[Path] = typer.Option(..., "--features", help="Per-site selected-features CSV (repeat --features for each site)"),
    emit_json: bool = typer.Option(False, "--emit-json", help="Also write the feature space as JSON, for the sites to fit on"),
    iteration: int = _ITER,
):
    """Aggregator: the consensus feature set = the intersection of the per-site selections."""
    _consensus_features(features=[str(f) for f in features],
                        emit_json=emit_json, iteration=iteration)


# ------------------------------------------------------------ aggregator: Phase 2
@app.command("aggregate-vectors")
def aggregate_vectors_command(
    vector: List[Path] = typer.Option(..., "--vector", help="Per-site result: payload JSON, coefficient CSV, or RF .pkl (repeat for each)"),
    method: str = typer.Option("fedavg", help="Combine rule: fedavg | median | mean"),
    iteration: Optional[int] = typer.Option(None, "--iteration", help="Output round (default: one past the inputs')"),
):
    """Phase 2 (aggregator): combine the site results. JSON in -> one federated JSON per algorithm."""
    _aggregate_vectors(vectors=[str(f) for f in vector], method=method, iteration=iteration)


# ------------------------------------------------------- site: Phase 0 (discovery)
@app.command("describe-cohort")
def describe_cohort_command(
    cohort_id: str = typer.Option(..., "--cohort-id", help="CloudOS Cohort Browser cohort id (from the cohort's URL)"),
    cb_profile: str = _CB_PROFILE,
):
    """Phase 0 (site): print a cohort's schemas, tables and columns.

    Run this before writing any --cohort-sql. There is no list-cohorts call and no
    way to discover a cohort's shape offline, so this is how you find out which
    tables, concept ids and value columns the query has to select.
    """
    from .cohort_data import describe_cohort as _describe_cohort
    _describe_cohort(cohort_id, profilename=cb_profile)


def main():
    app()


if __name__ == "__main__":
    main()
