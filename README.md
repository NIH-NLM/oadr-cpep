# oadr-cpep

oadr-cpep is a python package that was built based upon the work demonstrated in [oadr-autoantibody](https://github.com/NIH-NLM/oadr-autoantibody/tree/main/ipynb)

## Three phases (one reusable workflow)

The artifact you pass selects the phase:

**Phase 1 — feature selection** (no artifact):

Runs LASSO on the site's data and emits `SDY524_selected_features.csv`. The
aggregator collects every site's selection and broadcasts back a
`consensus_features.csv`.

**Phase 2 — fit on consensus features** (`--consensus_features`):
```bash
```
Fits Ridge, LASSO, and Random Forest on the consensus features — LASSO
selection is not repeated — and emits `SDY524_ridge_vector.csv`,
`SDY524_lasso_vector.csv`, and `SDY524_rf.pkl`. These go to the aggregator.

**Phase 3 — incorporate the federated coefficients** (`--federated_coefficients`):
```bash
```

Takes the aggregator's central FedAvg vector and evaluates it from this site's
own view — a 5-fold CV comparing the site's **solo** model against the
**federated** model, with bootstrap 95% CIs on R² and an observed-vs-predicted
scatter. The central vector is applied **as-is**: it already includes this
site's contribution, so it is not re-blended with the site's own coefficients
(that would double-count this site). Emits
`SDY524_ridge_federated_performance.csv` (solo/federated R² + CIs, meant to
leave the site), `SDY524_ridge_federated_predictions.csv` (subject-level, kept
local), and `SDY524_ridge_federated.{png,pdf}`.

The method (`ridge`/`lasso`) is read from the vector; Random Forest is deferred
(its federated form — union of forests — is an aggregator method).

## Input data

The workflow reads the **same ImmPort-derived files** the oadr-autoantibody
notebooks use (via the embedded `oadr_data` loader) — you do not pre-build a
CSV. `--data_files` is a **glob**; every match is staged **flat** into each
process work dir, so nothing depends on a directory layout on ephemeral AWS spot
nodes. Upload the files flat and point the glob at them:

```
SDY<study>_tidy.csv                  Panel A features (per study)
SDY<study>_cpeptide_auc_tidy.csv     the C-peptide AUC target (both panels)
aa_<id>.csv, demo_<id>.csv           Panel B extended features (ids 524/569/1737)
SDY<study>_arm_or_cohort.txt         treatment (subject → arm → treatment)
SDY<study>_arm_2_subject.txt
```


## This Python Package

A self-contained image, `container/oadr-cpep/`, provides the per-site
`oadr-cpep-cli` (subcommands `select-features`, `fit-models`,
`apply-coefficients`) plus the embedded `oadr_data` loader. Build once and
publish; the site workflow references it. The aggregator step has its own image
in **oadr-cpep-fed-predict-aggregator-nf**.

```bash
docker build -t ghcr.io/nih-nlm/oadr-cpep:0.1.0 container/oadr-cpep/
```

## Layout


## Parameters

