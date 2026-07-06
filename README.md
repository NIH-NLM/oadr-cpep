# oadr-cpep

`oadr-cpep` is a Python package for **federated prediction of residual beta-cell
function** (C-peptide AUC) in Type 1 Diabetes. It packages the methods
demonstrated in the
[oadr-autoantibody](https://github.com/NIH-NLM/oadr-autoantibody/tree/main/ipynb)
notebooks into one `typer` CLI, used by two Nextflow workflows —
[oadr-cpep-fed-predict-site-nf](https://github.com/NIH-NLM/oadr-cpep-fed-predict-site-nf)
(per institution) and
[oadr-cpep-fed-predict-aggregator-nf](https://github.com/NIH-NLM/oadr-cpep-fed-predict-aggregator-nf)
(coordinator).

Only model parameters (feature lists, coefficient vectors, trained forests) and
scalar performance summaries ever cross the site boundary — never subject-level
data.

## Install

```bash
pip install -e .
oadr-cpep --help
```

## Commands

One CLI (`oadr-cpep`) provides both the per-site and coordinator steps:

| Command | Role | Does |
|---|---|---|
| `select-features` | site · Phase 1 | LASSO selects features on the site's own data |
| `fit-models` | site · Phase 2 | Ridge / LASSO / Random Forest on the consensus features |
| `apply-coefficients` | site · Phase 3 | incorporate the central federated vector — solo-vs-federated 5-fold CV, bootstrap 95% CI, scatter |
| `consensus-features` | aggregator · Phase 1 | tally site selections into a consensus feature set |
| `aggregate-vectors` | aggregator · Phase 2 | combine site vectors (FedAvg / median / mean) + union of forests |

The federated round trip:

```
site: select-features ──selected──▶ agg: consensus-features ──consensus──▶ site: fit-models
site: fit-models      ──vectors ──▶ agg: aggregate-vectors  ──federated──▶ site: apply-coefficients
```

## Data

Data is read through the embedded `oadr_data` loader — the same one the
oadr-autoantibody notebooks use — so Panel A / Panel B are built identically. The
site steps take `--site` (the study, e.g. `SDY524`), `--panel` (`A` = legacy 9
features, `B` = extended 12), and `--data-root` (a directory of the flat
ImmPort-derived files, found by name):

```
SDY<study>_tidy.csv                  Panel A features
SDY<study>_cpeptide_auc_tidy.csv     the C-peptide AUC target (both panels)
aa_<id>.csv, demo_<id>.csv           Panel B extended features (524/569/1737)
SDY<study>_arm_or_cohort.txt         treatment (subject → arm → treatment)
SDY<study>_arm_2_subject.txt
```

Example:

```bash
oadr-cpep select-features --site SDY524 --panel B --data-root data/
oadr-cpep fit-models      --site SDY524 --panel B --data-root data/ --features consensus_features.csv
oadr-cpep apply-coefficients --site SDY524 --panel B --data-root data/ \
    --coefficients federated_ridge_fedavg_vector.csv
```

## Container

The Nextflow workflows reference a container built from this repo and published
to `ghcr.io/nih-nlm/oadr-cpep` by GitHub Actions (`.github/workflows/docker-build.yml`)
on push to `main` and on release.

```bash
docker build -t ghcr.io/nih-nlm/oadr-cpep:latest .
```

## Documentation

Sphinx (autodoc + RTD theme) API docs are built and deployed to GitHub Pages by
`.github/workflows/docs.yml`. Build locally:

```bash
pip install sphinx myst-parser sphinx-rtd-theme
sphinx-apidoc -f --separate -o docs/source/ src/oadr_cpep
cd docs && make html
```

## Layout

```
src/oadr_cpep/
  cli.py            typer CLI (thin command wrappers)
  site.py           per-site steps: select_features, fit_models, apply_coefficients
  aggregate.py      coordinator steps: consensus_features, aggregate_vectors
  oadr_data.py      Panel A / Panel B loader (ported from oadr-autoantibody)
  logging_config.py
pyproject.toml      package metadata + oadr-cpep entry point
Dockerfile          ghcr image (clones this repo, pip installs it)
docs/               sphinx (RTD theme) → GitHub Pages
tests/              CLI tests
```
