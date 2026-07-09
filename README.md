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
| `select-features` | site · Phase 1 | LASSO selects features on the site's own data (alpha chosen by CV) |
| `fit-ridge` / `fit-lasso` / `fit-rf` | site · Phase 2 | fit one method on a given feature set → coefficient vector / forest + 5-fold CV MSE/R² + a fit graphic (png/svg/html) |
| `fit-models` | site · Phase 2 | convenience — runs all three fits on the same feature set |
| `apply-coefficients` | site · Phase 3 | this site's OWN outcome using the federated results — solo vs federated across Ridge/LASSO/RF (`--coefficients-dir`), bootstrap 95% CI, combined graphic |
| `consensus-features` | aggregator · Phase 1 | tally site selections into a consensus feature set |
| `aggregate-vectors` | aggregator · Phase 2 | combine site vectors (FedAvg / median / mean) + union of forests; `--panel` / `--from` scope which vectors combine |

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

## Quickstart — a full local round trip

Federated feature selection is not "one site's picks pushed to the others."
Each site selects on its **own** data; the aggregator combines those into a
**consensus**; that consensus is what every site then fits on.

Run from the repo root (so `--data-root data` resolves), letting each command
pick its **default output name** — outputs land in the current directory, so a
scratch dir is tidiest. The three Panel-B studies are SDY524, SDY569, SDY1737.

```bash
# 1. Each site selects on its OWN data. Two outputs per site:
#      <site>_panelB_lasso_selection.csv    full LASSO result (all features + coefficients)
#      <site>_panelB_selected_features.csv  just the selected features (feeds fit)
oadr-cpep select-features --site SDY524  --panel B --data-root data
oadr-cpep select-features --site SDY569  --panel B --data-root data
oadr-cpep select-features --site SDY1737 --panel B --data-root data

# 2. Aggregator builds the consensus (panel-scoped) -> consensus_panelB_features.csv
#    (majority of sites by default; --min-sites 1 = union of all selections)
oadr-cpep consensus-features --input-dir . --panel B --outdir .

# 3. Each site fits all three methods (or call fit-ridge / fit-lasso / fit-rf
#    separately). Every output is tagged with the feature source (here: consensus)
#    so you see what was used and nothing overwrites — per method:
#      <site>_from-consensus_panelB_ridge_vector.csv, _ridge_fit_metrics.csv,
#      _ridge_fit.{png,svg,html}   (and _lasso_*, _rf*)
oadr-cpep fit-models --site SDY524  --panel B --data-root data --features consensus_panelB_features.csv
oadr-cpep fit-models --site SDY569  --panel B --data-root data --features consensus_panelB_features.csv
oadr-cpep fit-models --site SDY1737 --panel B --data-root data --features consensus_panelB_features.csv

# 4. Aggregator combines the vectors (FedAvg) + union of forests -> federated_panelB_ridge_fedavg_vector.csv
oadr-cpep aggregate-vectors --input-dir . --panel B --method fedavg --outdir .

# 5. Each site's OWN outcome using the federated results — all three methods
#    (ridge/lasso vectors + rf union, auto-discovered from --coefficients-dir).
#    Writes <site>_panelB_federated_metrics.csv (solo vs federated per method)
#    and <site>_panelB_federated.{png,svg,html} (3 methods x solo|federated).
oadr-cpep apply-coefficients --site SDY524  --panel B --data-root data --coefficients-dir .
oadr-cpep apply-coefficients --site SDY569  --panel B --data-root data --coefficients-dir .
oadr-cpep apply-coefficients --site SDY1737 --panel B --data-root data --coefficients-dir .
```

Because every command is scoped with `--panel`, Panel A and Panel B runs can
share one working directory without ever mixing. In a real federated deployment
each site runs its own steps at its own institution (it only ever has its own
data); the aggregator sees only the parameter files — never subject-level data.
Here you drive all sites from one machine for testing.

### Single-site (bespoke) consensus

When the result is driven by one dominant study rather than a genuine multi-site
agreement, use that site's selection *as* the consensus — honest and
reproducible — instead of relying on a tally threshold:

```bash
oadr-cpep consensus-features --input-dir . --panel B --from-site SDY524 --outdir .
#   -> consensus_panelB_features.csv = exactly SDY524's selected features
```

Everything downstream (`fit-models`, `aggregate-vectors`, `apply-coefficients`)
is unchanged — it just proceeds on that singular feature set. (Equivalently, you
can skip the consensus step and feed one site's list straight in:
`fit-models … --features SDY524_panelB_selected_features.csv`.)

### Scoping to one feature source (`--from`)

Fit outputs are stamped with the feature source (`<site>_from-SDY524_panelB_…`).
When several sources coexist in one directory — e.g. you fit some sites on
SDY524's features and others on their own — scope the aggregate and apply with
`--from` so nothing mixes, and the federated files say which source they're built
on:

```bash
# combine only the vectors/forests fit on SDY524's features
oadr-cpep aggregate-vectors --input-dir . --panel B --from SDY524 --method fedavg --outdir .
#   -> federated_from-SDY524_panelB_ridge_fedavg_vector.csv  (+ lasso, rf_union)

# SDY524's own outcome, tuned with those federated results (all three methods)
oadr-cpep apply-coefficients --site SDY524 --panel B --data-root data --coefficients-dir . --from SDY524
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
src/oadr_cpep/        one function per process, grouped into logical modules
  cli.py            typer CLI (thin command wrappers)
  select.py         select_features                          (Phase 1)
  fit.py            fit_ridge, fit_lasso, fit_rf, fit_models   (Phase 2)
  apply.py          apply_coefficients                        (Phase 3, site outcome)
  aggregate.py      consensus_features, aggregate_vectors     (aggregator)
  plot.py           all graphics (matplotlib PNG/SVG, plotly HTML)
  common_utils.py   load / within-site scale / CV / R² / bootstrap
  oadr_data.py      Panel A / Panel B loader (ported from oadr-autoantibody)
  logging_config.py
pyproject.toml      package metadata + oadr-cpep entry point
Dockerfile          ghcr image (clones this repo, pip installs it)
docs/               sphinx (RTD theme) → GitHub Pages
tests/              CLI tests
```
