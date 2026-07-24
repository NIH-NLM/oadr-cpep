[![Build and Deploy Sphinx Documentation](https://github.com/NIH-NLM/oadr-cpep/actions/workflows/docs.yml/badge.svg)](https://github.com/NIH-NLM/oadr-cpep/actions/workflows/docs.yml)
[![Build and Push Docker Image](https://github.com/NIH-NLM/oadr-cpep/actions/workflows/docker-build.yml/badge.svg)](https://github.com/NIH-NLM/oadr-cpep/actions/workflows/docker-build.yml)
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
| `apply-coefficients` | site · Phase 3 | this site's OWN outcome using the federated results — solo vs federated across Ridge/LASSO/RF (`--ridge-vector` / `--lasso-vector` / `--rf-union`), bootstrap 95% CI, combined graphic |
| `consensus-features` | aggregator · Phase 1 | tally the given per-site selections (`--features`, repeat per site) into a consensus feature set |
| `aggregate-vectors` | aggregator · Phase 2 | combine the given per-site vectors/forests (`--vector`, repeat per file) — FedAvg / median / mean + union of forests |

The federated round trip:

```
site: select-features ──selected──▶ agg: consensus-features ──consensus──▶ site: fit-models
site: fit-models      ──vectors ──▶ agg: aggregate-vectors  ──federated──▶ site: apply-coefficients
```

## Data

Data is read through the embedded `oadr_data` loader — the same one the
oadr-autoantibody notebooks use — so Panel A / Panel B are built identically. Each
site command takes `--site` (the study, e.g. `SDY524`), `--panel` (`A` = legacy 9
features, `B` = extended 12), and the **explicit input files that panel needs** —
no directory, no glob, nothing resolved by name:

| Option | Panel | File |
|---|---|---|
| `--tidy` | A | `SDY<n>_tidy.csv` (features) |
| `--cpeptide` | A & B | `SDY<n>_cpeptide_auc_tidy.csv` (C-peptide AUC target) |
| `--aa` | B | `aa_<n>.csv` (autoantibodies + anthropometrics) |
| `--demo` | B | `demo_<n>.csv` (demographics) |
| `--arms` | B (optional) | `SDY<n>_arm_or_cohort.txt` (treatment: subject → arm → treatment) |
| `--arm-subjects` | B (optional) | `SDY<n>_arm_2_subject.txt` |

Omitting `--arms` / `--arm-subjects` leaves `received_active_treatment`
undetermined (0 for all) — as for SDY1737, which has no treatment arms.

The aggregator commands take their inputs the same way — explicit files, no
directory: `consensus-features --features …` (repeat per site) and
`aggregate-vectors --vector …` (repeat per per-site vector / forest). Their
output names (the `from-<src>` / `panel<X>` tags) are derived from the input
files' own provenance columns, so panels and feature sources never mix.

## Quickstart — a full local round trip

Each site selects on its **own** data; the aggregator combines those; each site
then applies the federated result to its own data. **Everything is explicit files
— no directories.** Run from the repo root (data in `data/`), outputs to the
current dir (a scratch dir is tidiest). Two sites shown (SDY524, SDY569); add
more the same way.

```bash
# 1. Each site selects features on its OWN data (Panel B = 5 files).
#    -> <site>_panelB_selected_features.csv (+ _lasso_selection.csv)
oadr-cpep select-features --site SDY524 --panel B \
  --aa data/aa_524.csv --demo data/demo_524.csv --cpeptide data/SDY524_cpeptide_auc_tidy.csv \
  --arms data/SDY524_arm_or_cohort.txt --arm-subjects data/SDY524_arm_2_subject.txt
oadr-cpep select-features --site SDY569 --panel B \
  --aa data/aa_569.csv --demo data/demo_569.csv --cpeptide data/SDY569_cpeptide_auc_tidy.csv \
  --arms data/SDY569_arm_or_cohort.txt --arm-subjects data/SDY569_arm_2_subject.txt

# 2. Aggregator builds the consensus from the explicit per-site selections
#    (--min-sites 1 = union of all selections). -> consensus_panelB_features.csv
oadr-cpep consensus-features --min-sites 1 \
  --features SDY524_panelB_selected_features.csv \
  --features SDY569_panelB_selected_features.csv

# 3. Each site fits all three methods on the chosen feature set (its own 5 files).
#    Outputs are tagged with the feature source (here 'consensus'), e.g.
#    <site>_from-consensus_panelB_{ridge,lasso}_vector.csv, _rf.pkl (+ metrics + graphics)
oadr-cpep fit-models --site SDY524 --panel B --features consensus_panelB_features.csv \
  --aa data/aa_524.csv --demo data/demo_524.csv --cpeptide data/SDY524_cpeptide_auc_tidy.csv \
  --arms data/SDY524_arm_or_cohort.txt --arm-subjects data/SDY524_arm_2_subject.txt
oadr-cpep fit-models --site SDY569 --panel B --features consensus_panelB_features.csv \
  --aa data/aa_569.csv --demo data/demo_569.csv --cpeptide data/SDY569_cpeptide_auc_tidy.csv \
  --arms data/SDY569_arm_or_cohort.txt --arm-subjects data/SDY569_arm_2_subject.txt

# 4. Aggregator combines the explicit per-site vectors + forests (FedAvg + union of
#    forests). The from-/panel tags are derived from the files themselves ->
#    federated_from-consensus_panelB_{ridge,lasso}_fedavg_vector.csv, _rf_union.pkl
oadr-cpep aggregate-vectors --method fedavg \
  --vector SDY524_from-consensus_panelB_ridge_vector.csv --vector SDY569_from-consensus_panelB_ridge_vector.csv \
  --vector SDY524_from-consensus_panelB_lasso_vector.csv --vector SDY569_from-consensus_panelB_lasso_vector.csv \
  --vector SDY524_from-consensus_panelB_rf.pkl           --vector SDY569_from-consensus_panelB_rf.pkl

# 5. Each site's OWN outcome using the federated results — pass them explicitly.
#    -> <site>_panelB_federated_metrics.csv (solo vs federated) + _federated.{png,svg,html}
oadr-cpep apply-coefficients --site SDY524 --panel B \
  --aa data/aa_524.csv --demo data/demo_524.csv --cpeptide data/SDY524_cpeptide_auc_tidy.csv \
  --arms data/SDY524_arm_or_cohort.txt --arm-subjects data/SDY524_arm_2_subject.txt \
  --ridge-vector federated_from-consensus_panelB_ridge_fedavg_vector.csv \
  --lasso-vector federated_from-consensus_panelB_lasso_fedavg_vector.csv \
  --rf-union     federated_from-consensus_panelB_rf_union.pkl
```

**Panel A** needs only two files — swap the five Panel-B files for
`--tidy data/SDY524_tidy.csv --cpeptide data/SDY524_cpeptide_auc_tidy.csv`.

The site commands take `--panel`; the aggregator derives the panel (and feature
source) from the files themselves and refuses to mix them — so Panel A and Panel B
runs can share a working directory. In a real federated deployment each site runs
its own steps at its own institution (only ever its own data); the aggregator sees
only the parameter files — never subject-level data.

### Single-site (bespoke) consensus

When the result is driven by one dominant study rather than a genuine multi-site
agreement, use that site's selection *as* the consensus — honest and
reproducible — instead of relying on a tally threshold:

```bash
oadr-cpep consensus-features --from-site SDY524 \
  --features SDY524_panelB_selected_features.csv \
  --features SDY569_panelB_selected_features.csv
#   -> consensus_panelB_features.csv = exactly SDY524's selected features
```

Everything downstream (`fit-models`, `aggregate-vectors`, `apply-coefficients`)
is unchanged — it just proceeds on that singular feature set. (Equivalently, you
can skip the consensus step and feed one site's list straight in:
`fit-models … --features SDY524_panelB_selected_features.csv`.)

### Feature-source tags (`from-<src>`)

Every fit output is stamped with the feature source it was fit on
(`<site>_from-<src>_panelB_…`, where `<src>` is the leading token of the
`--features` filename). To combine several sites on **one** feature source, fit
them all on the *same* features file, then hand those vectors to
`aggregate-vectors` — it reads the `from-<src>` / `panel` provenance **from the
files**, tags its output the same way, and errors if you mix sources or panels.

Example — aggregate SDY524 **and** SDY569, both fit on **SDY524's** features:

```bash
# fit EACH site on SDY524's features  ->  <site>_from-SDY524_panelB_*
oadr-cpep fit-models --site SDY524 --panel B --outdir fit \
    --features SDY524_panelB_selected_features.csv \
    --aa data/aa_524.csv --demo data/demo_524.csv --cpeptide data/SDY524_cpeptide_auc_tidy.csv \
    --arms data/SDY524_arm_or_cohort.txt --arm-subjects data/SDY524_arm_2_subject.txt
oadr-cpep fit-models --site SDY569 --panel B --outdir fit \
    --features SDY524_panelB_selected_features.csv \
    --aa data/aa_569.csv --demo data/demo_569.csv --cpeptide data/SDY569_cpeptide_auc_tidy.csv \
    --arms data/SDY569_arm_or_cohort.txt --arm-subjects data/SDY569_arm_2_subject.txt

# combine the explicit vectors/forests (from-SDY524 derived from the files)
oadr-cpep aggregate-vectors --method fedavg --outdir aggregated \
    --vector fit/SDY524_from-SDY524_panelB_ridge_vector.csv --vector fit/SDY569_from-SDY524_panelB_ridge_vector.csv \
    --vector fit/SDY524_from-SDY524_panelB_lasso_vector.csv --vector fit/SDY569_from-SDY524_panelB_lasso_vector.csv \
    --vector fit/SDY524_from-SDY524_panelB_rf.pkl           --vector fit/SDY569_from-SDY524_panelB_rf.pkl
#   -> aggregated/federated_from-SDY524_panelB_ridge_fedavg_vector.csv  (+ lasso, rf_union)

# SDY524's own outcome, tuned with those federated results — pass them explicitly
oadr-cpep apply-coefficients --site SDY524 --panel B \
    --aa data/aa_524.csv --demo data/demo_524.csv --cpeptide data/SDY524_cpeptide_auc_tidy.csv \
    --arms data/SDY524_arm_or_cohort.txt --arm-subjects data/SDY524_arm_2_subject.txt \
    --ridge-vector aggregated/federated_from-SDY524_panelB_ridge_fedavg_vector.csv \
    --lasso-vector aggregated/federated_from-SDY524_panelB_lasso_fedavg_vector.csv \
    --rf-union     aggregated/federated_from-SDY524_panelB_rf_union.pkl
```

Hand `aggregate-vectors` vectors fit on *different* sources and it stops with
`mix feature sources` — pass one source's vectors at a time.

## Method — this implementation vs. the notebook spec

The math here is deliberately *not* identical to the
[oadr-autoantibody](https://github.com/NIH-NLM/oadr-autoantibody/tree/main/ipynb)
Stage-2 notebooks. Both use **FedAvg**; they differ in *where the average is
formed and how the federated arm is scored*. The difference is small in numbers
but matters for how you describe the result, so it is spelled out here.

**Notation.** Site `s` holds `(X_s, y_s)` with `N_s` subjects and fits a full-data
model → coefficient vector `β_s` (Ridge/LASSO) or forest `F_s` (RF). Only `β_s`
/ `F_s` ever leave the site — never `(X_s, y_s)`.

**`aggregate-vectors` — FedAvg (central).** The coordinator forms **one** vector,
weighted by cohort size:

```
β̄ = ( Σ_s N_s · β_s ) / ( Σ_s N_s )        # method=fedavg (size-weighted mean)
                                            # method=median / mean = unweighted
RF: F̄ = union of the per-site forests F_s  # trees concatenated, not averaged
```

**`apply-coefficients` — solo vs. federated at site `s` (5-fold CV).**

- *Solo* — for each fold, refit `β_s^(train)` on the training rows and predict the
  held-out rows. **Out-of-fold** → honest.
- *Federated (this implementation)* — take the **fixed** central `β̄` from the
  aggregator; for each fold, scale the held-out rows with the training-fold scaler
  and predict with `β̄`. `β̄` is **not** refit per fold.

**How the notebooks differ.** The notebooks form the federated vector **inside
each fold**, re-fitting the site out-of-fold and averaging with the *partner*
vectors only:

```
notebook (per fold k):  β_fed^(k) = ( N_train·β_s^(train,k) + Σ_{p≠s} N_p·β_p )
                                    / ( N_train + Σ_{p≠s} N_p )
this repo (all folds):  β_fed      = β̄ = ( N_s·β_s + Σ_{p≠s} N_p·β_p )
                                    / ( N_s + Σ_{p≠s} N_p )
```

Two concrete differences:

1. **Site's own contribution** — full-data `β_s` here vs. the training-fold
   `β_s^(train,k)` in the notebook.
2. **When it's formed** — a single fixed `β̄` here vs. recomputed every fold there.

**Consequence (read this before quoting numbers).** Because `β̄` contains the
site's *full-data* `β_s`, the federated arm's coefficients partly *saw* the
held-out rows they are scored on — a mild optimism (leakage) that the notebook's
out-of-fold refit avoids. So the solo→federated gap here mixes **two** effects:
genuine partner signal, *and* the federated arm using more of the site's own data
than the fold-refit solo arm. Frame the result as **"the deployed federated model
vs. the site's solo model,"** which is exactly true — not as "partner *X* improves
site *Y* by Δ," which over-attributes the gap. For a contrast that isolates the
partner contribution, use the notebook's per-fold, out-of-fold method.

**Why this implementation still chooses central-apply.** The model you actually
*deploy* is `β̄` — the single artifact that scores a new patient — and in a real
federated deployment (e.g. sites in separate Lifebit workspaces) each site needs
only that one central vector, not every partner's individual vector. Applying
`β̄` as-is is therefore both deployment-faithful and simplest to run federated.

**Why not "blend."** An earlier option mixed the site's own vector back in,
`w·β_s + (1−w)·β̄`. Since `β̄` *already* contains `β_s`, that counts the site
twice (double-counting) — so it was removed. Note the notebook's per-fold average
is **not** a blend: it combines `β_s^(train)` with the *partners*, each study once.

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

## Companion repositories

* Main Python Package: [oadr-cpep](https://github.com/NIH-NLM/oadr-cpep) (here)
* Site specific workflow: [oadr-cpep-fed-predict-site-nf](https://github.com/NIH-NLM/oadr-cpep-fed-predict-site-nf)
* Consensus and Aggregation specific workflow: [oadr-cpep-fed-predict-site-nf](https://github.com/NIH-NLM/oadr-cpep-fed-predict-aggregation-nf)

## Citation

Bhattacharya S, Dunn P, Thomas CG, Smith B, Schaefer H, Chen J, Hu Z, Zalocusky KA, Shankar RD, Shen-Orr SS, Thomson E, Wiser J, Butte AJ. ImmPort, toward repurposing of open access immunological assay data for translational and clinical research. Sci Data. 2018 Feb 27;5:180015. doi: 10.1038/sdata.2018.15. PMID: 29485622; PMCID: PMC5827693.

Kong YM, Dahlke C, Xiang Q, Qian Y, Karp D, Scheuermann RH. Toward an ontology-based framework for clinical research databases. J Biomed Inform. 2011 Feb;44(1):48-58. doi: 10.1016/j.jbi.2010.05.001. Epub 2010 May 10. PMID: 20460173; PMCID: PMC2953614.

The data supporting this publication is available at ImmPort (immport.org) under study accession SDY524, SDY569, SDY797, SDY1737.


