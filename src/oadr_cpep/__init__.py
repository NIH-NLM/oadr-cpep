"""
oadr-cpep: Federated prediction of residual beta-cell function (C-peptide AUC)
in Type 1 Diabetes.

One typer CLI; one function per process, grouped into logical modules:

  select.py     : select_features                       (Phase 1)
  fit.py        : fit_ridge, fit_lasso, fit_rf, fit_models (Phase 2)
  apply.py      : apply_coefficients                     (Phase 3, site outcome)
  aggregate.py  : consensus_features, aggregate_vectors  (aggregator)
  plot.py       : all graphics (matplotlib PNG/SVG, plotly HTML)
  common_utils.py, oadr_data.py, logging_config.py       (shared)

Only model parameters and scalar performance summaries cross the site boundary —
never subject-level data.
"""
# src/oadr_cpep/__init__.py

__version__ = "0.1.0"
