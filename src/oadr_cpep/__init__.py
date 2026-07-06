"""
oadr-cpep: Federated prediction of residual beta-cell function (C-peptide AUC)
in Type 1 Diabetes.

A single typer CLI drives both federated workflows:

  site       : select-features, fit-models, apply-coefficients
  aggregator : consensus-features, aggregate-vectors

Data is read through the embedded oadr_data loader (the same one the
oadr-autoantibody notebooks use). Only model parameters and scalar performance
summaries cross the site boundary — never subject-level data.
"""
# src/oadr_cpep/__init__.py

__version__ = "0.1.0"
