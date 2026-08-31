"""
Read a site's design matrix from a CloudOS Cohort Browser selection.

The alternative to ``oadr_data``'s file loaders: same return shape, different
source. Lifebit's Cohort Browser serves data already mapped to OMOP, so the
concept ids arrive with the rows and this module does not re-derive them — it
carries them through to the coefficient payloads, which is what lets two sites
combine the same clinical entity without agreeing on column names.

``cloudos_cb`` is an **optional** dependency, imported inside the functions that
use it. Nothing here is imported unless a ``--cohort-id`` is actually passed, so
the package installs and the whole pipeline still runs against local files.

Running requires the platform, not just credentials: per the client's own README,
calls fail unless Bastion is enabled for the workspace and the code runs from
inside a CloudOS interactive session in the same workspace as the cohort. The
``cohort_id`` itself comes from the Cohort Browser URL — the client has no
list-cohorts call. Use ``describe_cohort`` first to see what a cohort holds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import oadr_data as od
from .logging_config import setup_logger

logger = setup_logger("oadr_cpep")

# Long-format column names as they come back from an OMOP measurement query.
PERSON_KEY = "person_id"
CONCEPT_KEY = "measurement_concept_id"
VALUE_KEY = "value_as_number"

_IMPORT_HINT = (
    "reading a cohort needs the CloudOS client: pip install 'oadr-cpep[cloudos]'. "
    "It must run inside a CloudOS interactive session, in the same workspace as "
    "the cohort, with Bastion enabled."
)


def _client():
    """Import cloudos_cb on demand, with an actionable message when it is absent."""
    try:
        import cloudos_cb
    except ImportError as e:                      # pragma: no cover - env dependent
        raise SystemExit(f"{_IMPORT_HINT}\n({e})") from e
    return cloudos_cb


def describe_cohort(cohort_id, profilename=""):
    """The cohort's schemas, tables and columns — run this before writing any SQL.

    There is no way to discover a cohort's shape offline, and no list-cohorts
    call, so this is the entry point for finding out what a cohort actually holds.
    """
    cb = _client()
    tables = cb.cohort_tables(cohort_id, profilename=profilename)
    logger.info(str(tables))
    return tables


def run_query(cohort_id, sql, profilename="", max_rows=None):
    """Validate, count, then fetch — so a large cohort is never pulled by accident.

    ``query()`` submits one async task per 1000-row page, so the row count matters
    before the fetch, not after. ``max_rows`` refuses anything larger rather than
    quietly starting hundreds of tasks.
    """
    cb = _client()
    check = cb.sql_validate(sql, profilename=profilename)
    if not check.get("isValid", True):
        raise SystemExit(f"invalid SQL for cohort {cohort_id}: {check.get('error')}")

    n = cb.query_count(cohort_id, sql, profilename=profilename)
    logger.info(f"cohort {cohort_id}: query matches {n} row(s)")
    if max_rows is not None and n > max_rows:
        raise SystemExit(f"query matches {n} rows, above --max-rows {max_rows}; "
                         f"narrow the selection or raise the limit")
    return cb.query(cohort_id, sql, profilename=profilename)


def pivot_measurements(long_df, *, person_key=PERSON_KEY, concept_key=CONCEPT_KEY,
                       value_key=VALUE_KEY, agg="mean"):
    """Long OMOP measurement rows -> (wide frame, concepts).

    One row per person, one column per concept. Columns are named
    ``concept_<id>`` — the concept id is the identity that travels, and naming
    the column after it means nothing has to be mapped back later. ``concepts``
    maps each column to the metadata the payload carries.

    A person with several measurements of the same concept (repeat visits) is
    reduced by ``agg``; the default mean matches how the file loaders treat
    repeated assays.
    """
    d = long_df.copy()
    d[value_key] = pd.to_numeric(d[value_key], errors="coerce")
    wide = d.pivot_table(index=person_key, columns=concept_key,
                         values=value_key, aggfunc=agg)

    names = {}
    if "concept_name" in d.columns:
        names = d.drop_duplicates(concept_key).set_index(concept_key)["concept_name"].to_dict()
    domains = {}
    if "domain_id" in d.columns:
        domains = d.drop_duplicates(concept_key).set_index(concept_key)["domain_id"].to_dict()

    concepts = {}
    cols = {}
    for cid in wide.columns:
        col = f"concept_{int(cid)}"
        cols[cid] = col
        concepts[col] = {"concept_id": int(cid),
                         "concept_name": str(names.get(cid, col)),
                         "domain_id": str(domains.get(cid, "Measurement"))}
    wide = wide.rename(columns=cols).reset_index()
    return wide, concepts


def load_cohort(cohort_id, panel="B", *, sql=None, target_sql=None, cpeptide=None,
                profilename="", max_rows=None, agg="mean"):
    """Load one cohort -> (frame, feature_names, target), matching ``oadr_data``.

    Args:
        cohort_id: the Cohort Browser cohort (from its URL).
        panel: retained for symmetry with the file loaders; the feature set here
            is whatever concepts the cohort returns, not a fixed panel.
        sql: the measurement query. Must select at least person_id,
            measurement_concept_id and value_as_number; concept_name and
            domain_id are used when present. Required — the schema differs per
            cohort, so run ``describe_cohort`` and pass what fits rather than
            letting this module guess.
        target_sql: query returning person_id and the C-peptide AUC target.
        cpeptide: alternative to ``target_sql`` — a local target CSV, so a cohort
            whose export lacks the outcome can still be fit.
        max_rows: refuse a query matching more rows than this.

    The frame carries its concept metadata in ``frame.attrs["concepts"]``, which
    is what the fit steps put into the coefficient payloads.
    """
    if not sql:
        raise SystemExit("--cohort-sql is required: run `describe_cohort` to see the "
                         "cohort's tables and columns, then pass the query that "
                         "selects person_id, measurement_concept_id, value_as_number")

    long_df = run_query(cohort_id, sql, profilename=profilename, max_rows=max_rows)
    wide, concepts = pivot_measurements(long_df, agg=agg)
    logger.info(f"cohort {cohort_id}: {len(wide)} subject(s), {len(concepts)} concept(s)")

    if target_sql:
        tgt = run_query(cohort_id, target_sql, profilename=profilename, max_rows=max_rows)
        tgt = tgt.rename(columns={tgt.columns[0]: PERSON_KEY, tgt.columns[1]: "C_Peptide_AUC_4Hrs"})
    elif cpeptide:
        tgt = od._read_cpeptide(cpeptide).rename(columns={"Subject_ID": PERSON_KEY})
    else:
        raise SystemExit("no target given: pass --cohort-target-sql (the outcome from the "
                         "cohort) or --cpeptide (a local target file)")

    df = wide.merge(tgt, on=PERSON_KEY, how="inner")
    if df.empty:
        raise SystemExit(f"cohort {cohort_id}: no subjects join the target — check that the "
                         f"target's person key matches {PERSON_KEY}")

    df[od.PANEL_A_TARGET] = np.log(pd.to_numeric(df["C_Peptide_AUC_4Hrs"], errors="coerce"))
    df = df.dropna(subset=[od.PANEL_A_TARGET]).reset_index(drop=True)

    feats = sorted(concepts)
    df[feats] = df[feats].fillna(0.0).astype(float)
    df.attrs["concepts"] = concepts
    df.attrs["cohort_id"] = cohort_id
    return df, feats, od.PANEL_A_TARGET
