"""
Shared data loaders for OADR autoantibody work — ported from the oadr-autoantibody
repo (src/oadr_data.py) so oadr-cpep builds Panel A / Panel B exactly the same way.

Change from the source: instead of finding files by name under a data root, every
loader takes the **explicit file paths** it needs — nothing is guessed. The CLI
hands each command its precise input files; this module reads them and parses.

Two feature panels:

    Panel A (legacy, 9 features): MIAA, GAD65, IA2IC, ICA, ZNT8, 8-12, 13-17,
    >18, Sex. Studies: SDY524, SDY569, SDY797, SDY1737.
    Files: SDY<n>_tidy.csv (features), SDY<n>_cpeptide_auc_tidy.csv (target).

    Panel B (extended): Sex, age_years, disease_duration_years, bmi, height_cm,
    weight_kg, GAD65, IA2IC, MIAA, ZNT8, ICA, received_active_treatment.
    Studies: SDY524, SDY569, SDY1737 only (no extended data for SDY797).
    Files: aa_<n>.csv, demo_<n>.csv, SDY<n>_cpeptide_auc_tidy.csv,
    SDY<n>_arm_or_cohort.txt, SDY<n>_arm_2_subject.txt (arms optional).

Target for both: log(C_Peptide_AUC_4Hrs) = log_auc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PANEL_A_FEATURES = [
    "MIAA", "GAD65", "IA2IC", "ICA", "ZNT8",
    "8-12", "13-17", ">18", "Sex",
]
PANEL_A_TARGET = "log_auc"
PANEL_A_RAW_TARGET = "C_Peptide_AUC_4Hrs"

# Canonical extended panel (matches the notebooks' PANEL_B_FEATS — the 12
# comparable columns; race/ethnicity one-hots from the design matrix are dropped).
PANEL_B_FEATURES = [
    "Sex", "age_years", "disease_duration_years",
    "bmi", "height_cm", "weight_kg",
    "GAD65", "IA2IC", "MIAA", "ZNT8", "ICA",
    "received_active_treatment",
]

PANEL_A_STUDIES = ["SDY524", "SDY569", "SDY797", "SDY1737"]
PANEL_B_STUDIES = ["SDY524", "SDY569", "SDY1737"]


def _normalize_property(p: str) -> str:
    """Map per-study property labels to canonical names."""
    p = p.strip()
    if p.upper() == "IA_2IC":
        return "IA2IC"
    return p.upper() if p.upper() in {"GAD65", "IA2IC", "MIAA", "ICA", "ZNT8"} else p


def _normalize_age_group(a: str) -> str:
    """Map age-group strings into the legacy {'8-12', '13-17', '>18'} schema."""
    if a in ("18-30", ">30", ">18"):
        return ">18"
    return a


def _read_cpeptide(path) -> pd.DataFrame:
    """Load a c-peptide AUC tidy file; normalize columns to (Subject_ID, C_Peptide_AUC_4Hrs)."""
    df = pd.read_csv(path)
    rename = {
        "ImmPort Accession": "Subject_ID",
        "Subject_IDel": "Subject_ID",  # SDY569 typo
        "C_Peptide_AUC": "C_Peptide_AUC_4Hrs",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df[["Subject_ID", "C_Peptide_AUC_4Hrs"]]


def load_panel_a(study, tidy_path, cpeptide_path) -> pd.DataFrame:
    """Build the 9-feature panel for one study from its explicit files.

    Returns a DataFrame with columns:
        Subject_ID, Study, <PANEL_A_FEATURES>, C_Peptide_AUC_4Hrs, log_auc
    Antibodies absent from the study are filled with 0.0.
    """
    if study not in PANEL_A_STUDIES:
        raise ValueError(f"Unknown study {study!r} for Panel A")

    feat = pd.read_csv(tidy_path)
    feat = feat.rename(columns={"Accession": "Subject_ID"})
    feat["Property"] = feat["Property"].map(_normalize_property)
    feat["Age_Group"] = feat["Age_Group"].astype(str).map(_normalize_age_group)

    wide = feat.pivot_table(
        index=["Subject_ID", "Sex"],
        columns="Property",
        values="Value",
    ).reset_index()
    wide["Sex"] = wide["Sex"].map({"Male": 0, "Female": 1}).astype(float)

    age = (
        pd.get_dummies(feat[["Subject_ID", "Age_Group"]], columns=["Age_Group"])
        .groupby("Subject_ID")
        .max()
        .reset_index()
    )
    for col in ("Age_Group_8-12", "Age_Group_13-17", "Age_Group_>18"):
        if col not in age.columns:
            age[col] = 0
    age = age.rename(columns={
        "Age_Group_8-12": "8-12",
        "Age_Group_13-17": "13-17",
        "Age_Group_>18": ">18",
    })
    wide = wide.merge(age[["Subject_ID", "8-12", "13-17", ">18"]], on="Subject_ID", how="left")

    cpep = _read_cpeptide(cpeptide_path)
    df = wide.merge(cpep, on="Subject_ID", how="inner")

    for ab in ("MIAA", "GAD65", "IA2IC", "ICA", "ZNT8"):
        if ab not in df.columns:
            df[ab] = 0.0
    df[PANEL_A_FEATURES] = df[PANEL_A_FEATURES].fillna(0.0).astype(float)

    df["log_auc"] = np.log(df["C_Peptide_AUC_4Hrs"])
    df["Study"] = study
    return df[["Subject_ID", "Study"] + PANEL_A_FEATURES + [PANEL_A_RAW_TARGET, PANEL_A_TARGET]]


# ---------- Panel B (extended) ----------

_JEFF_AA_MAP = {
    "gad65": "GAD65",
    "ia_2ic": "IA2IC",
    "miaa": "MIAA",
    "zn_t8": "ZNT8",
}


def _treatment_from_arms(arms_path, arm_subjects_path, subject_ids):
    """Per-subject active-treatment flag by transitive closure: subject -> arm
    (arm_2_subject) -> treatment, where the control arm is identified by
    name/description (placebo / control / no treatment) and the treatment arm is
    the other. Works across drugs (hOKT3, teplizumab, alefacept). If the arm files
    are not supplied (or a study has no control arm, e.g. SDY1737 arms are age
    groups) treatment is undetermined and all subjects are 0. Returns a 0/1 Series
    aligned to subject_ids."""
    subject_ids = list(subject_ids)
    zero = pd.Series([0.0] * len(subject_ids))
    if not (arms_path and arm_subjects_path
            and Path(arms_path).exists() and Path(arm_subjects_path).exists()):
        return zero                      # arm files not provided -> treatment undetermined
    arms = pd.read_csv(arms_path, sep="\t")
    a2s = pd.read_csv(arm_subjects_path, sep="\t")
    ctrl_pat = "placebo|control|no treatment"
    ctrl_mask = (arms["NAME"].str.contains(ctrl_pat, case=False, na=False)
                 | arms["DESCRIPTION"].str.contains(ctrl_pat, case=False, na=False))
    if not ctrl_mask.any():
        return zero                      # no control arm -> treatment undetermined
    ctrl_arms = set(arms.loc[ctrl_mask, "ARM_ACCESSION"])
    treat_arms = set(arms["ARM_ACCESSION"]) - ctrl_arms
    sub2treat = {r.SUBJECT_ACCESSION: (1.0 if r.ARM_ACCESSION in treat_arms else 0.0)
                 for r in a2s.itertuples()}
    return pd.Series([sub2treat.get(s, 0.0) for s in subject_ids])


def _parse_date(s):
    return pd.to_datetime(s, errors="coerce")


def load_panel_b(study, aa_path, demo_path, cpeptide_path,
                 arms_path=None, arm_subjects_path=None) -> pd.DataFrame:
    """Build the extended feature panel for one study from its explicit files.

    Columns returned:
        Subject_ID, Study, Sex, age_years, disease_duration_years,
        bmi, height_cm, weight_kg, race, ethnicity, cohort_group,
        received_active_treatment, <autoantibodies>, ICA (0 if unmeasured),
        C_Peptide_AUC_4Hrs, log_auc.
    """
    if study not in PANEL_B_STUDIES:
        raise ValueError(f"Study {study!r} has no extended-panel data")

    aa = pd.read_csv(aa_path)
    demo = pd.read_csv(demo_path)

    aa = aa.rename(columns={"accession": "Subject_ID", **_JEFF_AA_MAP})
    for ab in ("GAD65", "IA2IC", "MIAA", "ZNT8", "ICA"):
        if ab not in aa.columns:
            aa[ab] = 0.0

    if "date_of_screening_visit" in aa.columns:
        aa["assay_date"] = _parse_date(aa["date_of_screening_visit"])
    elif "numeric_date_drawn" in aa.columns:
        aa["assay_date"] = _parse_date(aa["numeric_date_drawn"])
    else:
        aa["assay_date"] = pd.NaT

    keep_aa = ["Subject_ID", "GAD65", "IA2IC", "MIAA", "ZNT8", "ICA",
               "baseline_height_cm", "baseline_weight_kg", "baseline_bmi_kg_m_2",
               "assay_date"]
    aa = aa[[c for c in keep_aa if c in aa.columns]]
    aa = aa.rename(columns={
        "baseline_height_cm": "height_cm",
        "baseline_weight_kg": "weight_kg",
        "baseline_bmi_kg_m_2": "bmi",
    })

    demo = demo.rename(columns={"accession": "Subject_ID"})
    demo["Sex"] = demo["sex"].map({"Male": 0, "Female": 1}).astype(float)
    demo["t1d_dx_date"] = _parse_date(demo["date_of_t1dm_diagnosis"])
    demo["day_0_date"] = _parse_date(demo["day_0_date"])
    demo["age_years"] = (demo["day_0_date"] - pd.to_datetime(
        demo["year_of_birth"].astype("Int64").astype(str) + "-07-01",
        errors="coerce")).dt.days / 365.25
    demo["disease_duration_years"] = (demo["day_0_date"] - demo["t1d_dx_date"]).dt.days / 365.25

    keep_demo = ["Subject_ID", "Sex", "age_years", "disease_duration_years",
                 "race", "ethnicity", "cohort_group"]
    demo = demo[[c for c in keep_demo if c in demo.columns]]

    cpep = _read_cpeptide(cpeptide_path)

    df = aa.merge(demo, on="Subject_ID", how="inner").merge(cpep, on="Subject_ID", how="inner")
    df["log_auc"] = np.log(df["C_Peptide_AUC_4Hrs"])
    df["Study"] = study
    df["received_active_treatment"] = _treatment_from_arms(arms_path, arm_subjects_path, df["Subject_ID"]).values

    cols = (["Subject_ID", "Study", "Sex", "age_years", "disease_duration_years",
             "bmi", "height_cm", "weight_kg",
             "race", "ethnicity", "cohort_group", "received_active_treatment",
             "GAD65", "IA2IC", "MIAA", "ZNT8", "ICA",
             "C_Peptide_AUC_4Hrs", "log_auc"])
    return df[[c for c in cols if c in df.columns]]


def panel_b_design_matrix(df: pd.DataFrame):
    """Turn a Panel B frame into (X, y, feature_names) with categoricals one-hot encoded.

    ``cohort_group`` is dropped (study-specific arm codes / age bins, not
    comparable); race and ethnicity are one-hot encoded.
    """
    y = df["log_auc"].astype(float)
    base_cont = ["Sex", "age_years", "disease_duration_years",
                 "bmi", "height_cm", "weight_kg",
                 "GAD65", "IA2IC", "MIAA", "ZNT8", "ICA"]
    if "bmi_missing" in df.columns:
        base_cont.append("bmi_missing")
    if "received_active_treatment" in df.columns:
        base_cont.append("received_active_treatment")
    cont = df[[c for c in base_cont if c in df.columns]].astype(float)

    cat_df = df[[c for c in ("race", "ethnicity") if c in df.columns]].copy()
    for c in cat_df.columns:
        cat_df[c] = cat_df[c].astype(str).fillna("MISSING")
    cats = pd.get_dummies(cat_df, drop_first=True).astype(float)
    X = pd.concat([cont, cats], axis=1)
    return X, y, list(X.columns)


def panel_a_design_matrix(df: pd.DataFrame):
    """Return (X, y, feature_names) for Panel A — features are already numeric."""
    return df[PANEL_A_FEATURES].astype(float), df[PANEL_A_TARGET].astype(float), list(PANEL_A_FEATURES)


def _require(**named):
    """Raise a clear error naming any required input file that wasn't provided."""
    missing = [f"--{k.replace('_', '-')}" for k, v in named.items() if v is None]
    if missing:
        raise ValueError("missing required input file(s): " + ", ".join(missing))


def load_features(study, panel, *, tidy=None, aa=None, demo=None,
                  cpeptide=None, arms=None, arm_subjects=None):
    """Return (frame, feature_names, target) for one study + panel from explicit
    file paths, with the same within-study cleanup the notebooks apply. ``frame``
    has the feature columns plus the log_auc target (and Subject_ID). Panel A -> 9
    legacy features (needs ``tidy`` + ``cpeptide``); Panel B -> the canonical 12
    extended features (needs ``aa`` + ``demo`` + ``cpeptide``; ``arms`` +
    ``arm_subjects`` optional)."""
    panel = panel.upper()
    if panel == "A":
        _require(tidy=tidy, cpeptide=cpeptide)
        a = load_panel_a(study, tidy, cpeptide)
        return a, list(PANEL_A_FEATURES), PANEL_A_TARGET
    if panel == "B":
        _require(aa=aa, demo=demo, cpeptide=cpeptide)
        b = load_panel_b(study, aa, demo, cpeptide, arms, arm_subjects)
        # within-study cleanup (this study's own median; per-row height repair)
        for col in ("bmi", "height_cm", "weight_kg"):
            b[col] = b[col].fillna(b[col].median())
        bad_h = b["height_cm"] <= 0
        b.loc[bad_h, "height_cm"] = np.sqrt(b.loc[bad_h, "weight_kg"] / b.loc[bad_h, "bmi"]) * 100
        X, y, _ = panel_b_design_matrix(b)
        X = X.reindex(columns=PANEL_B_FEATURES)          # canonical 12, drop race/ethnicity
        out = X.copy()
        out["Subject_ID"] = b["Subject_ID"].values
        out[PANEL_A_TARGET] = y.values
        return out, list(PANEL_B_FEATURES), PANEL_A_TARGET
    raise ValueError(f"panel must be 'A' or 'B', got {panel!r}")
