"""
Track A: load cell parameter files, run a CV audit to decide which
parameters are worth including in conductance-space PCA, and prep the
filtered feature matrix.

ASSUMPTION (adjust load_cell_params if wrong):
    Each file in src/models is a .txt file with 40 numeric values,
    one per line, in this order:

    0: gNa, 1: gCaT, 2: gCaS, 3: gA, 4: gKCa, 5: gKd, 6: gH, 7: gL, 8: gImi
    9: EL, 10: ENa, 11: EKd, 12: EH
    13-15: q10_gNa, q10_gNa_m, q10_gNa_h
    16-18: q10_gCaT, q10_gCaT_m, q10_gCaT_h
    19-21: q10_gCaS, q10_gCaS_m, q10_gCaS_h
    22-24: q10_gA, q10_gA_m, q10_gA_h
    25-27: q10_gKCa, q10_gKCa_m, q10_gKCa_h
    28-30: q10_gKdr, q10_gKdr_m, q10_gKdr_h
    31-33: q10_gH, q10_gH_m, q10_gH_h
    34: q10_g_leak
    35-36: q10_g_imi, q10_g_imi_m
    37: q10_tau_Ca
    38: tauIntCa
    39: Iapp   <- dropped immediately, not a cell-intrinsic feature

If your files are instead "name value" pairs per line, comma-separated,
or JSON, only load_cell_params() needs to change — everything downstream
operates on the resulting DataFrame and doesn't care about file format.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

PARAM_NAMES = [
    "gNa", "gCaT", "gCaS", "gA", "gKCa", "gKd", "gH", "gL", "gImi",
    "EL", "ENa", "EKd", "EH",
    "q10_gNa", "q10_gNa_m", "q10_gNa_h",
    "q10_gCaT", "q10_gCaT_m", "q10_gCaT_h",
    "q10_gCaS", "q10_gCaS_m", "q10_gCaS_h",
    "q10_gA", "q10_gA_m", "q10_gA_h",
    "q10_gKCa", "q10_gKCa_m", "q10_gKCa_h",
    "q10_gKdr", "q10_gKdr_m", "q10_gKdr_h",
    "q10_gH", "q10_gH_m", "q10_gH_h",
    "q10_g_leak",
    "q10_g_imi", "q10_g_imi_m",
    "q10_tau_Ca",
    "tauIntCa",
    "Iapp",
]

CANDIDATE_TRACK_A_COLS = PARAM_NAMES[0:13]  # gNa..gImi, EL..EH -- audit will trim further


def load_cell_params(models_dir: str) -> pd.DataFrame:
    """
    Read every .txt file in models_dir, parse 40 whitespace/newline-separated
    numeric values per file, return a cell_id x parameter DataFrame.

    cell_id is taken from the filename stem (e.g. cell_014.txt -> "cell_014").
    """
    models_dir = Path(models_dir)
    files = sorted(models_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {models_dir}")

    rows = {}
    for f in files:
        text = f.read_text()
        # pulls every numeric token (handles one-per-line, whitespace-separated,
        # or comma-separated, and tolerates scientific notation / negative signs)
        values = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)]
        if len(values) != len(PARAM_NAMES):
            print(f"WARNING: {f.name} has {len(values)} values, expected {len(PARAM_NAMES)} — skipping")
            continue
        rows[f.stem] = values

    df = pd.DataFrame.from_dict(rows, orient="index", columns=PARAM_NAMES)
    df.index.name = "cell_id"
    return df


def cv_audit(df: pd.DataFrame, cv_threshold: float = 0.05, outlier_z: float = 3.0) -> pd.DataFrame:
    """
    For each column, compute:
      - cv:          std/|mean| across cells
      - raw_std:     raw standard deviation
      - raw_range:   max - min
      - n_outliers:  cells with |z-score| > outlier_z
      - outlier_ids: which cell_ids those are (empty list if none/many)
      - flag:        'keep' / 'drop_low_cv' / 'inspect_outliers'
    """
    audit = pd.DataFrame({
        "mean": df.mean(),
        "raw_std": df.std(),
        "raw_range": df.max() - df.min(),
    })
    audit["cv"] = audit["raw_std"] / audit["mean"].abs()

    outlier_counts = []
    outlier_id_list = []
    for col in df.columns:
        z = (df[col] - df[col].mean()) / df[col].std()
        outliers = df.index[z.abs() > outlier_z].tolist()
        outlier_counts.append(len(outliers))
        outlier_id_list.append(outliers if 0 < len(outliers) <= 2 else [])

    audit["n_outliers"] = outlier_counts
    audit["outlier_ids"] = outlier_id_list

    def flag(row):
        if 0 < row["n_outliers"] <= 2:
            return "inspect_outliers"
        if row["cv"] < cv_threshold:
            return "drop_low_cv"
        return "keep"

    audit["flag"] = audit.apply(flag, axis=1)
    return audit.sort_values("cv")


def bootstrap_cv_stability(df: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Resample cells with replacement n_boot times, recompute CV each time.
    Returns per-column mean/std of the bootstrap CV distribution --
    a large std relative to mean means that column's CV (and thus its
    keep/drop status) is unstable given the current sample of cells.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    boot_cvs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx]
        boot_cvs.append(sample.std() / sample.mean().abs())
    boot_df = pd.DataFrame(boot_cvs)
    return boot_df.agg(["mean", "std"]).T.rename(columns={"mean": "boot_cv_mean", "std": "boot_cv_std"})


if __name__ == "__main__":
    MODELS_DIR = "src/models"  # adjust if needed

    df_all = load_cell_params(MODELS_DIR)
    print(f"Loaded {len(df_all)} cells, {df_all.shape[1]} parameters each\n")

    # Drop Iapp immediately -- it's stimulus, not a cell property
    df_features = df_all.drop(columns=["Iapp"])

    # Run the audit on everything except Iapp (conductances, reversal
    # potentials, q10s, tauIntCa all included so the audit can confirm
    # which q10s/tauIntCa are truly near-constant, per our prior discussion)
    audit = cv_audit(df_features)
    print("=== CV audit (sorted by CV, ascending) ===")
    print(audit.to_string())

    print("\n=== Outlier cells needing manual inspection ===")
    for col, row in audit[audit["flag"] == "inspect_outliers"].iterrows():
        vals = df_features.loc[row["outlier_ids"], col]
        rest = df_features.loc[~df_features.index.isin(row["outlier_ids"]), col]
        print(f"{col}: outliers={dict(vals)}, rest range=({rest.min():.4g}, {rest.max():.4g})")

    # After eyeballing the outlier print above, list here any "inspect_outliers"
    # columns you've manually confirmed are real biophysical variation (not
    # artifacts/defaults) and want included in Track A despite the flag.
    # e.g. MANUALLY_APPROVED = ["gH", "EKd"]
    MANUALLY_APPROVED = ["gImi", "gCaS"]

    print("\n=== Bootstrap CV stability (flag columns with large boot_cv_std) ===")
    stability = bootstrap_cv_stability(df_features)
    print(stability.to_string())

    # Candidate Track A matrix: keep-flagged columns, plus any
    # inspect_outliers columns you've manually approved above.
    # Columns flagged drop_low_cv are never auto-included -- if you want
    # one of those back in too, add it to MANUALLY_APPROVED as well.
    keep_cols = audit[audit["flag"] == "keep"].index.tolist()
    approved_outlier_cols = [
        c for c in MANUALLY_APPROVED
        if c in audit.index and audit.loc[c, "flag"] == "inspect_outliers"
    ]
    unrecognized = [c for c in MANUALLY_APPROVED if c not in approved_outlier_cols]
    if unrecognized:
        print(f"\nWARNING: MANUALLY_APPROVED entries not found as inspect_outliers "
              f"columns (check spelling / already 'keep' / flagged 'drop_low_cv'): {unrecognized}")

    track_a_cols = [c for c in (keep_cols + approved_outlier_cols) if c in CANDIDATE_TRACK_A_COLS]
    print(f"\nCandidate Track A columns after audit: {track_a_cols}")

    df_track_a = df_features[track_a_cols]
    df_track_a.to_csv("track_a_features.csv")
    print(f"\nSaved filtered Track A feature matrix -> track_a_features.csv "
          f"({df_track_a.shape[0]} cells x {df_track_a.shape[1]} features)")