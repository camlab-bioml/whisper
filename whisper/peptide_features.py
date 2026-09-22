# whisper/peptide_features.py

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Labelling enzymes are part of the construct, not a finding. APEX2 is left out: it is a real human gene outside APEX experiments.
LABELLING_ENZYMES = {
    "bira", "bioid2", "turboid", "miniturbo", "nminiturbo",
}

# Stand-ins for a missing protein name; '0' is what a gene mapping writes when no gene name exists.
MISSING_IDENTIFIERS = {"", "0", "nan", "none"}


def feature_engineering_peptide(intensity_df: pd.DataFrame, controls: list, labelling_enzyme=None) -> pd.DataFrame:
    """
    Compute peptide-level features.
    Mirrors the logic of protein_features.feature_engineering_protein,
    but operates per peptide rather than per protein.

    Parameters
    ----------
    intensity_df : pd.DataFrame
        Peptide-level intensity matrix (columns = bait replicates and controls,
        rows = peptide entries with columns ['Protein', 'Peptide', sample columns])
    controls : list
        List of control identifiers (e.g., ["EGFP", "Empty", "NminiTurbo"])
    labelling_enzyme : str, list or None, default=None
        Name(s) of the labelling enzyme, flagged inert alongside the bait's
        own protein. None uses the BioID defaults in LABELLING_ENZYMES; pass
        "APEX2" for an APEX experiment, or an empty list where no enzyme is
        quantified, as in affinity purification.

    Returns
    -------
    pd.DataFrame
        Aggregated feature table per bait–peptide pair with computed metrics,
        including an `is_inert` column. Also written to 'features_peptide.csv'.

    Notes
    -----
    Peptides of the bait's own protein and of the labelling enzyme are scored
    like any other peptide but flagged `is_inert`: their enrichment is
    guaranteed by the construct rather than by association. The control means
    and SDs, the floors taken from them, and the per-bait scaler behind the
    heuristic score are computed on the eligible rows alone, so their presence
    moves no other pair's numbers.
    """

    intensity_df = intensity_df.copy()

    # Rows with no protein identifier: unreportable, and they collide in the global-CV dictionary.
    _unnamed = intensity_df["Protein"].isna() | (
        intensity_df["Protein"].astype(str).str.strip().str.lower()
        .isin(MISSING_IDENTIFIERS))
    if _unnamed.any():
        print(f"Dropping {int(_unnamed.sum())} row(s) with no protein identifier")
        intensity_df = intensity_df.loc[~_unnamed].reset_index(drop=True)

    # Repeated identifier keys duplicate pairs downstream; flagged rather than silently merged.
    _dups = intensity_df.duplicated(subset=["Protein", "Peptide"], keep=False)
    if _dups.any():
        n_keys = intensity_df.loc[_dups, ["Protein", "Peptide"]].drop_duplicates().shape[0]
        print(f"WARNING: {int(_dups.sum())} rows share {n_keys} duplicated "
              f"(Protein, Peptide) keys")

    # --- Identify columns ---
    control_columns = [c for c in intensity_df.columns if any(ctrl in c for ctrl in controls)]
    all_sample_columns = [c for c in intensity_df.columns if c not in ["Protein", "Peptide"]]
    baits = sorted(list(set(col.split("_")[0] for col in all_sample_columns if col not in control_columns)))
    intensity_columns = control_columns + [c for c in all_sample_columns if c not in control_columns]

    # --- Compute global CVs for every row ---
    global_cv = {}
    for _, row in intensity_df.iterrows():
        vals = row[intensity_columns].astype(float).values
        mean_all = np.mean(vals)
        sd_all = np.std(vals)
        global_cv[(row["Protein"], row["Peptide"])] = sd_all / mean_all if mean_all > 0 else 0

    # Enzyme names to treat as inert: the BioID defaults, or whatever labelling_enzyme names.
    if labelling_enzyme is None:
        _enzymes = set(LABELLING_ENZYMES)
    elif isinstance(labelling_enzyme, str):
        _enzymes = {labelling_enzyme.lower()}
    else:
        _enzymes = {e.lower() for e in labelling_enzyme}

    all_bait_features = []

    for bait in baits:
        bait_columns = [c for c in intensity_df.columns if re.fullmatch(fr"{bait}_\d+", c)]

        # Inert rows: the bait's own protein and the enzyme, tested against protein-group members.
        _inert_names = {bait.lower()} | _enzymes
        _is_inert = intensity_df["Protein"].astype(str).map(
            lambda p: any(m.strip().lower() in _inert_names for m in p.split(";")))
        filtered_df = intensity_df[~_is_inert]
        inert_df = intensity_df[_is_inert]

        # --- Control summary stats, on the eligible rows alone ---
        ctrl_vals = filtered_df[control_columns].astype(float).values
        ctrl_means = np.mean(ctrl_vals, axis=1)
        ctrl_sds = np.std(ctrl_vals, axis=1)

        min_mean_ctrl = np.min(ctrl_means[ctrl_means > 0]) if np.any(ctrl_means > 0) else 1.0
        min_sd_ctrl = np.min(ctrl_sds[ctrl_sds > 0]) if np.any(ctrl_sds > 0) else 1.0

        features = []
        # Eligible rows first, then inert ones through identical code and control statistics.
        for _source_df, _inert_flag in ((filtered_df, False), (inert_df, True)):
          for _, row in _source_df.iterrows():
            prey = row["Protein"]
            peptide = row["Peptide"]

            bait_int = row[bait_columns].astype(float).values
            ctrl_int = row[control_columns].astype(float).values

            mean_bait = np.mean(bait_int)
            median_bait = np.median(bait_int)
            sd_bait = np.std(bait_int)

            mean_ctrl = np.mean(ctrl_int)
            sd_ctrl = np.std(ctrl_int)
            mean_ctrl = mean_ctrl if mean_ctrl > 0 else min_mean_ctrl
            sd_ctrl = sd_ctrl if sd_ctrl > 0 else min_sd_ctrl

            zero_count = np.sum(bait_int == 0)
            fold_change = mean_bait / mean_ctrl
            log_fc = np.log2(fold_change + 1e-5)
            penalized_log_fc = log_fc / max(1, zero_count)
            snr = mean_bait / sd_ctrl
            penalized_snr = snr / max(1, zero_count)

            replicate_fc_sd = np.std(bait_int / mean_ctrl)
            bait_cv = sd_bait / mean_bait if mean_bait != 0 else 0
            bait_ctrl_sd_ratio = sd_bait / sd_ctrl

            nonzero_reps = int(np.sum(bait_int > 0))
            reps_above_ctrl_med = int(np.sum(bait_int > np.median(ctrl_int)))
            single_rep_flag = 1 if nonzero_reps == 1 else 0

            features.append({
                "Bait": bait,
                "Protein": prey,
                "Peptide": peptide,
                "log_fold_change": penalized_log_fc,
                "snr": penalized_snr,
                "mean_diff": mean_bait - mean_ctrl,
                "median_diff": median_bait - np.median(ctrl_int),
                "replicate_fold_change_sd": replicate_fc_sd,
                "bait_cv": bait_cv,
                "bait_control_sd_ratio": bait_ctrl_sd_ratio,
                "zero_or_neg_fc": 0 if penalized_log_fc <= 0 else 1,
                "nonzero_reps": nonzero_reps,
                "reps_above_ctrl_med": reps_above_ctrl_med,
                "single_rep_flag": single_rep_flag,
                "is_inert": _inert_flag,
            })

        bait_features = pd.DataFrame(features)

        # --- Scaling and heuristic score ---
        scale_cols = [
            "log_fold_change", "snr", "mean_diff", "median_diff",
            "replicate_fold_change_sd", "bait_cv", "bait_control_sd_ratio",
            "zero_or_neg_fc",
        ]
        # Fit on eligible rows only: an inert row would shift every other pair's z-score.
        _eligible = ~bait_features["is_inert"].to_numpy()
        scaler = StandardScaler()
        scaler.fit(bait_features.loc[_eligible, scale_cols])
        scaled_df = pd.DataFrame(
            scaler.transform(bait_features[scale_cols]),
            columns=scale_cols, index=bait_features.index,
        )

        bait_features["heuristic_score"] = scaled_df[
            ["log_fold_change", "snr", "mean_diff", "median_diff"]
        ].mean(axis=1)

        bait_features["global_cv"] = bait_features.apply(
            lambda r: global_cv.get((r["Protein"], r["Peptide"]), np.nan), axis=1
        )

        all_bait_features.append(bait_features.sort_values("heuristic_score", ascending=False))

    aggregated_features_df = pd.concat(all_bait_features, ignore_index=True)
    aggregated_features_df.to_csv("features_peptide.csv", index=False)
    return aggregated_features_df
