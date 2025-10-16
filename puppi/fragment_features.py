# puppi/fragment_features.py

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def feature_engineering_fragment(intensity_df: pd.DataFrame, controls: list) -> pd.DataFrame:
    """
    Compute fragment-level features for PU-learning.
    Mirrors protein/peptide feature engineering but operates on (Protein, Peptide, Fragment).

    Parameters
    ----------
    intensity_df : pd.DataFrame
        Fragment-level intensity matrix with columns:
        ['Protein', 'Peptide', 'Fragment', <bait/replicate and control columns>]
    controls : list
        List of control identifiers (e.g., ["EGFP", "Empty", "NminiTurbo"])

    Returns
    -------
    pd.DataFrame
        Aggregated feature table per bait–(protein, peptide, fragment) with computed metrics.
    """

    # --- Identify columns ---
    id_cols = ["Protein", "Peptide", "Fragment"]
    control_columns = [c for c in intensity_df.columns if any(ctrl in c for ctrl in controls)]
    sample_columns = [c for c in intensity_df.columns if c not in id_cols]
    bait_like_columns = [c for c in sample_columns if c not in control_columns]
    baits = sorted(list(set(col.split("_")[0] for col in bait_like_columns)))
    intensity_columns = control_columns + [c for c in bait_like_columns if any(b in c for b in baits)]

    # --- Global CV across ALL samples for each fragment (exclude BirA and bait proteins for map) ---
    global_cv = {}
    for _, row in intensity_df.iterrows():
        prot = row["Protein"]
        vals = row[intensity_columns].astype(float).values
        mean_all = np.mean(vals)
        sd_all = np.std(vals)
        global_cv[(row["Protein"], row["Peptide"], row["Fragment"])] = sd_all / mean_all if mean_all > 0 else 0

    all_bait_features = []

    for bait in baits:
        bait_columns = [c for c in intensity_df.columns if re.fullmatch(fr"{bait}_\d+", c)]
        if len(bait_columns) == 0:
            # skip baits without replicate columns that match the "<bait>_#" pattern
            continue

        # Exclude the bait's own protein & BirA
        filtered_df = intensity_df[~intensity_df["Protein"].isin([bait, "birA"])].copy()

        # --- Control summary stats ---
        ctrl_vals = filtered_df[control_columns].astype(float).values
        ctrl_means = np.mean(ctrl_vals, axis=1)
        ctrl_sds = np.std(ctrl_vals, axis=1)

        min_mean_ctrl = np.min(ctrl_means[ctrl_means > 0]) if np.any(ctrl_means > 0) else 1.0
        min_sd_ctrl = np.min(ctrl_sds[ctrl_sds > 0]) if np.any(ctrl_sds > 0) else 1.0

        features = []
        for _, row in filtered_df.iterrows():
            prey_prot = row["Protein"]
            prey_pep = row["Peptide"]
            prey_frag = row["Fragment"]

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
                "Protein": prey_prot,
                "Peptide": prey_pep,
                "Fragment": prey_frag,
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
            })

        bait_features = pd.DataFrame(features)

        # --- Scale and composite score (consistent with protein/peptide) ---
        scale_cols = [
            "log_fold_change", "snr", "mean_diff", "median_diff",
            "replicate_fold_change_sd", "bait_cv", "bait_control_sd_ratio",
            "zero_or_neg_fc",
        ]
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(
            scaler.fit_transform(bait_features[scale_cols]),
            columns=scale_cols, index=bait_features.index,
        )

        bait_features["composite_score"] = scaled_df[
            ["log_fold_change", "snr", "mean_diff", "median_diff"]
        ].mean(axis=1)

        bait_features["global_cv"] = bait_features.apply(
            lambda r: global_cv.get((r["Protein"], r["Peptide"], r["Fragment"]), np.nan), axis=1
        )

        all_bait_features.append(bait_features.sort_values("composite_score", ascending=False))

    aggregated_features_df = pd.concat(all_bait_features, ignore_index=True)
    aggregated_features_df.to_csv("features_fragment.csv", index=False)
    return aggregated_features_df
