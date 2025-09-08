# puppi/features.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def feature_engineering(intensity_df: pd.DataFrame, controls=None) -> pd.DataFrame:
    """
    Generalized feature engineering for BioID / AP-MS datasets.

    Parameters
    ----------
    intensity_df : pd.DataFrame
        Input dataframe with a 'Protein' column and replicate intensity columns named as '<BAIT>_<rep>'.
    controls : list of str
        List of substrings that identify control columns (e.g., ["EGFP", "Empty", "NminiTurbo"]).

    Returns
    -------
    pd.DataFrame
        Aggregated engineered features for all bait–prey pairs with the following columns (per bait):
        - Bait, Prey
        - log_fold_change (penalized by zero_count_baits)
        - snr (penalized by zero_count_baits)
        - mean_diff, median_diff
        - replicate_fold_change_sd
        - bait_cv, bait_control_sd_ratio
        - zero_or_neg_fc (flag: 0/1)
        - nonzero_reps (count)
        - reps_above_ctrl_med (count)
        - single_rep_flag (flag: 0/1)
        - replicate_stability (1 / (1 + bait_cv))
        - composite_score (mean of scaled log_fc, snr, mean_diff, median_diff)
        - global_cv (CV across ALL samples and controls for that prey)
    """
    if controls is None:
        controls = []

    # Identify columns
    all_sample_columns = [col for col in intensity_df.columns if col != "Protein"]
    control_columns = [col for col in all_sample_columns if any(ctrl in col for ctrl in controls)]
    bait_names = sorted(
        set(col.split("_")[0] for col in all_sample_columns if col not in control_columns)
    )

    # === Precompute global CVs for all proteins ===
    global_cv_dict = {}
    for _, row in intensity_df.iterrows():
        prey = row["Protein"]
        all_vals = row[all_sample_columns].astype(float).values
        mean_all = np.mean(all_vals)
        sd_all = np.std(all_vals)
        cv = sd_all / mean_all if mean_all > 0 else 0.0
        global_cv_dict[prey] = cv

    all_bait_features = []

    for bait in bait_names:
        bait_columns = [col for col in all_sample_columns if col.startswith(f"{bait}_")]
        if not bait_columns:
            continue

        # Exclude bait self and common enzyme tag if present
        filtered_df = intensity_df[~intensity_df["Protein"].isin([bait, "birA"])].copy()

        # Precompute control stats
        if control_columns:
            control_matrix = filtered_df[control_columns].astype(float).values
            control_means = np.mean(control_matrix, axis=1)
            control_sds = np.std(control_matrix, axis=1)
        else:
            # If no controls provided, fall back to small positive placeholders to avoid divide-by-zero
            control_means = np.full(len(filtered_df), 1e-6, dtype=float)
            control_sds = np.full(len(filtered_df), 1.0, dtype=float)

        # Robust small positive substitutes for zero controls
        nonzero_mean_controls = control_means[control_means > 0]
        nonzero_sd_controls = control_sds[control_sds > 0]
        min_mean_control = nonzero_mean_controls.min() if len(nonzero_mean_controls) > 0 else 1000.0
        min_sd_control = nonzero_sd_controls.min() if len(nonzero_sd_controls) > 0 else 1000.0

        features = []

        for idx, row in filtered_df.iterrows():
            prey = row["Protein"]
            bait_intensities = row[bait_columns].astype(float).values

            if control_columns:
                control_intensities = row[control_columns].astype(float).values
            else:
                # Synthetic controls if none provided, to keep formulas defined
                control_intensities = np.array([min_mean_control], dtype=float)

            mean_baits = np.mean(bait_intensities)
            median_baits = np.median(bait_intensities)
            sd_baits = np.std(bait_intensities)

            mean_controls = np.mean(control_intensities)
            sd_controls = np.std(control_intensities)
            ctrl_median = np.median(control_intensities)

            # Guard against zeros by substituting the smallest positive observed value
            mean_controls = mean_controls if mean_controls > 0 else min_mean_control
            sd_controls = sd_controls if sd_controls > 0 else min_sd_control

            # Core ratios and spreads
            replicate_fc_sd = np.std(bait_intensities / mean_controls)
            bait_cv = sd_baits / mean_baits if mean_baits != 0 else 0.0
            replicate_stability = 1.0 / (1.0 + bait_cv)
            bait_control_sd_ratio = sd_baits / sd_controls if sd_controls != 0 else np.inf
            zero_count_baits = int(np.sum(bait_intensities == 0))

            # Fold-changes and contrasts
            fold_change = mean_baits / mean_controls
            log_fc = np.log2(fold_change + 1e-5)
            penalized_log_fc = log_fc / max(1, zero_count_baits)

            snr = mean_baits / sd_controls if sd_controls != 0 else np.inf
            penalized_snr = snr / max(1, zero_count_baits)

            mean_diff = mean_baits - mean_controls
            median_diff = median_baits - ctrl_median
            zero_or_neg_fc = 0 if penalized_log_fc <= 0 else 1

            # Per-bait replicate support flags/counts
            nonzero_reps = int(np.sum(bait_intensities > 0))
            reps_above_ctrl_med = int(np.sum(bait_intensities > ctrl_median))
            single_rep_flag = 1 if nonzero_reps == 1 else 0  # 1 = only one replicate has signal

            features.append(
                {
                    "Bait": bait,
                    "Prey": prey,
                    "log_fold_change": penalized_log_fc,
                    "snr": penalized_snr,
                    "mean_diff": mean_diff,
                    "median_diff": median_diff,
                    "replicate_fold_change_sd": replicate_fc_sd,
                    "bait_cv": bait_cv,
                    "bait_control_sd_ratio": bait_control_sd_ratio,
                    "zero_or_neg_fc": zero_or_neg_fc,
                    "nonzero_reps": nonzero_reps,
                    "reps_above_ctrl_med": reps_above_ctrl_med,
                    "single_rep_flag": single_rep_flag,
                    "replicate_stability": replicate_stability,
                }
            )

        bait_df = pd.DataFrame(features)

        # Explicitly choose columns to scale (exclude count/flag-like fields)
        scale_cols = [
            "log_fold_change",
            "snr",
            "mean_diff",
            "median_diff",
            "replicate_fold_change_sd",
            "bait_cv",
            "bait_control_sd_ratio",
            "zero_or_neg_fc",  # keep for consistency with your script
            "replicate_stability",
        ]

        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(bait_df[scale_cols])
        scaled_df = pd.DataFrame(scaled_vals, columns=scale_cols, index=bait_df.index)

        # Composite score from the four key (scaled) signals
        bait_df["composite_score"] = scaled_df[
            ["log_fold_change", "snr", "mean_diff", "median_diff"]
        ].mean(axis=1)

        # Add global CV
        bait_df["global_cv"] = bait_df["Prey"].map(global_cv_dict)

        # Sort and collect
        bait_df_sorted = bait_df.sort_values(by="composite_score", ascending=False)
        all_bait_features.append(bait_df_sorted)

    aggregated_features_df = pd.concat(all_bait_features, ignore_index=True)
    return aggregated_features_df
