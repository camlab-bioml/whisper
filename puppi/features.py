# puppi/features.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def feature_engineering(intensity_df: pd.DataFrame, controls=None) -> pd.DataFrame:
    """
    Generalized feature engineering for BioID, AP-MS datasets.

    Parameters:
        intensity_df (pd.DataFrame): Input dataframe with 'Protein' column and replicate intensity columns.
        controls (list of str): List of substrings to identify control columns (e.g., ["EGFP", "Empty"]).

    Returns:
        pd.DataFrame: Aggregated engineered features for all bait–prey pairs.
    """

    all_sample_columns = [col for col in intensity_df.columns if col != "Protein"]
    control_columns = [col for col in all_sample_columns if any(ctrl in col for ctrl in controls)]
    bait_names = sorted(set(col.split("_")[0] for col in all_sample_columns if col not in control_columns))

    global_cv_dict = {}
    for _, row in intensity_df.iterrows():
        prey = row["Protein"]
        all_vals = row[all_sample_columns].astype(float).values
        mean_all = np.mean(all_vals)
        sd_all = np.std(all_vals)
        cv = sd_all / mean_all if mean_all > 0 else 0
        global_cv_dict[prey] = cv

    all_bait_features = []

    for bait in bait_names:
        bait_columns = [col for col in all_sample_columns if col.startswith(f"{bait}_")]
        if not bait_columns:
            continue

        filtered_df = intensity_df[~intensity_df["Protein"].isin([bait, "birA"])]
        control_matrix = filtered_df[control_columns].values
        control_means = np.mean(control_matrix, axis=1)
        control_sds = np.std(control_matrix, axis=1)

        min_mean_control = np.min(control_means[control_means > 0]) if np.any(control_means > 0) else 1000
        min_sd_control = np.min(control_sds[control_sds > 0]) if np.any(control_sds > 0) else 1000

        features = []

        for idx, row in filtered_df.iterrows():
            prey = row["Protein"]
            bait_intensities = row[bait_columns].astype(float).values
            control_intensities = row[control_columns].astype(float).values

            mean_baits = np.mean(bait_intensities)
            median_baits = np.median(bait_intensities)
            sd_baits = np.std(bait_intensities)

            mean_controls = np.mean(control_intensities)
            sd_controls = np.std(control_intensities)

            mean_controls = mean_controls if mean_controls > 0 else min_mean_control
            sd_controls = sd_controls if sd_controls > 0 else min_sd_control

            replicate_fc_sd = np.std(bait_intensities / mean_controls)
            bait_cv = sd_baits / mean_baits if mean_baits != 0 else 0
            bait_control_sd_ratio = sd_baits / sd_controls
            zero_count_baits = np.sum(bait_intensities == 0)
            fold_change = mean_baits / mean_controls
            log_fc = np.log2(fold_change + 1e-5)

            penalized_log_fc = log_fc / max(1, zero_count_baits)
            snr = mean_baits / sd_controls
            penalized_snr = snr / max(1, zero_count_baits)
            mean_diff = mean_baits - mean_controls
            median_diff = median_baits - np.median(control_intensities)
            zero_or_neg_fc = 0 if penalized_log_fc <= 0 else 1

            features.append({
                "Bait": bait,
                "Prey": prey,
                "log_fold_change": penalized_log_fc,
                "snr": penalized_snr,
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "replicate_fold_change_sd": replicate_fc_sd,
                "bait_cv": bait_cv,
                "bait_control_sd_ratio": bait_control_sd_ratio,
                "zero_or_neg_fc": zero_or_neg_fc
            })

        bait_df = pd.DataFrame(features)

        scaler = StandardScaler()
        scaled_vals = scaler.fit_transform(bait_df.iloc[:, 2:])
        scaled_df = pd.DataFrame(scaled_vals, columns=bait_df.columns[2:], index=bait_df.index)

        composite_score = scaled_df[["log_fold_change", "snr", "mean_diff", "median_diff"]].mean(axis=1)
        bait_df["composite_score"] = composite_score
        bait_df["global_cv"] = bait_df["Prey"].map(global_cv_dict)

        bait_df_sorted = bait_df.sort_values(by="composite_score", ascending=False)
        all_bait_features.append(bait_df_sorted)

    aggregated_features_df = pd.concat(all_bait_features, ignore_index=True)
    return aggregated_features_df
