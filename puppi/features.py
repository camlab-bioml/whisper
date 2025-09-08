# puppi/features.py

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def feature_engineering(intensity_df: pd.DataFrame, controls: list) -> pd.DataFrame:


    # Detect control columns
    control_columns = [col for col in intensity_df.columns if any(ctrl in col for ctrl in controls)]

    # Infer baits (anything that is not a control)
    all_sample_columns = [col for col in intensity_df.columns if col != 'Protein']
    baits = sorted(list(set(col.split('_')[0] for col in all_sample_columns if col not in control_columns)))

    # All intensity columns (for CV calculation)
    intensity_columns = control_columns + [col for col in intensity_df.columns if any(bait in col for bait in baits)]

    # === Precompute global CVs (excluding 'birA' and bait proteins) ===
    global_cv_dict = {}
    for idx, row in intensity_df.iterrows():
        prey = row['Protein']
        if prey in baits or prey == 'birA':
            continue
        all_vals = row[intensity_columns].astype(float).values
        mean_all = np.mean(all_vals)
        sd_all = np.std(all_vals)
        cv = sd_all / mean_all if mean_all > 0 else 0
        global_cv_dict[prey] = cv

    all_bait_features = []

    for bait in baits:
        bait_columns = [col for col in intensity_df.columns if re.fullmatch(fr'{bait}_\d+', col)]
        filtered_df = intensity_df[~intensity_df['Protein'].isin([bait, 'birA'])].copy()

        # Precompute control stats
        control_matrix = filtered_df[control_columns].values
        control_means = np.mean(control_matrix, axis=1)
        control_sds = np.std(control_matrix, axis=1)

        nonzero_mean_controls = control_means[control_means > 0]
        nonzero_sd_controls = control_sds[control_sds > 0]

        min_mean_control = nonzero_mean_controls.min() if len(nonzero_mean_controls) > 0 else 1000
        min_sd_control = nonzero_sd_controls.min() if len(nonzero_sd_controls) > 0 else 1000

        features = []

        for index, row in filtered_df.iterrows():
            prey = row['Protein']
            bait_intensities = row[bait_columns].values
            control_intensities = row[control_columns].values

            mean_baits = np.mean(bait_intensities)
            median_baits = np.median(bait_intensities)
            sd_baits = np.std(bait_intensities)

            mean_controls = np.mean(control_intensities)
            sd_controls = np.std(control_intensities)

            mean_controls = mean_controls if mean_controls > 0 else min_mean_control
            sd_controls = sd_controls if sd_controls > 0 else min_sd_control

            replicate_fold_change_sd = np.std(bait_intensities / mean_controls)
            bait_cv = sd_baits / mean_baits if mean_baits != 0 else 0
            replicate_stability = 1 / (1 + bait_cv)
            bait_control_sd_ratio = sd_baits / sd_controls
            zero_count_baits = np.sum(bait_intensities == 0)
            fold_change = mean_baits / mean_controls
            log_fold_change = np.log2(fold_change + 1e-5)

            penalized_log_fold_change = log_fold_change / max(1, zero_count_baits)
            snr = mean_baits / sd_controls
            penalized_snr = snr / max(1, zero_count_baits)
            mean_diff = mean_baits - mean_controls
            median_diff = median_baits - np.median(control_intensities)
            zero_or_neg_fc = 0 if penalized_log_fold_change <= 0 else 1

            nonzero_reps = int(np.sum(bait_intensities > 0))
            reps_above_ctrl_med = int(np.sum(bait_intensities > np.median(control_intensities)))
            single_rep_flag = 1 if nonzero_reps == 1 else 0

            features.append({
                'Bait': bait,
                'Prey': prey,
                'log_fold_change': penalized_log_fold_change,
                'snr': penalized_snr,
                'mean_diff': mean_diff,
                'median_diff': median_diff,
                'replicate_fold_change_sd': replicate_fold_change_sd,
                'bait_cv': bait_cv,
                'bait_control_sd_ratio': bait_control_sd_ratio,
                'zero_or_neg_fc': zero_or_neg_fc,
                'nonzero_reps': nonzero_reps,
                'reps_above_ctrl_med': reps_above_ctrl_med,
                'single_rep_flag': single_rep_flag
            })

        bait_features_df = pd.DataFrame(features)

        scale_cols = [
            'log_fold_change', 'snr', 'mean_diff', 'median_diff',
            'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio',
            'zero_or_neg_fc'
        ]
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(
            scaler.fit_transform(bait_features_df[scale_cols]),
            columns=scale_cols, index=bait_features_df.index
        )

        bait_features_df['composite_score'] = scaled_df[['log_fold_change','snr','mean_diff','median_diff']].mean(axis=1)
        bait_features_df['global_cv'] = bait_features_df['Prey'].map(global_cv_dict)
        bait_features_df_sorted = bait_features_df.sort_values(by='composite_score', ascending=False)
        all_bait_features.append(bait_features_df_sorted)

    aggregated_features_df = pd.concat(all_bait_features, ignore_index=True)
    aggregated_features_df.to_csv('features.csv', index=False)
    return aggregated_features_df
