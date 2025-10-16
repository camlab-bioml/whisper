# puppi/protein_features.py

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def feature_engineering_protein(intensity_df: pd.DataFrame, controls: list) -> pd.DataFrame:
    """
    Protein-level feature engineering for PUPPI.

    Parameters
    ----------
    intensity_df : pd.DataFrame
        Wide matrix with columns:
          - 'Protein'
          - sample intensity columns named as '<BAIT>_<rep>' (e.g., LMNA_1, LMNA_2, ...)
          - control intensity columns containing any of the strings in `controls`
    controls : list
        List of substrings that identify control columns (e.g., ["EGFP", "Empty", "NminiTurbo"])

    Returns
    -------
    pd.DataFrame
        Aggregated feature table with one row per (Bait, Prey) containing:
        ['Bait','Prey','log_fold_change','snr','mean_diff','median_diff',
         'replicate_fold_change_sd','bait_cv','bait_control_sd_ratio','zero_or_neg_fc',
         'nonzero_reps','reps_above_ctrl_med','single_rep_flag','composite_score','global_cv']
        The table is also written to 'features.csv'.
    """

    # --- Identify control columns ---
    control_columns = [col for col in intensity_df.columns if any(ctrl in col for ctrl in controls)]

    # --- Infer baits (anything that is not a control, parsed from '<BAIT>_<rep>') ---
    all_sample_columns = [col for col in intensity_df.columns if col != 'Protein']
    baits = sorted(list(set(
        col.split('_')[0]
        for col in all_sample_columns
        if (col not in control_columns) and ('_' in col)
    )))

    # --- All intensity columns (for global CV) ---
    intensity_columns = control_columns + [c for c in all_sample_columns if c not in control_columns]

    # === Global CV across ALL samples (exclude bait protein itself and 'birA') ===
    global_cv_dict = {}
    for _, row in intensity_df.iterrows():
        prey = row['Protein']
        if prey in baits or prey == 'birA':
            continue
        vals = row[intensity_columns].astype(float).values
        mean_all = np.mean(vals)
        sd_all = np.std(vals)
        cv = sd_all / mean_all if mean_all > 0 else 0.0
        global_cv_dict[prey] = cv

    all_bait_features = []

    for bait in baits:
        # replicate columns for this bait (match strictly '<BAIT>_<rep>')
        bait_columns = [col for col in intensity_df.columns if re.fullmatch(fr'{bait}_[0-9]+', col)]

        # filter out bait protein and 'birA'
        filtered_df = intensity_df[~intensity_df['Protein'].isin([bait, 'birA'])].copy()

        # precompute control stats
        control_matrix = filtered_df[control_columns].astype(float).values
        control_means = np.mean(control_matrix, axis=1)
        control_sds   = np.std(control_matrix, axis=1)

        # small positive fallbacks for zeros
        nonzero_mean_controls = control_means[control_means > 0]
        nonzero_sd_controls   = control_sds[control_sds > 0]
        min_mean_control = nonzero_mean_controls.min() if len(nonzero_mean_controls) > 0 else 1.0
        min_sd_control   = nonzero_sd_controls.min()   if len(nonzero_sd_controls)   > 0 else 1.0

        features = []

        for idx, row in filtered_df.iterrows():
            prey = row['Protein']

            bait_intensities    = row[bait_columns].astype(float).values if len(bait_columns) else np.array([0.0])
            control_intensities = row[control_columns].astype(float).values if len(control_columns) else np.array([0.0])

            mean_baits   = np.mean(bait_intensities)
            median_baits = np.median(bait_intensities)
            sd_baits     = np.std(bait_intensities)

            mean_controls = np.mean(control_intensities)
            sd_controls   = np.std(control_intensities)

            # guard against zeros
            mean_controls = mean_controls if mean_controls > 0 else min_mean_control
            sd_controls   = sd_controls   if sd_controls   > 0 else min_sd_control

            # core ratios / penalties
            zero_count_baits = int(np.sum(bait_intensities == 0))
            fold_change      = mean_baits / mean_controls
            log_fold_change  = np.log2(fold_change + 1e-5)
            penalized_log_fc = log_fold_change / max(1, zero_count_baits)

            snr          = mean_baits / sd_controls
            penalized_snr = snr / max(1, zero_count_baits)

            replicate_fc_sd        = np.std(bait_intensities / mean_controls)
            bait_cv                = (sd_baits / mean_baits) if mean_baits != 0 else 0.0
            bait_control_sd_ratio  = sd_baits / sd_controls
            mean_diff              = mean_baits  - mean_controls
            median_diff            = median_baits - np.median(control_intensities)
            zero_or_neg_fc         = 0 if penalized_log_fc <= 0 else 1

            # replicate support flags
            nonzero_reps        = int(np.sum(bait_intensities > 0))
            reps_above_ctrl_med = int(np.sum(bait_intensities > np.median(control_intensities)))
            single_rep_flag     = 1 if nonzero_reps == 1 else 0

            features.append({
                'Bait': bait,
                'Prey': prey,
                'log_fold_change': penalized_log_fc,
                'snr': penalized_snr,
                'mean_diff': mean_diff,
                'median_diff': median_diff,
                'replicate_fold_change_sd': replicate_fc_sd,
                'bait_cv': bait_cv,
                'bait_control_sd_ratio': bait_control_sd_ratio,
                'zero_or_neg_fc': zero_or_neg_fc,
                'nonzero_reps': nonzero_reps,
                'reps_above_ctrl_med': reps_above_ctrl_med,
                'single_rep_flag': single_rep_flag
            })

        bait_features_df = pd.DataFrame(features)

        # scale per bait for stability
        scale_cols = [
            'log_fold_change', 'snr', 'mean_diff', 'median_diff',
            'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio',
            'zero_or_neg_fc'
        ]
        scaler = StandardScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(bait_features_df[scale_cols]),
            columns=scale_cols, index=bait_features_df.index
        )

        # composite score = mean of main signal features
        bait_features_df['composite_score'] = scaled[['log_fold_change','snr','mean_diff','median_diff']].mean(axis=1)

        # map global CV (NaN if not computed—e.g., prey was bait/birA everywhere)
        bait_features_df['global_cv'] = bait_features_df['Prey'].map(global_cv_dict)

        # sort and collect
        all_bait_features.append(bait_features_df.sort_values(by='composite_score', ascending=False))

    aggregated_features_df = pd.concat(all_bait_features, ignore_index=True)

    # write (kept for backward compatibility)
    aggregated_features_df.to_csv('features.csv', index=False)

    return aggregated_features_df
