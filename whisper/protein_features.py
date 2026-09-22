# whisper/protein_features.py

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Labelling enzymes are part of the construct, not a finding. APEX2 is left out: it is a real human gene outside APEX experiments.
LABELLING_ENZYMES = {
    'bira', 'bioid2', 'turboid', 'miniturbo', 'nminiturbo',
}

# Stand-ins for a missing protein name; '0' is what a gene mapping writes when no gene name exists.
MISSING_IDENTIFIERS = {'', '0', 'nan', 'none'}


def feature_engineering_protein(intensity_df: pd.DataFrame, controls: list, labelling_enzyme=None) -> pd.DataFrame:
    """
    Protein-level feature engineering for whisper.

    Parameters
    ----------
    intensity_df : pd.DataFrame
        Wide matrix with columns:
          - 'Protein'
          - sample intensity columns named as '<BAIT>_<rep>' (e.g., LMNA_1, LMNA_2, ...)
          - control intensity columns containing any of the strings in `controls`
    controls : list
        List of substrings that identify control columns (e.g., ["EGFP", "Empty", "NminiTurbo"])
    labelling_enzyme : str, list or None, default=None
        Name(s) of the labelling enzyme, flagged inert alongside the bait's
        own protein. None uses the BioID defaults in LABELLING_ENZYMES; pass
        "APEX2" for an APEX experiment, or an empty list where no enzyme is
        quantified, as in affinity purification.

    Returns
    -------
    pd.DataFrame
        Aggregated feature table with one row per (Bait, Prey) containing:
        ['Bait','Prey','log_fold_change','snr','mean_diff','median_diff',
         'replicate_fold_change_sd','bait_cv','bait_control_sd_ratio','zero_or_neg_fc',
         'nonzero_reps','reps_above_ctrl_med','single_rep_flag','is_inert',
         'heuristic_score','global_cv']
        The table is also written to 'features.csv'.

    Notes
    -----
    The bait's own protein and the labelling enzyme are scored like any other
    prey but flagged `is_inert`: their enrichment is guaranteed by the
    construct rather than by association. Every statistic here -- the control
    means and SDs, the floors taken from them, and the per-bait scaler behind
    the heuristic score -- is computed on the eligible rows alone, so their
    presence moves no other pair's numbers. `train_and_score_protein` keeps
    them out of the bait clustering, the weak labels, the decoy null and the
    FDR denominator.
    """

    intensity_df = intensity_df.copy()

    # Rows with no protein identifier: unreportable, and they collide in the global-CV dictionary.
    _unnamed = intensity_df['Protein'].isna() | (
        intensity_df['Protein'].astype(str).str.strip().str.lower()
        .isin(MISSING_IDENTIFIERS))
    if _unnamed.any():
        print(f"Dropping {int(_unnamed.sum())} row(s) with no protein identifier")
        intensity_df = intensity_df.loc[~_unnamed].reset_index(drop=True)

    # Repeated identifier keys duplicate pairs downstream; flagged rather than silently merged.
    _dups = intensity_df['Protein'].duplicated(keep=False)
    if _dups.any():
        print(f"WARNING: {int(_dups.sum())} rows share "
              f"{intensity_df.loc[_dups, 'Protein'].nunique()} duplicated protein names")

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

    # === Global CV across ALL samples, for every row ===
    global_cv_dict = {}
    for _, row in intensity_df.iterrows():
        prey = row['Protein']
        vals = row[intensity_columns].astype(float).values
        mean_all = np.mean(vals)
        sd_all = np.std(vals)
        global_cv_dict[prey] = sd_all / mean_all if mean_all > 0 else 0.0

    # Enzyme names to treat as inert: the BioID defaults, or whatever labelling_enzyme names.
    if labelling_enzyme is None:
        _enzymes = set(LABELLING_ENZYMES)
    elif isinstance(labelling_enzyme, str):
        _enzymes = {labelling_enzyme.lower()}
    else:
        _enzymes = {e.lower() for e in labelling_enzyme}

    all_bait_features = []

    for bait in baits:
        # replicate columns for this bait (match strictly '<BAIT>_<rep>')
        bait_columns = [col for col in intensity_df.columns if re.fullmatch(fr'{bait}_[0-9]+', col)]

        # Inert rows: the bait's own protein and the enzyme, tested against protein-group members.
        _inert_names = {bait.lower()} | _enzymes
        _is_inert = intensity_df['Protein'].astype(str).map(
            lambda p: any(m.strip().lower() in _inert_names for m in p.split(';')))
        filtered_df = intensity_df[~_is_inert]
        inert_df = intensity_df[_is_inert]

        # precompute control stats on the eligible rows alone
        control_matrix = filtered_df[control_columns].astype(float).values
        control_means = np.mean(control_matrix, axis=1)
        control_sds   = np.std(control_matrix, axis=1)

        # small positive fallbacks for zeros
        nonzero_mean_controls = control_means[control_means > 0]
        nonzero_sd_controls   = control_sds[control_sds > 0]
        min_mean_control = nonzero_mean_controls.min() if len(nonzero_mean_controls) > 0 else 1.0
        min_sd_control   = nonzero_sd_controls.min()   if len(nonzero_sd_controls)   > 0 else 1.0

        features = []

        # Eligible rows first, then inert ones through identical code and control statistics.
        for _source_df, _inert_flag in ((filtered_df, False), (inert_df, True)):
          for idx, row in _source_df.iterrows():
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
                'single_rep_flag': single_rep_flag,
                'is_inert': _inert_flag
            })

        bait_features_df = pd.DataFrame(features)

        # scale per bait for stability
        scale_cols = [
            'log_fold_change', 'snr', 'mean_diff', 'median_diff',
            'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio',
            'zero_or_neg_fc'
        ]
        # Fit on eligible rows only: an inert row would shift every other pair's z-score.
        _eligible = ~bait_features_df['is_inert'].to_numpy()
        scaler = StandardScaler()
        scaler.fit(bait_features_df.loc[_eligible, scale_cols])
        scaled = pd.DataFrame(
            scaler.transform(bait_features_df[scale_cols]),
            columns=scale_cols, index=bait_features_df.index
        )

        # heuristic score = mean of main signal features
        bait_features_df['heuristic_score'] = scaled[['log_fold_change','snr','mean_diff','median_diff']].mean(axis=1)

        # map global CV
        bait_features_df['global_cv'] = bait_features_df['Prey'].map(global_cv_dict)

        # sort and collect
        all_bait_features.append(bait_features_df.sort_values(by='heuristic_score', ascending=False))

    aggregated_features_df = pd.concat(all_bait_features, ignore_index=True)

    # write (kept for backward compatibility)
    aggregated_features_df.to_csv('features.csv', index=False)

    return aggregated_features_df
