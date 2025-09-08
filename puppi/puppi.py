import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

def feature_engineering(intensity_file, controls, output_csv_path, baits=None):
    """
    Feature engineering for BioID/APMS data analysis.
    
    Parameters:
    -----------
    intensity_file : str
        Path to the TSV file containing intensity data
    controls : list of str
        List of control prefixes (e.g., ['EGFP', 'Empty', 'NminiTurbo'])
    output_csv_path : str
        Path where the engineered features CSV will be saved
    baits : list of str, optional
        List of bait names. If None, will be inferred from column names
    
    Returns:
    --------
    pd.DataFrame
        Aggregated features dataframe (also saved to CSV)
    """
    
    # Load intensity data
    intensity_df = pd.read_csv(intensity_file, sep='\t')
    
    # Define control columns
    control_columns = [col for col in intensity_df.columns if any(ctrl in col for ctrl in controls)]
    
    # Infer baits if not provided
    if baits is None:
        all_sample_columns = [col for col in intensity_df.columns if col != 'Protein']
        baits = sorted(list(set([col.split('_')[0] for col in all_sample_columns if col not in control_columns])))
    
    # All intensity columns (bait + control)
    intensity_columns = [col for col in intensity_df.columns if col != 'Protein']
    
    # === Precompute global CVs for all proteins ===
    global_cv_dict = {}
    for idx, row in intensity_df.iterrows():
        prey = row['Protein']
        all_vals = row[intensity_columns].astype(float).values
        mean_all = np.mean(all_vals)
        sd_all = np.std(all_vals)
        cv = sd_all / mean_all if mean_all > 0 else 0
        global_cv_dict[prey] = cv
    
    # === Feature engineering for each bait ===
    all_bait_features = []
    
    for bait in baits:
        bait_columns = [col for col in intensity_df.columns if re.fullmatch(fr'{bait}_\d+', col)]
        filtered_df = intensity_df[~intensity_df['Protein'].isin([bait, 'birA'])]
    
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
    
            # Per-bait replicate support flags
            nonzero_reps = int(np.sum(bait_intensities > 0))
            reps_above_ctrl_med = int(np.sum(bait_intensities > np.median(control_intensities)))
            single_rep_flag = 1 if nonzero_reps == 1 else 0  # 1 = only one replicate has signal
    
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
    
        # Explicitly choose columns to scale (exclude flags)
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
    
        # Composite from the four key (scaled) signals
        bait_features_df['composite_score'] = scaled_df[['log_fold_change','snr','mean_diff','median_diff']].mean(axis=1)
    
        # Add global CV
        bait_features_df['global_cv'] = bait_features_df['Prey'].map(global_cv_dict)
    
        # Sort and store
        bait_features_df_sorted = bait_features_df.sort_values(by='composite_score', ascending=False)
        all_bait_features.append(bait_features_df_sorted)
    
    # === Aggregate and save ===
    aggregated_features_df = pd.concat(all_bait_features, ignore_index=True)
    aggregated_features_df.to_csv(output_csv_path, index=False)
    
    print(f"Feature engineering completed. Features saved to {output_csv_path}")
    print(f"Shape: {aggregated_features_df.shape}")
    print(f"Baits processed: {baits}")
    print(f"Controls used: {controls}")
    
    return aggregated_features_df


def train_and_score(features_csv_path, output_csv_path, initial_positives, initial_negatives):
    """
    Train a model and score interactions for BioID DIA data with FDR estimation.
    
    Parameters:
    -----------
    features_csv_path : str
        Path to CSV file containing the engineered features
    output_csv_path : str
        Path where the final results CSV will be saved
    initial_positives : int
        Number of initial positive examples to use for training
    initial_negatives : int
        Number of initial negative examples to use for training
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with predicted probabilities and FDR values (also saved to CSV)
    """
    

    

    
    # Read the CSV file exactly as in original script
    df_real = pd.read_csv(features_csv_path)
    
    # Set random seed exactly as in original
    np.random.seed(42)

    # Feature columns
    feature_columns = [
        'log_fold_change', 'snr', 'mean_diff', 'median_diff', 
        'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio', 
        'zero_or_neg_fc',
    ]

    X_real = df_real[feature_columns].values

    # --- Hierarchical Clustering of Baits using std ---
    bait_top50_stds = {
        bait: df_real[df_real['Bait'] == bait]['composite_score'].nlargest(50).std()
        for bait in df_real['Bait'].unique()
    }

    bait_names  = np.array(list(bait_top50_stds.keys()))
    bait_scores = np.array(list(bait_top50_stds.values())).reshape(-1, 1)

    # Cluster only if we have > 2 baits
    if len(bait_names) > 2:
        linkage_matrix = linkage(bait_scores, method='ward')
        clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
    else:
        clusters = np.ones(len(bait_names), dtype=int)  # all strong

    # Map bait -> cluster label
    bait_cluster_map = {bait: int(cluster) for bait, cluster in zip(bait_names, clusters)}

    # Decide which cluster is "strong":
    # 1) more members; 2) if tie, higher mean bait score
    unique_clusters = np.unique(clusters)
    cluster_sizes = {c: int(np.sum(clusters == c)) for c in unique_clusters}
    cluster_means  = {c: float(bait_scores[clusters == c].mean()) for c in unique_clusters}

    # find clusters with max size
    max_size = max(cluster_sizes.values())
    cands = [c for c, n in cluster_sizes.items() if n == max_size]
    if len(cands) == 1:
        strong_cluster_id = cands[0]
    else:
        # tie-break by higher mean score
        strong_cluster_id = max(cands, key=lambda c: cluster_means[c])

    print("\nCluster assignments:")
    for bait, cluster in bait_cluster_map.items():
        print(f"Bait: {bait}, Cluster: {cluster}")
    print(f"\nStrong cluster chosen: {strong_cluster_id} "
          f"(size={cluster_sizes[strong_cluster_id]}, mean={cluster_means[strong_cluster_id]:.4f})")

    # --- Distribute total positives based on self-thresholded top-50 means ---
    baits = list(df_real['Bait'].unique())
    strong_baits = [b for b in bait_names if bait_cluster_map[b] == strong_cluster_id]

    all_counts = []
    for bait in strong_baits:
        bait_df = df_real[df_real['Bait'] == bait]
        top50_mean = bait_df['composite_score'].nlargest(50).mean()
        count_above = (bait_df['composite_score'] > top50_mean).sum()
        all_counts.append(count_above)

    # Use the provided initial_positives parameter
    average_top_count = initial_positives

    bait_scaled_positives = {bait: (average_top_count if bait in strong_baits else 0)
                             for bait in baits}

    print("\nAssigned positives:")
    for bait, npos in bait_scaled_positives.items():
        print(f"Bait: {bait}, Positives: {npos}")

    # Label positives & negatives
    y_train_labels = pd.Series(0, index=df_real.index)
    N_negatives = initial_negatives

    for bait in df_real['Bait'].unique():
        bait_df = df_real[df_real['Bait'] == bait].copy()
        N_positives = bait_scaled_positives[bait]

        if N_positives > 0:
            # Sort by composite score (best first)
            ranked = bait_df.sort_values('composite_score', ascending=False)

            # Exclude single-replicate spikes from positives
            elig_pos = ranked[ranked['single_rep_flag'] != 1]

            # Take exactly N_positives (or as many as available)
            top_positives = elig_pos.index[:N_positives]
            y_train_labels.loc[top_positives] = 1

            # Pick negatives from the bottom, excluding chosen positives
            remaining = bait_df.drop(index=top_positives, errors='ignore')
            bottom_negatives = remaining['composite_score'].nsmallest(N_negatives).index
            y_train_labels.loc[bottom_negatives] = -1

    labeled_indices = y_train_labels[y_train_labels != 0].index
    X_train, y_train = X_real[labeled_indices], y_train_labels.loc[labeled_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    bagging_rf = BaggingClassifier(estimator=rf_classifier, n_estimators=100, random_state=42)
    bagging_rf.fit(X_train_scaled, y_train)

    X_scaled = scaler.transform(X_real)
    y_pred_proba_calibrated = bagging_rf.predict_proba(X_scaled)[:, 1]
    df_real['predicted_probability'] = y_pred_proba_calibrated

    # -----------------------------
    # Decoy Generation for Global FDR
    # -----------------------------
    all_decoy_probs = []
    bait_decoy_probs_map = {}

    for bait in df_real['Bait'].unique():
        bait_df = df_real[df_real['Bait'] == bait]
        df_decoy = bait_df.copy()
        for col in feature_columns:
            df_decoy[col] = np.random.permutation(df_decoy[col].values)
        X_decoy = df_decoy[feature_columns].values
        X_decoy_scaled = scaler.transform(X_decoy)
        decoy_probs = bagging_rf.predict_proba(X_decoy_scaled)[:, 1]
        all_decoy_probs.extend(decoy_probs)
        bait_decoy_probs_map[bait] = decoy_probs

    # Calculate FDR for each unique probability
    unique_real_probs = np.unique(df_real['predicted_probability'])
    all_decoy_probs_array = np.array(all_decoy_probs)

    # First pass: calculate raw FDR values
    raw_fdr_dict = {}
    for prob in unique_real_probs:
        real_count = (df_real['predicted_probability'] >= prob).sum()
        decoy_count = np.sum(all_decoy_probs_array >= prob)
        raw_fdr = min(decoy_count / real_count if real_count > 0 else 1.0, 1.0)
        raw_fdr_dict[prob] = raw_fdr

    # Second pass: enforce monotonicity
    # Sort probabilities in ascending order
    sorted_probs = np.sort(unique_real_probs)
    fdr_global_dict = {}

    # Start from the lowest probability (highest FDR)
    fdr_global_dict[sorted_probs[0]] = raw_fdr_dict[sorted_probs[0]]

    # For each subsequent probability, FDR should be <= previous FDR
    for i in range(1, len(sorted_probs)):
        current_prob = sorted_probs[i]
        previous_prob = sorted_probs[i-1]
        
        raw_fdr = raw_fdr_dict[current_prob]
        previous_fdr = fdr_global_dict[previous_prob]
        
        # FDR cannot increase as probability increases
        fdr_global_dict[current_prob] = min(raw_fdr, previous_fdr)

    df_real['FDR'] = df_real['predicted_probability'].map(fdr_global_dict)

    # --- Flag preys based on global_cv ---
    cv_threshold = np.percentile(df_real['global_cv'], 25)  # 25th percentile threshold
    df_real['global_cv_flag'] = df_real['global_cv'].apply(
        lambda cv: 'likely background' if cv <= cv_threshold else ''
    )

    print(f"\nBackground flagging based on global_cv ≤ {cv_threshold:.4f}")
    print("Example flagged entries:")
    flagged_entries = df_real[df_real['global_cv_flag'] != ''][['Prey', 'global_cv', 'global_cv_flag']].head()
    if not flagged_entries.empty:
        print(flagged_entries)
    else:
        print("No entries flagged")

    # Save results to CSV
    df_real.to_csv(output_csv_path, index=False)
    print(f"\nModel training and bait-specific decoy FDR estimation completed.")
    print(f"Results saved to {output_csv_path}")
    print(f"Final dataset shape: {df_real.shape}")
    
    return df_real


# Example usage:
# Step 1: Feature engineering
# features_df = feature_engineering(
#     intensity_file='Benchmarking_BioID_DIA.tsv',
#     controls=['EGFP', 'Empty', 'NminiTurbo'],
#     output_csv_path='aggregated_features_labeled_Benchmarking_BioID_DIA.csv'
# )

# Step 2: Training and scoring
# result_df = train_and_score(
#     features_csv_path='aggregated_features_labeled_Benchmarking_BioID_DIA.csv',
#     output_csv_path='results_Benchmarking_BioID_DIA.csv',
#     initial_positives=15,
#     initial_negatives=200
# )