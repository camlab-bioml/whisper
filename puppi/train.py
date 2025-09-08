import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import os

def train_and_score(
    features_df: pd.DataFrame,
    initial_positives: int = 15,
    initial_negatives: int = 200,
    output_file: str = 'puppi_results.csv'
) -> pd.DataFrame:
    """
    Train a PU-learning model and assign FDR scores to BioID features.

    Parameters:
        features_df (pd.DataFrame): Input feature dataframe with one row per bait-prey.
        initial_positives (int): Number of positives to sample per strong bait.
        initial_negatives (int): Number of negatives to sample per bait.
        output_file (str): Path to save results.

    Returns:
        pd.DataFrame: The same input dataframe with added prediction and FDR columns.
    """

    np.random.seed(42)

    feature_columns = [
        'log_fold_change', 'snr', 'mean_diff', 'median_diff',
        'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio',
        'zero_or_neg_fc',
    ]
    
    df_real = features_df.copy()
    df_real = df_real.sort_values(["Bait", "Prey"]).reset_index(drop=True)
    X_real = df_real[feature_columns].values

    # --- Bait clustering to determine strong/weak baits ---
    bait_top50_stds = {
        bait: df_real[df_real['Bait'] == bait]['composite_score'].nlargest(50).std()
        for bait in df_real['Bait'].unique()
    }

    bait_names = np.array(list(bait_top50_stds.keys()))
    bait_scores = np.array(list(bait_top50_stds.values())).reshape(-1, 1)

    if len(bait_names) > 2:
        linkage_matrix = linkage(bait_scores, method='ward')
        clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
    else:
        clusters = np.ones(len(bait_names), dtype=int)

    bait_cluster_map = {bait: int(cluster) for bait, cluster in zip(bait_names, clusters)}

    unique_clusters = np.unique(clusters)
    cluster_sizes = {c: int(np.sum(clusters == c)) for c in unique_clusters}
    cluster_means = {c: float(bait_scores[clusters == c].mean()) for c in unique_clusters}
    max_size = max(cluster_sizes.values())
    cands = [c for c, n in cluster_sizes.items() if n == max_size]

    strong_cluster_id = cands[0] if len(cands) == 1 else max(cands, key=lambda c: cluster_means[c])
    strong_baits = [b for b in bait_names if bait_cluster_map[b] == strong_cluster_id]

    bait_scaled_positives = {
        bait: (initial_positives if bait in strong_baits else 0)
        for bait in df_real['Bait'].unique()
    }

    # --- Label positives & negatives ---
    y_train_labels = pd.Series(0, index=df_real.index)

    for bait in df_real['Bait'].unique():
        bait_df = df_real[df_real['Bait'] == bait].copy()
        N_positives = bait_scaled_positives[bait]

        if N_positives > 0:
            ranked = bait_df.sort_values('composite_score', ascending=False)
            elig_pos = ranked[ranked['single_rep_flag'] != 1]
            top_positives = elig_pos.index[:N_positives]
            y_train_labels.loc[top_positives] = 1

            remaining = bait_df.drop(index=top_positives, errors='ignore')
            bottom_negatives = remaining['composite_score'].nsmallest(initial_negatives).index
            y_train_labels.loc[bottom_negatives] = -1

    # --- Train PU classifier ---
    labeled_indices = y_train_labels[y_train_labels != 0].index
    X_train = X_real[labeled_indices]
    y_train = y_train_labels.loc[labeled_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    bagging_rf = BaggingClassifier(estimator=rf_classifier, n_estimators=100, random_state=42)
    bagging_rf.fit(X_train_scaled, y_train)

    X_scaled = scaler.transform(X_real)
    y_pred_proba_calibrated = bagging_rf.predict_proba(X_scaled)[:, 1]
    df_real['predicted_probability'] = y_pred_proba_calibrated

    # --- Bait-Specific Decoy Generation ---
    fdr_values_list = []
    decoy_bait_list = []

    for i, bait in enumerate(df_real['Bait'].unique()):
        np.random.seed(42 + i)  # reproducible decoy shuffling    
        for bait in df_real['Bait'].unique():
            bait_df = df_real[df_real['Bait'] == bait].copy()
            df_decoy = bait_df.copy()
            for column in feature_columns:
                df_decoy[column] = np.random.permutation(bait_df[column].values)
            X_decoy = df_decoy[feature_columns].values
            X_decoy_scaled = scaler.transform(X_decoy)
            y_pred_proba_decoy = bagging_rf.predict_proba(X_decoy_scaled)[:, 1]
            df_decoy['predicted_probability'] = y_pred_proba_decoy
            fdr_values_list.append(y_pred_proba_decoy)
            decoy_bait_list.append(df_decoy)

    all_decoy_probs = np.concatenate(fdr_values_list)


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
    cv_threshold = np.percentile(df_real['global_cv'], 25)  # 10th percentile threshold
    df_real['global_cv_flag'] = df_real['global_cv'].apply(
        lambda cv: 'likely background' if cv <= cv_threshold else ''
    )

    return df_real
