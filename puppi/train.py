# puppi/train.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import warnings

warnings.filterwarnings('ignore')


def train_and_score(
    df_real: pd.DataFrame,
    initial_positives: int = 15,   
    initial_negatives: int = 200,

) -> pd.DataFrame:
    """
    Train Bagging(RandomForest) using your PU labeling scheme and compute global decoy FDR.
    Logic matches the provided Benchmarking BioID DIA script.

    Returns the input df_real with added columns:
      - 'predicted_probability'
      - 'FDR'
      - 'global_cv_flag'
    Optionally saves to CSV if save_path is provided.
    """

    # Set random seed exactly as in your script (controls decoy permutations, etc.)
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

    # keep your original cap at 15
    initial_positives = 15

    bait_scaled_positives = {bait: (initial_positives if bait in strong_baits else 0)
                            for bait in baits}

    print("\nAssigned positives:")
    for bait, npos in bait_scaled_positives.items():
        print(f"Bait: {bait}, Positives: {npos}")

    # Label positives & negatives
    y_train_labels = pd.Series(0, index=df_real.index)
    initial_negatives = 200



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
            bottom_negatives = remaining['composite_score'].nsmallest(initial_negatives).index
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
    cv_threshold = np.percentile(df_real['global_cv'], 25)  # 10th percentile threshold
    df_real['global_cv_flag'] = df_real['global_cv'].apply(
        lambda cv: 'likely background' if cv <= cv_threshold else ''
    )
