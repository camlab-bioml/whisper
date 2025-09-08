import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

def train_and_score(
    features_df: pd.DataFrame,
    initial_positives: int = 15,
    initial_negatives: int = 200,
) -> pd.DataFrame:
    """
    Trains a PU-learning model using hierarchical bait clustering, assigns positives and negatives,
    computes decoy-based monotonic FDR, and flags likely background preys based on global_cv.

    Parameters:
        features_df (pd.DataFrame): Aggregated feature matrix with 'global_cv', 'composite_score', etc.
        initial_positives (int): Number of positives to assign per strong bait.
        initial_negatives (int): Number of negatives to assign per bait.
        output_file (str): Optional path to write CSV output.

    Returns:
        pd.DataFrame: Scored DataFrame with columns: predicted_probability, FDR, global_cv_flag
    """
    df_real = features_df.copy()
    df_real = df_real.sort_values(["Bait", "Prey"]).reset_index(drop=True)

    np.random.seed(42)

    feature_columns = [
        'log_fold_change', 'snr', 'mean_diff', 'median_diff', 
        'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio', 
        'zero_or_neg_fc',
    ]
    X_real = df_real[feature_columns].values

    # --- Hierarchical Clustering of Baits ---
    bait_top50_stds = {
        bait: df_real[df_real['Bait'] == bait]['composite_score'].nlargest(50).std()
        for bait in df_real['Bait'].unique()
    }

    bait_names  = np.array(list(bait_top50_stds.keys()))
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

    # --- Assign positives proportionally to top-50 mean ---
    all_counts = []
    for bait in strong_baits:
        bait_df = df_real[df_real['Bait'] == bait]
        top50_mean = bait_df['composite_score'].nlargest(50).mean()
        count_above = (bait_df['composite_score'] > top50_mean).sum()
        all_counts.append(count_above)

    bait_scaled_positives = {
        bait: (initial_positives if bait in strong_baits else 0)
        for bait in df_real['Bait'].unique()
    }

    # --- Label positives and negatives ---
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

    # --- Classifier training ---
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

    # --- Bait-wise Decoy Generation ---
    all_decoy_probs = []
    for i, bait in enumerate(df_real['Bait'].unique()):
        np.random.seed(42 + i)
        bait_df = df_real[df_real['Bait'] == bait]
        df_decoy = bait_df.copy()
        for col in feature_columns:
            df_decoy[col] = np.random.permutation(df_decoy[col].values)
        X_decoy = df_decoy[feature_columns].values
        X_decoy_scaled = scaler.transform(X_decoy)
        decoy_probs = bagging_rf.predict_proba(X_decoy_scaled)[:, 1]
        all_decoy_probs.extend(decoy_probs)

    all_decoy_probs_array = np.array(all_decoy_probs)
    unique_real_probs = np.unique(df_real['predicted_probability'])

    # --- FDR Estimation ---
    raw_fdr_dict = {}
    for prob in unique_real_probs:
        real_count = (df_real['predicted_probability'] >= prob).sum()
        decoy_count = np.sum(all_decoy_probs_array >= prob)
        raw_fdr = min(decoy_count / real_count if real_count > 0 else 1.0, 1.0)
        raw_fdr_dict[prob] = raw_fdr

    sorted_probs = np.sort(unique_real_probs)
    fdr_global_dict = {}
    fdr_global_dict[sorted_probs[0]] = raw_fdr_dict[sorted_probs[0]]
    for i in range(1, len(sorted_probs)):
        current_prob = sorted_probs[i]
        previous_prob = sorted_probs[i - 1]
        raw_fdr = raw_fdr_dict[current_prob]
        fdr_global_dict[current_prob] = min(raw_fdr, fdr_global_dict[previous_prob])

    df_real['FDR'] = df_real['predicted_probability'].map(fdr_global_dict)

    # --- Background flagging based on global CV ---
    if 'global_cv' in df_real.columns:
        cv_threshold = np.percentile(df_real['global_cv'], 25)
        df_real['global_cv_flag'] = df_real['global_cv'].apply(
            lambda cv: 'likely background' if cv <= cv_threshold else ''
        )
        print(f"\nBackground flagging based on global_cv ≤ {cv_threshold:.4f}")
        print("Example flagged entries:")
        print(df_real[df_real['global_cv_flag'] != ''][['Prey', 'global_cv', 'global_cv_flag']].head())
    else:
        df_real['global_cv_flag'] = ''

    df_real.to_csv('puppi_result.csv', index=False)

    return df_real
