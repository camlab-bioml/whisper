# whisper/protein_train.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

def train_and_score_protein(
    features_df: pd.DataFrame,
    initial_positives: int = 15,
    initial_negatives: int = 200,
) -> pd.DataFrame:
    """
    Train a model at the protein level using hierarchical bait clustering,
    pseudo-label assignment, decoy-based monotonic FDR estimation, and global CV flagging.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame returned from protein_features.feature_engineering().
    initial_positives : int, default=15
        Number of initial positive preys to assign per strong bait.
    initial_negatives : int, default=200
        Number of initial negatives to assign per bait.

    Returns
    -------
    pd.DataFrame
        Scored table with predicted probabilities, FDR, and background flags.
    """

    df_real = features_df.copy().sort_values(["Bait", "Prey"]).reset_index(drop=True)
    np.random.seed(42)

    feature_columns = [
        'log_fold_change', 'snr', 'mean_diff', 'median_diff',
        'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio',
        'zero_or_neg_fc',
    ]
    X_real = df_real[feature_columns].values

    # === Hierarchical clustering of baits ===
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

    bait_cluster_map = {b: int(c) for b, c in zip(bait_names, clusters)}

    # determine strong cluster (largest size, then highest mean std)
    unique_clusters = np.unique(clusters)
    cluster_sizes = {c: int(np.sum(clusters == c)) for c in unique_clusters}
    cluster_means = {c: float(bait_scores[clusters == c].mean()) for c in unique_clusters}
    max_size = max(cluster_sizes.values())
    cands = [c for c, n in cluster_sizes.items() if n == max_size]
    strong_cluster_id = cands[0] if len(cands) == 1 else max(cands, key=lambda c: cluster_means[c])

    strong_baits = [b for b in bait_names if bait_cluster_map[b] == strong_cluster_id]

    # === Assign positives ===
    bait_scaled_positives = {
        bait: (initial_positives if bait in strong_baits else 0)
        for bait in df_real['Bait'].unique()
    }

    y_labels = pd.Series(0, index=df_real.index)
    for bait in df_real['Bait'].unique():
        bait_df = df_real[df_real['Bait'] == bait]
        N_pos = bait_scaled_positives[bait]

        if N_pos > 0:
            ranked = bait_df.sort_values('composite_score', ascending=False)
            elig_pos = ranked[ranked['single_rep_flag'] != 1]
            top_pos = elig_pos.index[:N_pos]
            y_labels.loc[top_pos] = 1

            remaining = bait_df.drop(index=top_pos, errors='ignore')
            bottom_neg = remaining['composite_score'].nsmallest(initial_negatives).index
            y_labels.loc[bottom_neg] = -1

    # === Train classifier ===
    labeled_idx = y_labels[y_labels != 0].index
    X_train = X_real[labeled_idx]
    y_train = y_labels.loc[labeled_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    bag_rf = BaggingClassifier(estimator=rf, n_estimators=100, random_state=42)
    bag_rf.fit(X_train_scaled, y_train)

    X_scaled = scaler.transform(X_real)
    df_real["predicted_probability"] = bag_rf.predict_proba(X_scaled)[:, 1]

    # === Decoy-based FDR estimation ===
    all_decoy_probs = []
    for i, bait in enumerate(df_real['Bait'].unique()):
        np.random.seed(42 + i)
        bait_df = df_real[df_real['Bait'] == bait]
        df_decoy = bait_df.copy()
        for col in feature_columns:
            df_decoy[col] = np.random.permutation(df_decoy[col].values)
        X_decoy = scaler.transform(df_decoy[feature_columns].values)
        all_decoy_probs.extend(bag_rf.predict_proba(X_decoy)[:, 1])

    all_decoy_probs = np.array(all_decoy_probs)
    unique_probs = np.unique(df_real["predicted_probability"].values)

    raw_fdr = {}
    for p in unique_probs:
        n_real = (df_real["predicted_probability"] >= p).sum()
        n_decoy = (all_decoy_probs >= p).sum()
        raw_fdr[p] = min(n_decoy / n_real if n_real > 0 else 1.0, 1.0)

    sorted_probs = np.sort(unique_probs)
    fdr_map = {sorted_probs[0]: raw_fdr[sorted_probs[0]]}
    for i in range(1, len(sorted_probs)):
        curr = sorted_probs[i]
        prev = sorted_probs[i - 1]
        fdr_map[curr] = min(raw_fdr[curr], fdr_map[prev])

    df_real["FDR"] = df_real["predicted_probability"].map(fdr_map)

    # === Global CV background flag ===
    if "global_cv" in df_real.columns:
        cv_thresh = np.nanpercentile(df_real["global_cv"], 25)
        df_real["global_cv_flag"] = df_real["global_cv"].apply(
            lambda cv: "likely background" if cv <= cv_thresh else ""
        )
    else:
        df_real["global_cv_flag"] = ""

    # === Save ===
    df_real.to_csv("whisper_protein_scores.csv", index=False)
    return df_real
