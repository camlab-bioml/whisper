# puppi/train.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import warnings

warnings.filterwarnings("ignore")


def train_and_score(
    df_real: pd.DataFrame,
    initial_positives: int = 10,   # kept for API compatibility; overridden to 15 per new spec
    initial_negatives: int = 200   # kept for API compatibility; used below
) -> pd.DataFrame:
    """
    Trains a PU learning model using Bagging with Random Forest and estimates FDR using bait-specific decoys.

    Updates vs previous version:
      - Strong-bait detection via hierarchical clustering with tie-breaking by cluster size, then mean score.
      - Positive selection excludes single-replicate spikes (single_rep_flag == 1).
      - Fixed cap of 15 positives per strong bait (overrides initial_positives).
      - Global decoy-based FDR with a monotonic (non-increasing) pass over probability thresholds.
      - Background flagging via global_cv 25th percentile.

    Parameters
    ----------
    df_real : pd.DataFrame
        Feature-engineered dataframe with columns including:
        'Bait', 'Prey', feature columns, 'composite_score', 'single_rep_flag', 'global_cv'.
    initial_positives : int
        (Ignored; preserved for API compatibility) Kept but overridden to 15 as per new spec.
    initial_negatives : int
        Number of negatives per bait (default = 200).

    Returns
    -------
    pd.DataFrame
        Input dataframe with added:
        - 'predicted_probability'
        - 'FDR'
        - 'global_cv_flag'
    """

    np.random.seed(42)

    # Feature columns (match your latest spec)
    feature_columns = [
        "log_fold_change", "snr", "mean_diff", "median_diff",
        "replicate_fold_change_sd", "bait_cv", "bait_control_sd_ratio",
        "zero_or_neg_fc",
    ]
    X_real = df_real[feature_columns].values

    # --- Hierarchical Clustering of Baits using std of top-50 composite_score ---
    bait_top50_stds = {
        bait: df_real[df_real["Bait"] == bait]["composite_score"].nlargest(50).std()
        for bait in df_real["Bait"].unique()
    }
    bait_names = np.array(list(bait_top50_stds.keys()))
    bait_scores = np.array(list(bait_top50_stds.values())).reshape(-1, 1)

    # Cluster if > 2 baits; otherwise treat all as strong
    if len(bait_names) > 2:
        linkage_matrix = linkage(bait_scores, method="ward")
        clusters = fcluster(linkage_matrix, t=2, criterion="maxclust")
    else:
        clusters = np.ones(len(bait_names), dtype=int)

    bait_cluster_map = {bait: int(cluster) for bait, cluster in zip(bait_names, clusters)}

    # Decide which cluster is "strong":
    # 1) cluster with more members; 2) if tie, cluster with higher mean bait score
    unique_clusters = np.unique(clusters)
    cluster_sizes = {c: int(np.sum(clusters == c)) for c in unique_clusters}
    cluster_means = {c: float(bait_scores[clusters == c].mean()) for c in unique_clusters}

    max_size = max(cluster_sizes.values())
    candidates = [c for c, sz in cluster_sizes.items() if sz == max_size]
    if len(candidates) == 1:
        strong_cluster_id = candidates[0]
    else:
        strong_cluster_id = max(candidates, key=lambda c: cluster_means[c])

    # --- Distribute positives: cap at 15 for strong baits; 0 for others ---
    baits = list(df_real["Bait"].unique())
    strong_baits = [b for b in bait_names if bait_cluster_map[b] == strong_cluster_id]

    # Fixed cap per your new script (override initial_positives)
    average_top_count = 15
    bait_scaled_positives = {bait: (average_top_count if bait in strong_baits else 0) for bait in baits}

    # --- Label positives & negatives per bait (exclude single-replicate spikes from positives) ---
    y_train_labels = pd.Series(0, index=df_real.index)
    N_negatives = initial_negatives

    for bait in df_real["Bait"].unique():
        bait_df = df_real[df_real["Bait"] == bait].copy()
        N_positives = bait_scaled_positives[bait]

        if N_positives > 0:
            ranked = bait_df.sort_values("composite_score", ascending=False)

            # Exclude single-replicate spikes
            elig_pos = ranked[ranked.get("single_rep_flag", 0) != 1]

            # Take top N positives from eligible
            top_positives = elig_pos.index[:N_positives]
            y_train_labels.loc[top_positives] = 1

            # Negatives: bottom of the remaining
            remaining = bait_df.drop(index=top_positives, errors="ignore")
            bottom_negatives = remaining["composite_score"].nsmallest(N_negatives).index
            y_train_labels.loc[bottom_negatives] = -1

    # --- Train bagged Random Forest on labeled subset ---
    labeled_indices = y_train_labels[y_train_labels != 0].index
    X_train = df_real.loc[labeled_indices, feature_columns].values
    y_train = y_train_labels.loc[labeled_indices].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    bagged_rf = BaggingClassifier(estimator=rf, n_estimators=100, random_state=42)
    bagged_rf.fit(X_train_scaled, y_train)

    X_scaled = scaler.transform(X_real)
    y_pred_proba_real = bagged_rf.predict_proba(X_scaled)[:, 1]
    df_real["predicted_probability"] = y_pred_proba_real

    # -----------------------------
    # Decoy Generation for Global FDR
    # -----------------------------
    all_decoy_probs = []

    for bait in df_real["Bait"].unique():
        bait_df = df_real[df_real["Bait"] == bait].copy()
        df_decoy = bait_df.copy()

        for col in feature_columns:
            df_decoy[col] = np.random.permutation(df_decoy[col].values)

        X_decoy = df_decoy[feature_columns].values
        X_decoy_scaled = scaler.transform(X_decoy)
        decoy_probs = bagged_rf.predict_proba(X_decoy_scaled)[:, 1]
        all_decoy_probs.extend(decoy_probs)

    all_decoy_probs = np.asarray(all_decoy_probs)

    # --- Global FDR calculation with monotonic pass ---
    unique_real_probs = np.unique(df_real["predicted_probability"])

    # First pass: raw FDR at each prob threshold
    raw_fdr = {}
    for p in unique_real_probs:
        real_count = int((df_real["predicted_probability"] >= p).sum())
        decoy_count = int((all_decoy_probs >= p).sum())
        raw = decoy_count / real_count if real_count > 0 else 1.0
        raw_fdr[p] = min(raw, 1.0)

    # Second pass: enforce monotonic non-increasing FDR as probability increases
    sorted_probs = np.sort(unique_real_probs)  # ascending
    fdr_global = {}
    fdr_global[sorted_probs[0]] = raw_fdr[sorted_probs[0]]
    for i in range(1, len(sorted_probs)):
        p = sorted_probs[i]
        prev_p = sorted_probs[i - 1]
        fdr_global[p] = min(raw_fdr[p], fdr_global[prev_p])

    df_real["FDR"] = df_real["predicted_probability"].map(fdr_global)

    # --- Flag likely background via global_cv (≤ 25th percentile) ---
    if "global_cv" in df_real.columns:
        cv_threshold = np.percentile(df_real["global_cv"], 25)
        df_real["global_cv_flag"] = df_real["global_cv"].apply(
            lambda cv: "likely background" if cv <= cv_threshold else ""
        )
    else:
        df_real["global_cv_flag"] = ""

    return df_real
