# puppi/train.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import warnings

warnings.filterwarnings("ignore")


def train_and_score(df_real: pd.DataFrame, initial_positives: int = 10, initial_negatives: int = 200) -> pd.DataFrame:
    """
    Trains a PU learning model using Bagging with Random Forest and estimates FDR using bait-specific decoys.

    Parameters:
        df_real (pd.DataFrame): Feature-engineered dataframe with 'composite_score', 'Bait', and other features.
        initial_positives (int): Number of initial positives per strong bait (default = 10).
        initial_negatives (int): Number of negatives per bait (default = 200).

    Returns:
        pd.DataFrame: The same dataframe with added 'predicted_probability', 'FDR', and 'global_cv_flag' columns.
    """

    np.random.seed(42)

    feature_columns = [
        'log_fold_change', 'snr', 'mean_diff', 'median_diff',
        'replicate_fold_change_sd', 'bait_cv', 'bait_control_sd_ratio',
        'zero_or_neg_fc'
    ]
    X_real = df_real[feature_columns].values

    bait_top50_stds = {
        bait: df_real[df_real["Bait"] == bait]["composite_score"].nlargest(50).std()
        for bait in df_real["Bait"].unique()
    }
    bait_names = np.array(list(bait_top50_stds.keys()))
    bait_scores = np.array(list(bait_top50_stds.values())).reshape(-1, 1)

    linkage_matrix = linkage(bait_scores, method="ward")
    clusters = fcluster(linkage_matrix, t=1.0, criterion="distance")
    bait_cluster_map = {bait: cluster for bait, cluster in zip(bait_names, clusters)}

    baits = list(df_real["Bait"].unique())
    strong_cluster_id = 1
    strong_baits = [bait for bait in bait_names if bait_cluster_map[bait] == strong_cluster_id]

    all_counts = []
    for bait in strong_baits:
        bait_df = df_real[df_real["Bait"] == bait]
        top50_mean = bait_df["composite_score"].nlargest(50).mean()
        count_above = (bait_df["composite_score"] > top50_mean).sum()
        all_counts.append(count_above)

    average_top_count = initial_positives if int(np.mean(all_counts)) > initial_positives else int(np.mean(all_counts))
    bait_scaled_positives = {
        bait: average_top_count if bait in strong_baits else 0 for bait in baits
    }

    y_train_labels = pd.Series(0, index=df_real.index)
    for bait in baits:
        bait_df = df_real[df_real["Bait"] == bait]
        N_positives = bait_scaled_positives[bait]

        if N_positives > 0:
            top_positives = bait_df["composite_score"].nlargest(N_positives).index
            y_train_labels.loc[top_positives] = 1

            bottom_negatives = bait_df["composite_score"].nsmallest(initial_negatives).index
            y_train_labels.loc[bottom_negatives] = -1

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

    num_decoy_datasets = 1
    decoy_probs = []

    for _ in range(num_decoy_datasets):
        for bait in baits:
            bait_df = df_real[df_real["Bait"] == bait].copy()
            df_decoy = bait_df.copy()
            for col in feature_columns:
                df_decoy[col] = np.random.permutation(bait_df[col].values)

            X_decoy = df_decoy[feature_columns].values
            X_decoy_scaled = scaler.transform(X_decoy)
            y_pred_decoy = bagged_rf.predict_proba(X_decoy_scaled)[:, 1]
            decoy_probs.append(y_pred_decoy)

    all_decoy_probs = np.concatenate(decoy_probs)

    unique_probs = np.unique(y_pred_proba_real)
    fdr_dict = {}
    for p in unique_probs:
        num_real = np.sum(y_pred_proba_real >= p)
        num_decoy = np.sum(all_decoy_probs >= p)
        fdr = num_decoy / num_real if num_real > 0 and num_decoy <= num_real else 1.0
        fdr_dict[p] = fdr

    df_real["FDR"] = df_real["predicted_probability"].map(fdr_dict)

    if "global_cv" in df_real.columns:
        cv_threshold = np.percentile(df_real["global_cv"], 25)
        df_real["global_cv_flag"] = df_real["global_cv"].apply(
            lambda cv: "likely background" if cv <= cv_threshold else ""
        )
    else:
        df_real["global_cv_flag"] = ""

    return df_real
