# whsiper/fragment_train.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings("ignore")


def train_and_score_fragment(
    features_df: pd.DataFrame,
    initial_positives: int = 15,
    initial_negatives: int = 200,
    random_state: int = 42,
    save_dir: str = ".",
    fragment_out: str = "whisper_fragment_scores.csv",
    protein_out: str = "whisper_protein_scores_from_fragments.csv",
    aggregate_strategy: str = "max",   # "max" or "mean" for fragment->protein prob aggregation
):
    """
    Train a model on FRAGMENT-level features, compute bait-specific decoy FDR,
    and aggregate to PROTEIN-level scores per bait.

    Expected columns in `features_df`:
      - Bait, Protein, Peptide, Fragment
      - composite_score, global_cv (optional), single_rep_flag (optional)
      - Feature columns:
          ['log_fold_change','snr','mean_diff','median_diff',
           'replicate_fold_change_sd','bait_cv','bait_control_sd_ratio','zero_or_neg_fc']

    Saves:
      - <save_dir>/<fragment_out>: fragment-level scores with FDR
      - <save_dir>/<protein_out>: protein-level aggregation per bait

    Returns:
      (fragment_df, protein_df)
    """
    rng = np.random.RandomState(random_state)

    # Stable order
    df = (
        features_df.copy()
        .sort_values(["Bait", "Protein", "Peptide", "Fragment"])
        .reset_index(drop=True)
    )

    feature_columns = [
        "log_fold_change", "snr", "mean_diff", "median_diff",
        "replicate_fold_change_sd", "bait_cv", "bait_control_sd_ratio",
        "zero_or_neg_fc",
    ]
    X = df[feature_columns].values

    # ---------- Cluster baits to identify "strong" set ----------
    bait_top50_stds = {
        b: df[df["Bait"] == b]["composite_score"].nlargest(50).std()
        for b in df["Bait"].unique()
    }
    bait_names = np.array(list(bait_top50_stds.keys()))
    bait_scores = np.array(list(bait_top50_stds.values()), dtype=float).reshape(-1, 1)

    if len(bait_names) > 2:
        Z = linkage(bait_scores, method="ward")
        clusters = fcluster(Z, t=2, criterion="maxclust")
    else:
        clusters = np.ones(len(bait_names), dtype=int)

    cluster_sizes = {c: int(np.sum(clusters == c)) for c in np.unique(clusters)}
    cluster_means = {c: float(bait_scores[clusters == c].mean()) for c in np.unique(clusters)}
    max_size = max(cluster_sizes.values())
    cands = [c for c, n in cluster_sizes.items() if n == max_size]
    strong_cluster_id = cands[0] if len(cands) == 1 else max(cands, key=lambda c: cluster_means[c])
    strong_baits = [b for b, c in zip(bait_names, clusters) if c == strong_cluster_id]

    # ---------- Pseudo-labels ----------
    y = pd.Series(0, index=df.index)  # 0=unlabeled, 1=pos, -1=neg
    bait_pos_quota = {b: (initial_positives if b in strong_baits else 0) for b in df["Bait"].unique()}

    for bait in df["Bait"].unique():
        sub = df[df["Bait"] == bait].copy()
        n_pos = bait_pos_quota[bait]
        if n_pos > 0:
            ranked = sub.sort_values("composite_score", ascending=False)
            elig = ranked[ranked.get("single_rep_flag", 0) != 1]  # exclude single-rep spikes if present
            pos_idx = elig.index[:n_pos]
            y.loc[pos_idx] = 1

            remaining = sub.drop(index=pos_idx, errors="ignore")
            neg_idx = remaining["composite_score"].nsmallest(initial_negatives).index
            y.loc[neg_idx] = -1

    labeled_idx = y[y != 0].index
    X_tr = X[labeled_idx]
    y_tr = y.loc[labeled_idx].values

    # ---------- Train bagged RF ----------
    scaler = StandardScaler().fit(X_tr)
    X_tr_std = scaler.transform(X_tr)

    base = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf = BaggingClassifier(estimator=base, n_estimators=100, random_state=random_state)
    clf.fit(X_tr_std, y_tr)

    X_std = scaler.transform(X)
    df["predicted_probability"] = clf.predict_proba(X_std)[:, 1]

    # ---------- Bait-specific decoy shuffles for FDR ----------
    decoys = []
    for i, bait in enumerate(df["Bait"].unique()):
        rng_i = np.random.RandomState(random_state + i)
        sub = df[df["Bait"] == bait].copy()
        dec = sub[feature_columns].apply(lambda col: rng_i.permutation(col.values))
        X_dec = scaler.transform(dec.values)
        decoys.append(clf.predict_proba(X_dec)[:, 1])
    decoy_probs = np.concatenate(decoys) if len(decoys) else np.array([])

    real_probs = df["predicted_probability"].values
    unique_p = np.unique(real_probs)

    # raw FDR
    raw_fdr = {}
    for p in unique_p:
        n_real = np.sum(real_probs >= p)
        n_dec = np.sum(decoy_probs >= p) if decoy_probs.size else 0
        raw_fdr[p] = min(n_dec / n_real if n_real > 0 else 1.0, 1.0)

    # monotone FDR (non-increasing with prob)
    sorted_p = np.sort(unique_p)
    mono_fdr = {sorted_p[0]: raw_fdr[sorted_p[0]]}
    for i in range(1, len(sorted_p)):
        p = sorted_p[i]
        prev = sorted_p[i - 1]
        mono_fdr[p] = min(raw_fdr[p], mono_fdr[prev])

    df["FDR"] = df["predicted_probability"].map(mono_fdr)

    # ---------- Background flag by global CV (optional) ----------
    if "global_cv" in df.columns:
        cv_thresh = np.nanpercentile(df["global_cv"], 25)
        df["global_cv_flag"] = df["global_cv"].apply(
            lambda v: "likely background" if pd.notna(v) and v <= cv_thresh else ""
        )
    else:
        df["global_cv_flag"] = ""

    # ===== AGGREGATE TO PROTEIN-LEVEL (per bait) =====
    prob_agg = "max" if aggregate_strategy.lower() == "max" else "mean"

    grp = df.groupby(["Bait", "Protein"])
    protein_df = grp.agg(
        predicted_probability=("predicted_probability", prob_agg),
        FDR=("FDR", "min"),
        n_fragments=("Fragment", "count"),
        n_background=("global_cv_flag", lambda x: (x == "likely background").sum()),
        mean_cv=("global_cv", "mean"),
    ).reset_index()

    protein_df["background_flag_protein"] = np.where(
        protein_df["n_background"] >= 0.5 * protein_df["n_fragments"],
        "likely background",
        "",
    )

    # ----- Save both -----
    os.makedirs(save_dir, exist_ok=True)
    fragment_path = os.path.join(save_dir, fragment_out)
    protein_path = os.path.join(save_dir, protein_out)

    df.to_csv(fragment_path, index=False)
    protein_df.to_csv(protein_path, index=False)

    return df, protein_df
