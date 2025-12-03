# app/profiling/clustering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def run_sku_clustering(profile: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Clustering SKU berdasarkan:
    - qty_mean
    - cv
    - zero_ratio

    Sama dengan versi offline untuk LGBM.
    """
    cluster_feats = ["qty_mean", "cv", "zero_ratio"]
    prof_clean = profile.dropna(subset=cluster_feats).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(prof_clean[cluster_feats].values)

    km = KMeans(n_clusters=n_clusters, random_state=1337, n_init="auto")
    prof_clean["cluster"] = km.fit_predict(X_scaled)

    profile = profile.merge(
        prof_clean[["cabang", "sku", "cluster"]],
        on=["cabang", "sku"],
        how="left",
    )

    profile["cluster"] = profile["cluster"].fillna(-1).astype(int)
    return profile
