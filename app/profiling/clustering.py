from __future__ import annotations
import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def _norm_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cabang" in df.columns:
        df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    if "sku" in df.columns:
        df["sku"] = df["sku"].astype(str).str.strip().str.upper()
    return df

def run_sku_clustering(profile: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:

    profile = profile.copy()
    if profile is None or profile.empty:
        out = pd.DataFrame(columns=["cabang", "sku", "cluster"])
        return out

    profile = _norm_key(profile)

    cluster_feats = ["qty_mean", "cv", "zero_ratio"]
    for c in cluster_feats:
        if c not in profile.columns:
            raise ValueError(f"run_sku_clustering: kolom '{c}' tidak ada di profile.")

    prof_clean = profile.dropna(subset=cluster_feats).copy()
    if prof_clean.empty:
        profile["cluster"] = -1
        return profile

    # pastikan numeric
    for c in cluster_feats:
        prof_clean[c] = pd.to_numeric(prof_clean[c], errors="coerce")
    prof_clean = prof_clean.dropna(subset=cluster_feats).copy()
    if prof_clean.empty:
        profile["cluster"] = -1
        return profile

    X = prof_clean[cluster_feats].values.astype(float)

    # kalau semua fitur konstan, KMeans tidak masuk akal
    if np.all(np.nanstd(X, axis=0) < 1e-12):
        prof_clean["cluster"] = 0
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            km = KMeans(n_clusters=int(n_clusters), random_state=1337, n_init="auto")
        except TypeError:
            km = KMeans(n_clusters=int(n_clusters), random_state=1337, n_init=10)

        prof_clean["cluster"] = km.fit_predict(X_scaled).astype(int)

    # merge balik ke full profile
    out = profile.merge(
        prof_clean[["cabang", "sku", "cluster"]],
        on=["cabang", "sku"],
        how="left",
    )

    out["cluster"] = pd.to_numeric(out["cluster"], errors="coerce").fillna(-1).astype(int)
    return out


def apply_cluster_to_sku_profile(engine, profile_clustered: pd.DataFrame) -> int:
    if profile_clustered is None or profile_clustered.empty:
        return 0

    df = profile_clustered.copy()
    df = _norm_key(df)

    if "cluster" not in df.columns:
        raise ValueError("apply_cluster_to_sku_profile: kolom 'cluster' tidak ada.")

    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(int)

    upd = text("""
        UPDATE sku_profile
        SET
            cluster = :cluster,
            last_updated = CURRENT_TIMESTAMP
        WHERE cabang = :cabang AND sku = :sku
    """)

    touched = 0
    with engine.begin() as conn:
        for r in df[["cabang", "sku", "cluster"]].itertuples(index=False):
            res = conn.execute(
                upd,
                {"cabang": r.cabang, "sku": r.sku, "cluster": int(r.cluster)},
            )
            touched += int(res.rowcount)

    return int(touched)


def cluster_and_update_from_sku_profile(engine, n_clusters: int = 4) -> int:
    sql = text("""
        SELECT cabang, sku, qty_mean, cv, zero_ratio
        FROM sku_profile
        ORDER BY cabang, sku
    """)

    with engine.begin() as conn:
        prof = pd.read_sql(sql, conn)

    prof = _norm_key(prof)
    if prof.empty:
        return 0

    prof_clustered = run_sku_clustering(prof, n_clusters=n_clusters)
    return apply_cluster_to_sku_profile(engine, prof_clustered)
