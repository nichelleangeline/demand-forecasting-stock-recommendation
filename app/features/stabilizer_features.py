# app/features/stabilizer_features.py

import pandas as pd
import numpy as np

def add_stabilizer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versi stabilizer untuk LGBM (1:1 dengan script offline):

    - qty_mean_cs, qty_std_cs, qty_cnt_cs: dihitung HANYA dari data train (is_train == 1)
    - roll_mean_3, roll_mean_12: rolling mean causal per (cabang, sku)
    - seasonal_ratio_12: roll_mean_3 / roll_mean_12
    - volatility_score: qty_std_cs / qty_mean_cs

    Wajib ada kolom:
    - cabang, sku, periode, qty, is_train
    """
    if "is_train" not in df.columns:
        raise ValueError("add_stabilizer_features: kolom 'is_train' wajib ada supaya stats pakai TRAIN saja.")

    df = df.copy()

    # Stats SKU HANYA dari train (supaya tidak ngintip test)
    df_train = df[df["is_train"] == 1].copy()

    sku_stats = (
        df_train.groupby(["cabang", "sku"])["qty"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    sku_stats.columns = ["cabang", "sku", "qty_mean_cs", "qty_std_cs", "qty_cnt_cs"]

    # merge ke full panel (train + test)
    df = df.merge(sku_stats, on=["cabang", "sku"], how="left")

    # rolling + seasonal ratio pakai seri full, tapi rolling itu causal (hanya masa lalu)
    df = df.sort_values(["cabang", "sku", "periode"])

    df["roll_mean_3"] = (
        df.groupby(["cabang", "sku"])["qty"]
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["roll_mean_12"] = (
        df.groupby(["cabang", "sku"])["qty"]
          .transform(lambda x: x.rolling(12, min_periods=1).mean())
    )
    df["seasonal_ratio_12"] = df["roll_mean_3"] / (df["roll_mean_12"] + 1e-9)

    # volatility_score dari stats train
    df["volatility_score"] = df["qty_std_cs"] / (df["qty_mean_cs"] + 1e-9)

    return df
