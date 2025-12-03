# app/features/time_features.py

import pandas as pd
import numpy as np

def add_all_lags_and_rollings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cabang","sku","periode"]).copy()

    # Time features
    df["month"] = df["periode"].dt.month
    df["year"] = df["periode"].dt.year
    df["qtr"] = df["periode"].dt.quarter

    # Lags qty 1–12
    for i in range(1, 13):
        df[f"qty_lag{i}"] = (
            df.groupby(["cabang","sku"])["qty_wins"]
              .shift(i)
        )

    # Rolling windows
    df["qty_rollmean_3"] = (
        df.groupby(["cabang","sku"])["qty_wins"]
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["qty_rollstd_3"] = (
        df.groupby(["cabang","sku"])["qty_wins"]
          .transform(lambda x: x.rolling(3, min_periods=1).std())
    )

    df["qty_rollmean_6"] = (
        df.groupby(["cabang","sku"])["qty_wins"]
          .transform(lambda x: x.rolling(6, min_periods=1).mean())
    )
    df["qty_rollstd_6"] = (
        df.groupby(["cabang","sku"])["qty_wins"]
          .transform(lambda x: x.rolling(6, min_periods=1).std())
    )

    df["qty_rollmean_12"] = (
        df.groupby(["cabang","sku"])["qty_wins"]
          .transform(lambda x: x.rolling(12, min_periods=1).mean())
    )
    df["qty_rollstd_12"] = (
        df.groupby(["cabang","sku"])["qty_wins"]
          .transform(lambda x: x.rolling(12, min_periods=1).std())
    )

    return df
