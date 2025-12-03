# app/features/outlier_handler.py

import pandas as pd
def winsorize_outliers(df: pd.DataFrame, clip_ratio: float = 0.01) -> pd.DataFrame:
    df = df.copy()

    if "is_train" not in df.columns:
        raise ValueError("winsorize_outliers: kolom 'is_train' tidak ada di DataFrame.")

    if "qty" not in df.columns:
        raise ValueError("winsorize_outliers: kolom 'qty' tidak ada di DataFrame.")

    df_train = df[df["is_train"] == 1].copy()

    if df_train.empty:
        df["qty_wins"] = df["qty"].astype(float)
        return df

    q_stats = (
        df_train.groupby(["cabang", "sku"])["qty"]
        .quantile([clip_ratio, 1 - clip_ratio])
        .unstack()
        .reset_index()
    )
    q_stats.columns = ["cabang", "sku", "q_low", "q_high"]

    df = df.merge(q_stats, on=["cabang", "sku"], how="left")

    df["qty_wins"] = df["qty"].astype(float)
    mask = df["q_low"].notna()

    df.loc[mask, "qty_wins"] = df.loc[mask].apply(
        lambda row: max(min(row["qty"], row["q_high"]), row["q_low"]),
        axis=1,
    )

    # TIDAK DI-DROP. SAMA DENGAN OFFLINE.
    return df
