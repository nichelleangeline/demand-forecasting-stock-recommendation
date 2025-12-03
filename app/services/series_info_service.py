# app/services/series_info_service.py

import pandas as pd
import numpy as np


def compute_series_info_dynamic(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung statistik per (cabang, sku) dan flag eligible_model
    berdasarkan histori TERBARU di panel_global_monthly.

    Asumsi kolom minimal:
    - cabang, sku, periode, qty
    """

    if panel.empty:
        return pd.DataFrame(columns=[
            "cabang", "sku",
            "n_months", "nonzero_months", "total_qty",
            "qty_12m", "qty_6m",
            "zero_ratio_train",
            "has_last_month",
            "months_since_last_nz",
            "alive_recent",
            "eligible_model",
        ])

    df = panel.copy()
    df["periode"] = pd.to_datetime(df["periode"])
    df = df.sort_values(["cabang", "sku", "periode"])

    # periode terakhir di data (sinkron dengan DB)
    last_period = df["periode"].max()
    last_per_m = last_period.to_period("M")

    # pakai seluruh histori untuk agregat
    w = df.copy()

    # ada data di bulan terakhir atau tidak
    has_last = (
        w.query("periode == @last_period")
         .groupby(["cabang", "sku"], as_index=False)
         .size()
         .rename(columns={"size": "has_last_month"})
    )

    # agregat dasar
    agg = (
        w.groupby(["cabang", "sku"], as_index=False)
         .agg(
             n_months=("qty", "size"),
             nonzero_months=("qty", lambda s: (s > 0).sum()),
             total_qty=("qty", "sum"),
         )
    )

    # 12 bulan terakhir dari histori
    last12 = (
        w.sort_values(["cabang", "sku", "periode"])
         .groupby(["cabang", "sku"], as_index=False)
         .agg(qty_12m=("qty", lambda s: s.tail(12).sum()))
    )

    # 6 bulan terakhir
    last6 = (
        w.sort_values(["cabang", "sku", "periode"])
         .groupby(["cabang", "sku"], as_index=False)
         .agg(qty_6m=("qty", lambda s: s.tail(6).sum()))
    )

    # zero_ratio_train → pakai seluruh histori sebagai "train"
    zr = (
        w.groupby(["cabang", "sku"], as_index=False)["qty"]
         .apply(lambda s: (s == 0).mean())
         .rename(columns={"qty": "zero_ratio_train"})
    )

    # last non-zero
    nz = (
        w.loc[w["qty"] > 0]
         .groupby(["cabang", "sku"], as_index=False)["periode"]
         .max()
         .rename(columns={"periode": "last_nz"})
    )

    # merge semua info
    info = (
        agg.merge(last12, on=["cabang", "sku"], how="left")
           .merge(last6,  on=["cabang", "sku"], how="left")
           .merge(has_last, on=["cabang", "sku"], how="left")
           .merge(zr, on=["cabang", "sku"], how="left")
           .merge(nz, on=["cabang", "sku"], how="left")
    )

    # bereskan NaN
    info["has_last_month"] = info["has_last_month"].fillna(0).gt(0)
    for c in ["zero_ratio_train", "qty_12m", "qty_6m"]:
        info[c] = info[c].fillna(0)

    info["last_nz"] = pd.to_datetime(info["last_nz"], errors="coerce")

    # hitung selisih bulan dari last_nz ke last_period
    info["months_since_last_nz"] = 999

    mask_nz = info["last_nz"].notna()
    if mask_nz.any():
        last_nz_per = info.loc[mask_nz, "last_nz"].dt.to_period("M")
        diff_months = last_per_m.ordinal - last_nz_per.astype("int64")
        info.loc[mask_nz, "months_since_last_nz"] = diff_months.values

    info["months_since_last_nz"] = info["months_since_last_nz"].astype(int)

    # masih hidup di dekat akhir histori
    info["alive_recent"] = (
        (info["qty_6m"] > 0) &
        (info["months_since_last_nz"] <= 3)
    ).astype(int)

    # RULE ELIGIBLE (versi sistem)
    info["eligible_model"] = (
        (info["n_months"] >= 36) &          # minimal 3 tahun histori
        (info["nonzero_months"] >= 10) &    # tidak super jarang
        (info["qty_12m"] > 0) &             # 12 bulan terakhir masih hidup
        (info["total_qty"] >= 30) &         # tidak cuma sekali dua kali jual
        (info["zero_ratio_train"] <= 0.7) & # max 70% bulan = 0
        (info["has_last_month"] == True) &  # sinkron sampai bulan terakhir di DB
        (info["alive_recent"] == 1)         # last_nz dekat akhir
    ).astype(int)

    return info


def compute_series_info(panel: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Wrapper kompatibilitas lama.
    Abaikan argumen tambahan, pakai logic dynamic terbaru.
    """
    return compute_series_info_dynamic(panel)
