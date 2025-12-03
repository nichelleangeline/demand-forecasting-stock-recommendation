# app/services/panel_builder.py

import numpy as np
import pandas as pd
from sqlalchemy import text

WIN_START_DEFAULT = pd.Timestamp("2021-01-01")


# =========================================
# 1) Windowed panel dari DB (sales_monthly)
# =========================================

def _make_windowed_panel_from_db(engine) -> pd.DataFrame:
    """
    Versi DB dari make_windowed_panel offline:
    - baca sales_monthly
    - bikin kalender bulanan per (cabang, sku)
    - hitung qty, imputed, is_active
    """
    with engine.begin() as conn:
        sales = pd.read_sql(
            """
            SELECT periode, area, cabang, sku, qty
            FROM sales_monthly
            """,
            conn,
            parse_dates=["periode"],
        )

    sales["qty"] = pd.to_numeric(sales["qty"], errors="coerce").fillna(0.0)

    out = []
    for (cab, sku), g in sales.groupby(["cabang", "sku"], sort=False):
        g = g.sort_values("periode")
        cal = pd.date_range(g["periode"].min(), g["periode"].max(), freq="MS")

        idx = pd.MultiIndex.from_product(
            [[cab], [sku], cal],
            names=["cabang", "sku", "periode"]
        )
        g2 = (
            g.set_index(["cabang", "sku", "periode"])
             .reindex(idx)
             .reset_index()
        )

        exist = set(zip(g["cabang"], g["sku"], g["periode"]))
        g2["qty"] = pd.to_numeric(g2["qty"], errors="coerce").fillna(0.0)
        g2["imputed"] = [0 if (cab, sku, p) in exist else 1 for p in g2["periode"]]

        # area langsung dari raw
        g2["area"] = g["area"].iloc[0]

        nz_idx = g2.index[g2["qty"] != 0]
        if len(nz_idx):
            left, right = int(nz_idx.min()), int(nz_idx.max())
            g2["is_active"] = 0
            g2.loc[left:right, "is_active"] = 1
        else:
            g2["is_active"] = 0

        out.append(
            g2[["periode", "area", "cabang", "sku", "qty", "imputed", "is_active"]]
        )

    panel = (
        pd.concat(out, ignore_index=True)
          .sort_values(["cabang", "sku", "periode"])
          .reset_index(drop=True)
    )
    return panel


# =========================================
# 2) Train/Test flags (copy offline)
# =========================================

def _add_train_test_flags(panel: pd.DataFrame,
                          train_end: pd.Timestamp,
                          test_start: pd.Timestamp,
                          test_end: pd.Timestamp) -> pd.DataFrame:
    """
    Versi DB dari logic:
      _nonzero_bounds + is_train / is_test
    """
    panel = panel.copy()

    def _nonzero_bounds(g):
        nz = g.loc[g["qty"] != 0, "periode"]
        if len(nz):
            return pd.Series({"train_start": nz.min(), "nz_last": nz.max()})
        else:
            return pd.Series({"train_start": pd.NaT, "nz_last": pd.NaT})

    nz_bounds = (
        panel.groupby(["cabang", "sku"], sort=False)
             .apply(_nonzero_bounds)
             .reset_index()
    )

    panel = panel.merge(nz_bounds, on=["cabang", "sku"], how="left")

    panel["is_train"] = (
        (panel["is_active"] == 1)
        & panel["train_start"].notna()
        & (panel["periode"] >= panel["train_start"])
        & (panel["periode"] <= train_end)
    ).astype(int)

    panel["is_test"] = (
        (panel["is_active"] == 1)
        & (panel["periode"] >= test_start)
        & (panel["periode"] <= test_end)
    ).astype(int)

    return panel


# =========================================
# 3) Merge exog dari external_data
# =========================================

def _merge_exog_from_db(panel: pd.DataFrame, engine) -> pd.DataFrame:
    """
    Versi DB dari merge events/holidays/rain_16:
    pakai tabel external_data:
      area, cabang, periode, event_flag, holiday_count, rainfall
    """
    panel = panel.copy()
    with engine.begin() as conn:
        exog = pd.read_sql(
            """
            SELECT periode,
                   cabang,
                   event_flag,
                   holiday_count,
                   rainfall
            FROM external_data
            """,
            conn,
            parse_dates=["periode"],
        )

    exog["event_flag"] = (
        pd.to_numeric(exog["event_flag"], errors="coerce")
          .fillna(0)
          .astype(int)
    )
    exog["holiday_count"] = (
        pd.to_numeric(exog["holiday_count"], errors="coerce")
          .fillna(0)
          .astype(int)
    )
    exog["rainfall"] = (
        pd.to_numeric(exog["rainfall"], errors="coerce")
          .fillna(0.0)
    )

    panel = (
        panel.merge(exog, on=["periode", "cabang"], how="left")
             .sort_values(["cabang", "sku", "periode"])
             .reset_index(drop=True)
    )

    panel["event_flag"] = panel["event_flag"].fillna(0).astype(int)
    panel["holiday_count"] = panel["holiday_count"].fillna(0).astype(int)
    panel["rainfall"] = panel["rainfall"].fillna(0.0)

    return panel


# =========================================
# 4) Exog lag + qty lag & rolling
# =========================================

def _add_exog_qty_lags_and_rollings(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Copy dari offline:
    - event_flag_lag1, holiday_count_lag1, rainfall_lag1
    - qty_lag1..12
    - qty_rollmean_3/6/12 + qty_rollstd_3/6/12 (shift(1) rolling)
    """
    panel = panel.sort_values(["cabang", "sku", "periode"]).copy()

    for c in ["event_flag", "holiday_count", "rainfall"]:
        lagc = f"{c}_lag1"
        if lagc in panel.columns:
            panel.drop(columns=[lagc], inplace=True)
        panel[lagc] = panel.groupby(["cabang", "sku"], sort=False)[c].shift(1)

    # rainfall hanya untuk 16C
    panel.loc[panel["cabang"] != "16C", "rainfall_lag1"] = 0.0

    MAX_LAG = 12
    for k in range(1, MAX_LAG + 1):
        panel[f"qty_lag{k}"] = panel.groupby(["cabang", "sku"])["qty"].shift(k)

    for w in [3, 6, 12]:
        panel[f"qty_rollmean_{w}"] = (
            panel.groupby(["cabang", "sku"])["qty"]
                  .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        panel[f"qty_rollstd_{w}"] = (
            panel.groupby(["cabang", "sku"])["qty"]
                  .transform(lambda s: s.shift(1).rolling(w, min_periods=2).std(ddof=1))
        )

    return panel


# =========================================
# 5) Spike detection (sama offline)
# =========================================

def detect_spikes_contextual(df,
                             z_th=3.0,
                             iqr_k=1.5,
                             roll_k=3,
                             dev_k=3.0,
                             event_quant=0.90):
    outs = []
    for (cab, sku), g in df.groupby(["cabang", "sku"], sort=False):
        g = g.sort_values("periode").copy()
        x = g["qty"].astype(float).values
        event = g["event_flag"].fillna(0).values
        holiday = g["holiday_count"].fillna(0).values
        rainfall = g["rainfall"].fillna(0.0).values if cab == "16C" else np.zeros_like(x)

        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        mad = mad if mad > 0 else 1.0
        zrob = 0.6745 * (x - med) / mad
        out_z = (np.abs(zrob) >= z_th)

        q1, q3 = np.nanpercentile(x, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        out_iqr = (x < lo) | (x > hi)

        xs = pd.Series(x)
        rm = xs.shift(1).rolling(roll_k, min_periods=1).median().values
        dev = (xs.sub(pd.Series(rm)).abs()
               .shift(1).rolling(roll_k, min_periods=1).median().values)
        fallback = np.nanmedian(dev[dev > 0]) if np.any(dev > 0) else 1.0
        dev = np.where((dev == 0) | np.isnan(dev), fallback, dev)
        out_roll = (np.abs(x - rm) > dev_k * dev)

        spike_raw = ((out_z.astype(int) + out_iqr.astype(int) + out_roll.astype(int)) >= 2)

        is_ctx = (event > 0) | (holiday > 0) | ((cab == "16C") & (rainfall > 50))
        adjusted = np.zeros_like(spike_raw, dtype=bool)
        ev_ref = x[is_ctx]
        ev_q = np.nanquantile(ev_ref, event_quant) if len(ev_ref) else np.nan

        for i, flag in enumerate(spike_raw):
            if not flag:
                continue
            if not is_ctx[i]:
                adjusted[i] = True
            else:
                if np.isnan(ev_q) or (x[i] > ev_q):
                    adjusted[i] = True

        h = g.copy()
        h["spike_flag"] = adjusted.astype(int)
        outs.append(h)
    return pd.concat(outs, ignore_index=True)


# =========================================
# 6) Selection window (persis offline)
# =========================================

def build_selection_window(df: pd.DataFrame,
                           win_start: pd.Timestamp,
                           train_end: pd.Timestamp) -> pd.DataFrame:
    w = df.query("periode >= @win_start and periode <= @train_end").copy()

    has_may = (
        w.query("periode == @train_end")
         .groupby(["cabang", "sku"], as_index=False)
         .size()
         .rename(columns={"size": "has_may24"})
    )

    agg = (
        w.groupby(["cabang", "sku"], as_index=False)
         .agg(
             n_months=("qty", "size"),
             nonzero_months=("qty", lambda s: (s > 0).sum()),
             total_qty=("qty", "sum"),
         )
    )

    last12 = (
        w.sort_values(["cabang", "sku", "periode"])
         .groupby(["cabang", "sku"], as_index=False)
         .agg(qty_12m=("qty", lambda s: s.tail(12).sum()))
    )

    last6 = (
        w.sort_values(["cabang", "sku", "periode"])
         .groupby(["cabang", "sku"], as_index=False)
         .agg(qty_6m=("qty", lambda s: s.tail(6).sum()))
    )

    t = df.loc[df["is_train"] == 1].copy()

    zr = (
        t.groupby(["cabang", "sku"], as_index=False)["qty"]
         .apply(lambda s: (s == 0).mean())
         .rename(columns={"qty": "zero_ratio_train"})
    )

    ntr = (
        t.groupby(["cabang", "sku"], as_index=False)["qty"]
         .count()
         .rename(columns={"qty": "n_train"})
    )

    nz = (
        t.loc[t["qty"] > 0]
         .groupby(["cabang", "sku"], as_index=False)["periode"]
         .max()
         .rename(columns={"periode": "last_nz"})
    )

    info = (
        agg.merge(last12, on=["cabang", "sku"], how="left")
           .merge(last6, on=["cabang", "sku"], how="left")
           .merge(has_may, on=["cabang", "sku"], how="left")
           .merge(zr, on=["cabang", "sku"], how="left")
           .merge(ntr, on=["cabang", "sku"], how="left")
           .merge(nz, on=["cabang", "sku"], how="left")
    )

    info["has_may24"] = info["has_may24"].fillna(0).gt(0)
    for c in ["zero_ratio_train", "n_train", "qty_12m", "qty_6m"]:
        if c in info.columns:
            info[c] = info[c].fillna(0)

    info["last_nz"] = pd.to_datetime(info["last_nz"], errors="coerce")
    train_end_per = train_end.to_period("M")

    mask = info["last_nz"].notna()
    info["months_since_last_nz"] = 999

    if mask.any():
        last_nz_per = info.loc[mask, "last_nz"].dt.to_period("M")
        diff_months = train_end_per.ordinal - last_nz_per.astype("int64")
        info.loc[mask, "months_since_last_nz"] = diff_months.values

    info["months_since_last_nz"] = info["months_since_last_nz"].astype("int64")

    info["alive_recent"] = (
        (info["qty_6m"] > 0)
        & (info["months_since_last_nz"] <= 3)
    ).astype(int)

    info["eligible_model"] = (
        (info["n_months"] >= 36)
        & (info["nonzero_months"] >= 10)
        & (info["qty_12m"] > 0)
        & (info["total_qty"] >= 30)
        & (info["zero_ratio_train"] <= 0.7)
        & (info["has_may24"] == True)
        & (info["alive_recent"] == 1)
    ).astype(int)

    return info


# =========================================
# 7) Fullfeat builder (hasil akhir LGBM)
# =========================================

def build_lgbm_full_fullfeat_from_db(
    engine,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    win_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Output DF HARUS 1:1 dengan lgbm_dataset_full_fullfeat.csv offline:
    - sudah hanya SKU eligible_model == 1
    - kolom:
        area, cabang, sku, periode, qty,
        event_flag, event_flag_lag1,
        holiday_count, holiday_count_lag1,
        rainfall_lag1,
        is_train, is_test,
        imputed, spike_flag, sample_weight,
        month, year, qtr,
        qty_lag1..12,
        qty_rollmean_3/6/12,
        qty_rollstd_3/6/12
    """
    if win_start is None:
        win_start = WIN_START_DEFAULT

    # 1) panel dasar dari DB
    panel = _make_windowed_panel_from_db(engine)

    # 2) train/test flags
    panel = _add_train_test_flags(panel, train_end, test_start, test_end)

    # 3) exog dari external_data
    panel = _merge_exog_from_db(panel, engine)

    # 4) exog lag + qty lag & rolling
    panel = _add_exog_qty_lags_and_rollings(panel)

    # 5) spike + sample_weight (copy offline)
    panel = detect_spikes_contextual(panel)
    panel["sample_weight"] = 1.0
    panel.loc[(panel["is_train"] == 1) & (panel["spike_flag"] == 1), "sample_weight"] = 0.8
    panel.loc[(panel["is_train"] == 1) & (panel["qty"] == 0), "sample_weight"] *= 0.7

    # 6) kolom waktu
    panel["month"] = panel["periode"].dt.month
    panel["year"] = panel["periode"].dt.year
    panel["qtr"] = panel["periode"].dt.quarter

    # 7) selection window + eligible_model
    info = build_selection_window(panel, win_start, train_end)

    # merge eligible ke panel
    for c in ["eligible_model", "alive_recent"]:
        if c in panel.columns:
            panel.drop(columns=[c], inplace=True)

    panel = panel.merge(
        info[["cabang", "sku", "eligible_model", "alive_recent"]],
        on=["cabang", "sku"],
        how="left",
    )

    panel["eligible_model"] = panel["eligible_model"].fillna(0).astype(int)
    panel["alive_recent"] = panel["alive_recent"].fillna(0).astype(int)

    # filter hanya SKU eligible_model == 1
    panel_full = (
        panel.query("eligible_model == 1")
             .sort_values(["cabang", "sku", "periode"])
             .reset_index(drop=True)
    )

    # 8) Fullfeat (sama dengan build_fullfeat_from_panel offline)
    df = panel_full.copy()

    if "rainfall" in df.columns:
        df = df.drop(columns=["rainfall"])

    qty_lags = [c for c in df.columns if c.startswith("qty_lag")]
    qty_rolls = [
        c for c in df.columns
        if c.startswith("qty_rollmean_") or c.startswith("qty_rollstd_")
    ]

    base_cols = [
        "area", "cabang", "sku", "periode", "qty",
        "event_flag", "event_flag_lag1",
        "holiday_count", "holiday_count_lag1",
        "rainfall_lag1",
        "is_train", "is_test",
        "imputed", "spike_flag", "sample_weight",
        "month", "year", "qtr",
    ]

    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom BASE hilang di panel_full DB: {missing}")

    full_cols = base_cols + qty_lags + qty_rolls
    full_df = df[full_cols].copy()

    return full_df
