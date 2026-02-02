import numpy as np
import pandas as pd
from sqlalchemy import text

WIN_START_DEFAULT = pd.Timestamp("2021-01-01")


def _to_month_start(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _norm_keys_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cabang" in df.columns:
        df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    if "sku" in df.columns:
        df["sku"] = df["sku"].astype(str).str.strip().str.upper()
    if "area" in df.columns:
        df["area"] = df["area"].astype(str).str.strip()
    return df


def _make_windowed_panel_from_db(
    engine,
    win_start: pd.Timestamp,
    panel_end: pd.Timestamp,
) -> pd.DataFrame:
    # ini ambil data sales_monthly dari win_start sampai panel_end
    # panel_end ini harus mencakup TEST_END juga, supaya baris test beneran ada di panel
    # lalu dibikin kalender bulanan untuk tiap (cabang, sku)
    # supaya bulan yang kosong jadi 0 (imputed=1), jadi hitungan fitur/flag stabil

    win_start = _to_month_start(win_start)
    panel_end = _to_month_start(panel_end)

    sql = text("""
        SELECT periode, area, cabang, sku, qty
        FROM sales_monthly
        WHERE periode >= :win_start
          AND periode <= :panel_end
    """)

    with engine.begin() as conn:
        sales = pd.read_sql(
            sql,
            conn,
            params={"win_start": win_start, "panel_end": panel_end},
            parse_dates=["periode"],
        )

    if sales.empty:
        return pd.DataFrame(columns=["periode", "area", "cabang", "sku", "qty", "imputed", "is_active"])

    sales["periode"] = pd.to_datetime(sales["periode"], errors="coerce")
    sales["qty"] = pd.to_numeric(sales["qty"], errors="coerce").fillna(0.0)
    sales = _norm_keys_df(sales)

    out = []
    cal = pd.date_range(win_start, panel_end, freq="MS")

    for (cab, sku), g in sales.groupby(["cabang", "sku"], sort=False):
        g = g.sort_values("periode")
        if g["periode"].isna().all():
            continue

        idx = pd.MultiIndex.from_product(
            [[cab], [sku], cal],
            names=["cabang", "sku", "periode"],
        )

        g2 = (
            g.set_index(["cabang", "sku", "periode"])
             .reindex(idx)
             .reset_index()
        )

        exist = set(zip(g["cabang"], g["sku"], g["periode"]))
        g2["qty"] = pd.to_numeric(g2["qty"], errors="coerce").fillna(0.0)
        g2["imputed"] = [0 if (cab, sku, p) in exist else 1 for p in g2["periode"]]

        # area dipakai dari nilai pertama yang ada 
        if "area" in g.columns and len(g):
            area_val = g["area"].dropna()
            g2["area"] = area_val.iloc[0] if len(area_val) else None
        else:
            g2["area"] = None

        # is_active = 1 dari nonzero pertama sampai nonzero terakhir 
        nz_idx = g2.index[g2["qty"] != 0]
        if len(nz_idx):
            left, right = int(nz_idx.min()), int(nz_idx.max())
            g2["is_active"] = 0
            g2.loc[left:right, "is_active"] = 1
        else:
            g2["is_active"] = 0

        out.append(g2[["periode", "area", "cabang", "sku", "qty", "imputed", "is_active"]])

    if not out:
        return pd.DataFrame(columns=["periode", "area", "cabang", "sku", "qty", "imputed", "is_active"])

    panel = (
        pd.concat(out, ignore_index=True)
          .sort_values(["cabang", "sku", "periode"])
          .reset_index(drop=True)
    )
    return panel


def _add_train_test_flags(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    # ini bikin flag baris train dan test berdasarkan periode dan is_active
    # penting: panel harus sudah punya baris sampai test_end, kalau engga is_test pasti kosong semua

    panel = panel.copy()

    train_end = _to_month_start(train_end)
    test_start = _to_month_start(test_start)
    test_end = _to_month_start(test_end)

    def _nonzero_bounds(g):
        nz = g.loc[g["qty"] != 0, "periode"]
        if len(nz):
            return pd.Series({"train_start": nz.min(), "nz_last": nz.max()})
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


def _merge_exog_from_db(panel: pd.DataFrame, engine) -> pd.DataFrame:
    # ini gabung external_data ke panel (event, holiday, rainfall)
    panel = panel.copy()

    with engine.begin() as conn:
        exog = pd.read_sql(
            text("""
                SELECT periode, cabang, event_flag, holiday_count, rainfall
                FROM external_data
            """),
            conn,
            parse_dates=["periode"],
        )

    if exog.empty:
        panel["event_flag"] = 0
        panel["holiday_count"] = 0
        panel["rainfall"] = 0.0
        panel["event_flag_lag1"] = 0
        panel["holiday_count_lag1"] = 0
        panel["rainfall_lag1"] = 0.0
        return panel

    exog["periode"] = pd.to_datetime(exog["periode"], errors="coerce")
    exog["event_flag"] = pd.to_numeric(exog["event_flag"], errors="coerce").fillna(0).astype(int)
    exog["holiday_count"] = pd.to_numeric(exog["holiday_count"], errors="coerce").fillna(0).astype(int)
    exog["rainfall"] = pd.to_numeric(exog["rainfall"], errors="coerce").fillna(0.0)

    exog = _norm_keys_df(exog)

    panel = (
        panel.merge(exog, on=["periode", "cabang"], how="left")
             .sort_values(["cabang", "sku", "periode"])
             .reset_index(drop=True)
    )

    panel["event_flag"] = panel["event_flag"].fillna(0).astype(int)
    panel["holiday_count"] = panel["holiday_count"].fillna(0).astype(int)
    panel["rainfall"] = panel["rainfall"].fillna(0.0)

    return panel


def _add_exog_qty_lags_and_rollings(panel: pd.DataFrame) -> pd.DataFrame:
    # ini bikin lag exog (1 bulan) dan lag qty (1..12) + rolling mean/std
    panel = panel.sort_values(["cabang", "sku", "periode"]).copy()

    for c in ["event_flag", "holiday_count", "rainfall"]:
        lagc = f"{c}_lag1"
        if lagc in panel.columns:
            panel.drop(columns=[lagc], inplace=True)
        panel[lagc] = panel.groupby(["cabang", "sku"], sort=False)[c].shift(1)

    # rainfall cuma relevan untuk cabang 16C, cabang lain jadi 0
    if "rainfall_lag1" in panel.columns:
        panel.loc[panel["cabang"] != "16C", "rainfall_lag1"] = 0.0

    for k in range(1, 13):
        panel[f"qty_lag{k}"] = panel.groupby(["cabang", "sku"], sort=False)["qty"].shift(k)

    for w in [3, 6, 12]:
        panel[f"qty_rollmean_{w}"] = (
            panel.groupby(["cabang", "sku"], sort=False)["qty"]
                 .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        panel[f"qty_rollstd_{w}"] = (
            panel.groupby(["cabang", "sku"], sort=False)["qty"]
                 .transform(lambda s: s.shift(1).rolling(w, min_periods=2).std(ddof=1))
        )

    return panel


def detect_spikes_contextual(
    df: pd.DataFrame,
    z_th=3.0,
    iqr_k=1.5,
    roll_k=3,
    dev_k=3.0,
    event_quant=0.90,
) -> pd.DataFrame:
    # ini nandain spike (lonjakan) qty, tapi dikasih konteks event/holiday/rainfall
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
        dev = (
            xs.sub(pd.Series(rm)).abs()
              .shift(1).rolling(roll_k, min_periods=1).median().values
        )
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

    if not outs:
        df2 = df.copy()
        df2["spike_flag"] = 0
        return df2

    return pd.concat(outs, ignore_index=True)


def build_selection_window(
    df: pd.DataFrame,
    win_start: pd.Timestamp,
    train_end: pd.Timestamp,
    min_n_months: int = 36,
    min_nonzero: int = 10,
    min_total_qty: float = 30.0,
    max_zero_ratio: float = 0.7,
    alive_recent_months: int = 3,
    require_last_month: bool = True,
    require_qty_12m_gt0: bool = True,
) -> pd.DataFrame:
    # ini ngitung ringkasan per (cabang, sku) untuk nentuin eligible_model
    # tetap pakai batas train_end, walau panel-nya mungkin sampai test_end
    train_end = _to_month_start(train_end)
    win_start = _to_month_start(win_start)

    w = df.query("periode >= @win_start and periode <= @train_end").copy()

    has_last = (
        w.query("periode == @train_end")
         .groupby(["cabang", "sku"], as_index=False)
         .size()
         .rename(columns={"size": "has_last"})
    )

    agg = (
        w.groupby(["cabang", "sku"], as_index=False)
         .agg(
             n_months=("qty", "size"),
             nonzero_months=("qty", lambda s: int((s > 0).sum())),
             total_qty=("qty", "sum"),
         )
    )

    last12 = (
        w.sort_values(["cabang", "sku", "periode"])
         .groupby(["cabang", "sku"], as_index=False)
         .agg(qty_12m=("qty", lambda s: float(s.tail(12).sum())))
    )

    last6 = (
        w.sort_values(["cabang", "sku", "periode"])
         .groupby(["cabang", "sku"], as_index=False)
         .agg(qty_6m=("qty", lambda s: float(s.tail(6).sum())))
    )

    t = df.loc[df["is_train"] == 1].copy()

    zr = (
        t.groupby(["cabang", "sku"], as_index=False)
         .agg(zero_ratio_train=("qty", lambda s: float((s == 0).mean())))
    )

    ntr = (
        t.groupby(["cabang", "sku"], as_index=False)
         .agg(n_train=("qty", "count"))
    )

    nz = (
        t.loc[t["qty"] > 0]
         .groupby(["cabang", "sku"], as_index=False)
         .agg(last_nz=("periode", "max"))
    )

    info = (
        agg.merge(last12, on=["cabang", "sku"], how="left")
           .merge(last6, on=["cabang", "sku"], how="left")
           .merge(has_last, on=["cabang", "sku"], how="left")
           .merge(zr, on=["cabang", "sku"], how="left")
           .merge(ntr, on=["cabang", "sku"], how="left")
           .merge(nz, on=["cabang", "sku"], how="left")
    )

    info["has_last"] = info["has_last"].fillna(0).gt(0)

    for c in ["zero_ratio_train", "n_train", "qty_12m", "qty_6m"]:
        info[c] = pd.to_numeric(info[c], errors="coerce").fillna(0)

    info["last_nz"] = pd.to_datetime(info["last_nz"], errors="coerce")

    info["months_since_last_nz"] = 999
    mask = info["last_nz"].notna()
    if mask.any():
        last = info.loc[mask, "last_nz"]
        diff_months = (train_end.year - last.dt.year) * 12 + (train_end.month - last.dt.month)
        info.loc[mask, "months_since_last_nz"] = diff_months.astype(int).values

    info["months_since_last_nz"] = pd.to_numeric(info["months_since_last_nz"], errors="coerce").fillna(999).astype(int)

    info["alive_recent"] = (
        (info["qty_6m"] > 0)
        & (info["months_since_last_nz"] <= int(alive_recent_months))
    ).astype(int)

    rule = (
        (info["n_months"] >= int(min_n_months))
        & (info["nonzero_months"] >= int(min_nonzero))
        & (info["total_qty"] >= float(min_total_qty))
        & (info["zero_ratio_train"] <= float(max_zero_ratio))
        & (info["alive_recent"] == 1)
    )

    if require_qty_12m_gt0:
        rule &= (info["qty_12m"] > 0)

    if require_last_month:
        rule &= (info["has_last"] == True)

    info["eligible_model"] = rule.astype(int)
    return info


def build_base_panel_from_db(
    engine,
    win_start: pd.Timestamp,
    panel_end: pd.Timestamp,
) -> pd.DataFrame:
    # ini wrapper buat bikin panel dasar sampai panel_end
    return _make_windowed_panel_from_db(engine, win_start=win_start, panel_end=panel_end)


def build_full_panel_features_from_db(
    engine,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    win_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    # ini bikin panel + train/test flag + exog + lag/rolling
    # kunci perbaikannya: panel dibangun sampai max(train_end, test_end)
    if win_start is None:
        win_start = WIN_START_DEFAULT

    train_end = _to_month_start(train_end)
    test_start = _to_month_start(test_start)
    test_end = _to_month_start(test_end)
    win_start = _to_month_start(win_start)

    panel_end = max(train_end, test_end)

    panel = build_base_panel_from_db(engine, win_start=win_start, panel_end=panel_end)
    if panel.empty:
        return panel

    panel = _add_train_test_flags(panel, train_end, test_start, test_end)
    panel = _merge_exog_from_db(panel, engine)
    panel = _add_exog_qty_lags_and_rollings(panel)
    return panel


def compute_selection_info_from_db(
    engine,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    win_start: pd.Timestamp | None = None,
    selection_rule: dict | None = None,
) -> pd.DataFrame:
    # ini hasilin tabel ringkasan eligibility per (cabang, sku)
    if win_start is None:
        win_start = WIN_START_DEFAULT
    win_start = _to_month_start(win_start)

    rule = selection_rule or {}

    panel = build_full_panel_features_from_db(
        engine,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        win_start=win_start,
    )
    if panel.empty:
        return pd.DataFrame()

    info = build_selection_window(panel, win_start=win_start, train_end=train_end, **rule)
    info = _norm_keys_df(info)
    return info


def apply_eligibility_to_sku_profile_from_db(
    engine,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    win_start: pd.Timestamp | None = None,
    selection_rule: dict | None = None,
) -> int:
    # ini apply hasil eligible_model ke tabel sku_profile berdasarkan hitungan panel
    info = compute_selection_info_from_db(
        engine,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        win_start=win_start,
        selection_rule=selection_rule,
    )

    if info.empty:
        return 0

    with engine.begin() as conn:
        prof = pd.read_sql(
            text("SELECT id, cabang, sku, eligible_model FROM sku_profile"),
            conn,
        )

        if prof.empty:
            return 0

        prof = _norm_keys_df(prof)

        merged = prof.merge(
            info[["cabang", "sku", "eligible_model"]],
            on=["cabang", "sku"],
            how="left",
            suffixes=("_old", "_new"),
        )

        merged["eligible_model_old"] = merged["eligible_model_old"].fillna(0).astype(int)
        merged["eligible_model_new"] = merged["eligible_model_new"].fillna(0).astype(int)

        changed = merged[merged["eligible_model_new"] != merged["eligible_model_old"]]
        if changed.empty:
            return 0

        upd_sql = text("""
            UPDATE sku_profile
            SET eligible_model = :eligible_model
            WHERE id = :id
        """)

        for _, r in changed.iterrows():
            conn.execute(
                upd_sql,
                {"id": int(r["id"]), "eligible_model": int(r["eligible_model_new"])}
            )

        return int(len(changed))


def build_lgbm_full_fullfeat_from_db(
    engine,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    win_start: pd.Timestamp | None = None,
    selection_rule: dict | None = None,
) -> pd.DataFrame:
    # ini bikin dataset final buat training LGBM (sudah ada fitur, eligible, dan weight)
    if win_start is None:
        win_start = WIN_START_DEFAULT

    train_end = _to_month_start(train_end)
    test_start = _to_month_start(test_start)
    test_end = _to_month_start(test_end)
    win_start = _to_month_start(win_start)

    rule = selection_rule or {}

    panel = build_full_panel_features_from_db(
        engine,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        win_start=win_start,
    )
    if panel.empty:
        return pd.DataFrame()

    panel = detect_spikes_contextual(panel)

    panel["sample_weight"] = 1.0
    panel.loc[(panel["is_train"] == 1) & (panel["spike_flag"] == 1), "sample_weight"] = 0.8
    panel.loc[(panel["is_train"] == 1) & (panel["qty"] == 0), "sample_weight"] *= 0.7

    panel["month"] = panel["periode"].dt.month
    panel["year"] = panel["periode"].dt.year
    panel["qtr"] = panel["periode"].dt.quarter

    info = build_selection_window(panel, win_start=win_start, train_end=train_end, **rule)

    for c in ["eligible_model", "alive_recent"]:
        if c in panel.columns:
            panel.drop(columns=[c], inplace=True)

    panel = panel.merge(
        info[["cabang", "sku", "eligible_model", "alive_recent"]],
        on=["cabang", "sku"],
        how="left",
    )
    panel["eligible_model"] = panel["eligible_model"].fillna(0).astype(int)

    panel_full = (
        panel.query("eligible_model == 1")
             .sort_values(["cabang", "sku", "periode"])
             .reset_index(drop=True)
    )

    df = panel_full.copy()

    if "rainfall" in df.columns:
        df = df.drop(columns=["rainfall"])

    qty_lags = [c for c in df.columns if c.startswith("qty_lag")]
    qty_rolls = [c for c in df.columns if c.startswith("qty_rollmean_") or c.startswith("qty_rollstd_")]

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
        raise ValueError(f"Kolom BASE hilang: {missing}")

    full_cols = base_cols + qty_lags + qty_rolls
    df = _norm_keys_df(df)
    return df[full_cols].copy()


def apply_eligibility_to_sku_profile_fast(
    engine,
    selection_rule: dict | None = None,
) -> int:
    rule = selection_rule or {}

    params = {
        "min_n_months": int(rule.get("min_n_months", 36)),
        "min_nonzero": int(rule.get("min_nonzero", 10)),
        "min_total_qty": float(rule.get("min_total_qty", 30.0)),
        "max_zero_ratio": float(rule.get("max_zero_ratio", 0.7)),
        "alive_recent_months": int(rule.get("alive_recent_months", 3)),
        "require_last_month": 1 if bool(rule.get("require_last_month", True)) else 0,
        "require_qty_12m_gt0": 1 if bool(rule.get("require_qty_12m_gt0", True)) else 0,
    }

    sql = text("""
        UPDATE sku_profile
        SET
            eligible_model =
                CASE
                    WHEN
                        (COALESCE(n_months, 0) >= :min_n_months)
                        AND (COALESCE(nonzero_months, 0) >= :min_nonzero)
                        AND (COALESCE(total_qty, 0) >= :min_total_qty)
                        AND (COALESCE(zero_ratio_train, 1) <= :max_zero_ratio)
                        AND (COALESCE(qty_6m, 0) > 0)
                        AND (COALESCE(months_since_last_nz, 999) <= :alive_recent_months)
                        AND (:require_last_month = 0 OR COALESCE(has_last, 0) = 1)
                        AND (:require_qty_12m_gt0 = 0 OR COALESCE(qty_12m, 0) > 0)
                    THEN 1
                    ELSE 0
                END,
            last_updated = CURRENT_TIMESTAMP
    """)

    with engine.begin() as conn:
        res = conn.execute(sql, params)
        return int(res.rowcount)
