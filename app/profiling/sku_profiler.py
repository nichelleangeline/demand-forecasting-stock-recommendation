from __future__ import annotations
import pandas as pd
import numpy as np
from sqlalchemy import text
from app.db import engine


def _to_month_start(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _norm_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cabang" in df.columns:
        df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    if "sku" in df.columns:
        df["sku"] = df["sku"].astype(str).str.strip().str.upper()
    if "area" in df.columns:
        df["area"] = df["area"].astype(str).str.strip()
    return df


def _get_min_sales_periode_month_start() -> pd.Timestamp | None:
    sql = text("SELECT MIN(periode) AS minp FROM sales_monthly")
    with engine.begin() as conn:
        r = conn.execute(sql).mappings().fetchone()
    if not r or r.get("minp") is None:
        return None
    return _to_month_start(pd.Timestamp(r["minp"]))


def build_sku_profile(df: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cabang", "sku",
                "n_months", "qty_mean", "qty_std", "qty_max", "qty_min", "total_qty",
                "zero_months", "zero_ratio", "cv", "demand_level",
                "train_start", "last_nz", "nonzero_months",
            ]
        )

    df = _norm_key(df)
    if "periode" not in df.columns or "qty" not in df.columns:
        raise ValueError("build_sku_profile: kolom wajib: 'periode' dan 'qty'.")

    train_end = _to_month_start(train_end)

    df["periode"] = pd.to_datetime(df["periode"], errors="coerce").map(_to_month_start)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0).astype(float)
    df = df.dropna(subset=["cabang", "sku", "periode"])
    df = df.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)

    df_pos = df[df["qty"] > 0].copy()
    if df_pos.empty:
        return pd.DataFrame(
            columns=[
                "cabang", "sku",
                "n_months", "qty_mean", "qty_std", "qty_max", "qty_min", "total_qty",
                "zero_months", "zero_ratio", "cv", "demand_level",
                "train_start", "last_nz", "nonzero_months",
            ]
        )

    gpos = df_pos.groupby(["cabang", "sku"], as_index=False)

    span = gpos.agg(
        train_start=("periode", "min"),
        last_nz=("periode", "max"),
        total_qty=("qty", "sum"),
        nonzero_months=("periode", "nunique"),
        qty_std=("qty", "std"),
        qty_max=("qty", "max"),
        qty_min=("qty", "min"),
    )

    def _calc_n_months(a: pd.Timestamp, b: pd.Timestamp) -> int:
        return int((b.year - a.year) * 12 + (b.month - a.month) + 1)

    span["n_months"] = [
        _calc_n_months(a, b) if pd.notna(a) and pd.notna(b) else 0
        for a, b in zip(span["train_start"], span["last_nz"])
    ]
    span["n_months"] = pd.to_numeric(span["n_months"], errors="coerce").fillna(0).astype(int)

    span["nonzero_months"] = pd.to_numeric(span["nonzero_months"], errors="coerce").fillna(0).astype(int)
    span["zero_months"] = (span["n_months"] - span["nonzero_months"]).clip(lower=0).astype(int)

    span["zero_ratio"] = np.where(span["n_months"] > 0, span["zero_months"] / span["n_months"], 1.0)
    span["zero_ratio"] = pd.to_numeric(span["zero_ratio"], errors="coerce").fillna(1.0).astype(float)

    span["qty_mean"] = np.where(span["n_months"] > 0, span["total_qty"] / span["n_months"], 0.0)

    span["qty_std"] = pd.to_numeric(span["qty_std"], errors="coerce").fillna(0.0).astype(float)
    span["qty_max"] = pd.to_numeric(span["qty_max"], errors="coerce").fillna(0.0).astype(float)
    span["qty_min"] = pd.to_numeric(span["qty_min"], errors="coerce").fillna(0.0).astype(float)

    span["cv"] = np.where(span["qty_mean"] > 1e-9, span["qty_std"] / span["qty_mean"], 0.0)
    span["cv"] = pd.to_numeric(span["cv"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    try:
        span["demand_level"] = pd.qcut(
            span["qty_mean"],
            q=4,
            labels=[0, 1, 2, 3],
            duplicates="drop",
        ).astype(int)
        if span["demand_level"].isna().all():
            span["demand_level"] = 0
    except Exception:
        span["demand_level"] = 0

    out_cols = [
        "cabang", "sku",
        "n_months", "qty_mean", "qty_std", "qty_max", "qty_min", "total_qty",
        "zero_months", "zero_ratio", "cv", "demand_level",
        "train_start", "last_nz", "nonzero_months",
    ]
    return span[out_cols].copy()


def build_and_store_sku_profile(
    train_end: pd.Timestamp,
    win_start: pd.Timestamp | None = None,
) -> int:
    train_end = _to_month_start(train_end)

    if win_start is None:
        db_min = _get_min_sales_periode_month_start()
        if db_min is None:
            return 0
        win_start = db_min
    else:
        win_start = _to_month_start(win_start)

    sql = text(
        """
        WITH base AS (
            SELECT
                TRIM(area) AS area,
                UPPER(TRIM(cabang)) AS cabang,
                UPPER(TRIM(sku))    AS sku,
                DATE_FORMAT(periode, '%Y-%m-01') AS periode_ms,
                SUM(qty) AS qty
            FROM sales_monthly
            WHERE periode >= :win_start
              AND periode <= :train_end
            GROUP BY 1,2,3,4
        ),
        pos AS (
            SELECT * FROM base WHERE qty > 0
        ),
        span AS (
            SELECT
                cabang,
                sku,
                MIN(periode_ms) AS train_start,
                MAX(periode_ms) AS last_nz,
                SUM(qty) AS total_qty,
                COUNT(DISTINCT periode_ms) AS nonzero_months,
                STDDEV_POP(qty) AS qty_std_pos,
                MAX(qty) AS qty_max_pos,
                MIN(qty) AS qty_min_pos
            FROM pos
            GROUP BY cabang, sku
        ),
        last_flags AS (
            SELECT
                cabang,
                sku,
                MAX(CASE WHEN periode_ms = DATE_FORMAT(:train_end, '%Y-%m-01') AND qty > 0 THEN 1 ELSE 0 END) AS has_last,
                SUM(CASE WHEN periode_ms >= DATE_FORMAT(DATE_SUB(:train_end, INTERVAL 11 MONTH), '%Y-%m-01') AND qty > 0 THEN qty ELSE 0 END) AS qty_12m,
                SUM(CASE WHEN periode_ms >= DATE_FORMAT(DATE_SUB(:train_end, INTERVAL 5 MONTH),  '%Y-%m-01') AND qty > 0 THEN qty ELSE 0 END) AS qty_6m
            FROM base
            GROUP BY cabang, sku
        )
        SELECT
            s.cabang,
            s.sku,
            s.train_start,
            s.last_nz,
            (TIMESTAMPDIFF(MONTH, s.train_start, s.last_nz) + 1) AS n_months,
            s.nonzero_months,
            s.total_qty,
            (TIMESTAMPDIFF(MONTH, s.last_nz, DATE_FORMAT(:train_end, '%Y-%m-01'))) AS months_since_last_nz,
            COALESCE(lf.has_last, 0) AS has_last,
            COALESCE(lf.qty_12m, 0) AS qty_12m,
            COALESCE(lf.qty_6m, 0)  AS qty_6m,
            s.qty_std_pos,
            s.qty_max_pos,
            s.qty_min_pos
        FROM span s
        LEFT JOIN last_flags lf
          ON lf.cabang = s.cabang AND lf.sku = s.sku
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(sql, {"win_start": win_start, "train_end": train_end}).mappings().all()

    if not rows:
        return 0

    prof = pd.DataFrame(rows)
    prof = _norm_key(prof)

    for c in ["n_months", "nonzero_months", "total_qty", "months_since_last_nz", "has_last", "qty_12m", "qty_6m"]:
        if c in prof.columns:
            prof[c] = pd.to_numeric(prof[c], errors="coerce").fillna(0)

    prof["n_months"] = prof["n_months"].astype(int)
    prof["nonzero_months"] = prof["nonzero_months"].astype(int)
    prof["months_since_last_nz"] = prof["months_since_last_nz"].astype(int)
    prof["has_last"] = prof["has_last"].astype(int)

    prof["zero_months"] = (prof["n_months"] - prof["nonzero_months"]).clip(lower=0).astype(int)
    prof["zero_ratio"] = np.where(prof["n_months"] > 0, prof["zero_months"] / prof["n_months"], 1.0)
    prof["zero_ratio"] = pd.to_numeric(prof["zero_ratio"], errors="coerce").fillna(1.0).astype(float)
    prof["zero_ratio_train"] = prof["zero_ratio"].astype(float)

    prof["qty_mean"] = np.where(prof["n_months"] > 0, prof["total_qty"] / prof["n_months"], 0.0)

    prof["qty_std"] = pd.to_numeric(prof.get("qty_std_pos", 0), errors="coerce").fillna(0.0).astype(float)
    prof["qty_max"] = pd.to_numeric(prof.get("qty_max_pos", 0), errors="coerce").fillna(0.0).astype(float)
    prof["qty_min"] = pd.to_numeric(prof.get("qty_min_pos", 0), errors="coerce").fillna(0.0).astype(float)

    prof["cv"] = np.where(prof["qty_mean"] > 1e-9, prof["qty_std"] / prof["qty_mean"], 0.0)
    prof["cv"] = pd.to_numeric(prof["cv"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    try:
        prof["demand_level"] = pd.qcut(
            prof["qty_mean"],
            q=4,
            labels=[0, 1, 2, 3],
            duplicates="drop",
        ).astype(int)
        if prof["demand_level"].isna().all():
            prof["demand_level"] = 0
    except Exception:
        prof["demand_level"] = 0

    prof["cluster"] = -1
    prof["eligible_model"] = 0
    prof["n_train"] = prof["n_months"].astype(int)

    prof["train_start"] = pd.to_datetime(prof["train_start"], errors="coerce")
    prof["last_nz"] = pd.to_datetime(prof["last_nz"], errors="coerce")

    upsert_sql = text(
        """
        INSERT INTO sku_profile (
            cabang, sku,
            n_months, qty_mean, qty_std, qty_max, qty_min, total_qty,
            zero_months, zero_ratio, zero_ratio_train,
            cv, demand_level, cluster, eligible_model,
            nonzero_months, qty_6m, qty_12m, has_last,
            train_start, last_nz, months_since_last_nz, n_train,
            last_updated
        )
        VALUES (
            :cabang, :sku,
            :n_months, :qty_mean, :qty_std, :qty_max, :qty_min, :total_qty,
            :zero_months, :zero_ratio, :zero_ratio_train,
            :cv, :demand_level, :cluster, :eligible_model,
            :nonzero_months, :qty_6m, :qty_12m, :has_last,
            :train_start, :last_nz, :months_since_last_nz, :n_train,
            CURRENT_TIMESTAMP
        )
        ON DUPLICATE KEY UPDATE
            n_months             = VALUES(n_months),
            qty_mean             = VALUES(qty_mean),
            qty_std              = VALUES(qty_std),
            qty_max              = VALUES(qty_max),
            qty_min              = VALUES(qty_min),
            total_qty            = VALUES(total_qty),
            zero_months          = VALUES(zero_months),
            zero_ratio           = VALUES(zero_ratio),
            zero_ratio_train     = VALUES(zero_ratio_train),
            cv                   = VALUES(cv),
            demand_level         = VALUES(demand_level),
            cluster              = VALUES(cluster),
            eligible_model       = VALUES(eligible_model),
            nonzero_months       = VALUES(nonzero_months),
            qty_6m               = VALUES(qty_6m),
            qty_12m              = VALUES(qty_12m),
            has_last             = VALUES(has_last),
            train_start          = VALUES(train_start),
            last_nz              = VALUES(last_nz),
            months_since_last_nz = VALUES(months_since_last_nz),
            n_train              = VALUES(n_train),
            last_updated         = CURRENT_TIMESTAMP
        """
    )

    payload = []
    for _, r in prof.iterrows():
        payload.append(
            {
                "cabang": r["cabang"],
                "sku": r["sku"],
                "n_months": int(r["n_months"]),
                "qty_mean": float(r["qty_mean"]),
                "qty_std": float(r["qty_std"]),
                "qty_max": float(r["qty_max"]),
                "qty_min": float(r["qty_min"]),
                "total_qty": float(r["total_qty"]),
                "zero_months": int(r["zero_months"]),
                "zero_ratio": float(r["zero_ratio"]),
                "zero_ratio_train": float(r["zero_ratio_train"]),
                "cv": float(r["cv"]),
                "demand_level": int(r["demand_level"]) if not pd.isna(r["demand_level"]) else 0,
                "cluster": int(r["cluster"]),
                "eligible_model": int(r["eligible_model"]),
                "nonzero_months": int(r["nonzero_months"]),
                "qty_6m": float(r["qty_6m"]),
                "qty_12m": float(r["qty_12m"]),
                "has_last": int(r["has_last"]),
                "train_start": r["train_start"].date() if pd.notna(r["train_start"]) else None,
                "last_nz": r["last_nz"].date() if pd.notna(r["last_nz"]) else None,
                "months_since_last_nz": int(r["months_since_last_nz"]),
                "n_train": int(r["n_train"]),
            }
        )

    with engine.begin() as conn:
        conn.execute(upsert_sql, payload)

    return int(len(prof))
