import datetime as dt
from datetime import date
from pathlib import Path
import json
import re
import uuid

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

from app.db import engine
from app.ui.theme import inject_global_theme, render_sidebar_user_and_logout
from app.profiling.sku_profiler import build_and_store_sku_profile
from app.profiling.clustering import run_sku_clustering
from app.services.model_service import get_all_model_runs, activate_model
from app.features.hierarchy_features import add_hierarchy_features
from app.features.stabilizer_features import add_stabilizer_features
from app.features.outlier_handler import winsorize_outliers
from app.modeling.lgbm_trainer_cluster import train_lgbm_per_cluster
from app.services.panel_builder import (
    build_lgbm_full_fullfeat_from_db,
    apply_eligibility_to_sku_profile_fast,
)
from app.services.auth_guard import require_login

def _norm_keys_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cabang" in df.columns:
        df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    if "sku" in df.columns:
        df["sku"] = df["sku"].astype(str).str.strip().str.upper()
    if "area" in df.columns:
        df["area"] = df["area"].astype(str).str.strip()
    return df


def _to_month_start_date(d: dt.date) -> dt.date:
    return dt.date(d.year, d.month, 1)


def _to_month_start_any(x):
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        x = x.date()
    if isinstance(x, dt.date):
        return dt.date(x.year, x.month, 1)
    try:
        ts = pd.Timestamp(x)
        return dt.date(int(ts.year), int(ts.month), 1)
    except Exception:
        return None


def _to_month_start_ts_any(x):
    if x is None:
        return pd.NaT
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        return pd.Timestamp(year=int(ts.year), month=int(ts.month), day=1)
    except Exception:
        return pd.NaT


def _get_table_columns(table_name: str) -> set[str]:
    sql = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = :t
    """
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params={"t": table_name})
    if df.empty:
        return set()
    return set(df["COLUMN_NAME"].astype(str).tolist())


def _select_existing_cols(table: str, wanted: list[str]) -> list[str]:
    existing = _get_table_columns(table)
    return [c for c in wanted if c in existing]


def _month_index(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, errors="coerce")
    y = pd.Series(t.dt.year, index=t.index, dtype="Int64")
    m = pd.Series(t.dt.month, index=t.index, dtype="Int64")
    return (y * 12 + (m - 1)).astype("Int64")


def _months_between(a_ts: pd.Series, b_ts: pd.Series) -> pd.Series:
    a_idx = _month_index(a_ts)
    b_idx = _month_index(b_ts)
    return (b_idx - a_idx).astype("Int64")


def _build_profile_for_clustering_from_sku_profile(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = _norm_keys_df(df)

    if "qty_mean" not in df.columns:
        if ("total_qty" in df.columns) and ("n_months" in df.columns):
            n = pd.to_numeric(df["n_months"], errors="coerce").fillna(0).astype(float)
            t = pd.to_numeric(df["total_qty"], errors="coerce").fillna(0).astype(float)
            df["qty_mean"] = np.where(n > 0, t / n, 0.0)
        else:
            df["qty_mean"] = 0.0

    if "zero_ratio" not in df.columns:
        if "zero_ratio_train" in df.columns:
            df["zero_ratio"] = pd.to_numeric(df["zero_ratio_train"], errors="coerce").fillna(0.0).astype(float)
        else:
            df["zero_ratio"] = 0.0

    if "cv" not in df.columns:
        if ("qty_std" in df.columns) and ("qty_mean" in df.columns):
            stdv = pd.to_numeric(df["qty_std"], errors="coerce").fillna(0.0).astype(float)
            meanv = pd.to_numeric(df["qty_mean"], errors="coerce").fillna(0.0).astype(float)
            df["cv"] = np.where(meanv > 1e-9, stdv / meanv, 0.0)
        else:
            df["cv"] = 0.0

    out = df[["cabang", "sku", "qty_mean", "cv", "zero_ratio"]].copy()
    for c in ["qty_mean", "cv", "zero_ratio"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    return out


def cluster_and_update_from_sku_profile(engine, n_clusters: int = 4) -> int:
    wanted_base = ["id", "cabang", "sku", "total_qty", "n_months", "zero_ratio_train"]
    base_cols = _select_existing_cols("sku_profile", wanted_base)
    if not base_cols:
        return 0

    with engine.begin() as conn:
        df = pd.read_sql(
            text(f"SELECT {', '.join(base_cols)} FROM sku_profile ORDER BY cabang, sku"),
            conn,
        )

        optional_wanted = ["qty_mean", "qty_std", "cv", "zero_ratio"]
        optional_cols = _select_existing_cols("sku_profile", optional_wanted)
        if optional_cols:
            df_opt = pd.read_sql(
                text(f"SELECT cabang, sku, {', '.join(optional_cols)} FROM sku_profile"),
                conn,
            )
            df = df.merge(df_opt, on=["cabang", "sku"], how="left")

    if df.empty:
        return 0

    df = _norm_keys_df(df)
    prof_cluster = _build_profile_for_clustering_from_sku_profile(df)

    clustered = run_sku_clustering(prof_cluster, n_clusters=int(n_clusters))
    clustered = _norm_keys_df(clustered)

    if "cluster" not in clustered.columns:
        return 0

    m = df[["id", "cabang", "sku"]].merge(
        clustered[["cabang", "sku", "cluster"]],
        on=["cabang", "sku"],
        how="left",
    )
    m["cluster"] = pd.to_numeric(m["cluster"], errors="coerce").fillna(-1).astype(int)

    upd_rows = [{"id": int(r["id"]), "cluster": int(r["cluster"])} for _, r in m.iterrows()]
    if not upd_rows:
        return 0

    upd_sql = """
        UPDATE sku_profile
        SET cluster = :cluster,
            last_updated = CURRENT_TIMESTAMP
        WHERE id = :id
    """
    with engine.begin() as conn:
        res = conn.execute(text(upd_sql), upd_rows)
        try:
            return int(res.rowcount)
        except Exception:
            return int(len(upd_rows))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _metric_pack(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    out = {"mae": None, "mse": None, "rmse": None, "mape": None, "n": int(len(yt))}
    if len(yt) <= 0:
        return out
    out["mae"] = mae(yt, yp)
    out["mse"] = mse(yt, yp)
    out["rmse"] = rmse(yt, yp)
    out["mape"] = mape(yt, yp)
    return out


def _get_existing_model_run_cols() -> set[str]:
    return _get_table_columns("model_run")


def _get_forecast_config_from_db():
    keys = ["TRAIN_END", "TEST_START", "TEST_END", "WIN_START"]
    placeholders = ",".join([f":k{i}" for i in range(len(keys))])
    params = {f"k{i}": key for i, key in enumerate(keys)}

    sql = f"""
        SELECT config_key, config_value
        FROM forecast_config
        WHERE config_key IN ({placeholders})
    """

    mapping = {}
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).fetchall()
        for row in rows:
            mapping[row.config_key] = row.config_value

    def parse_date(val):
        if not val:
            return None
        try:
            d = dt.date.fromisoformat(str(val))
            return _to_month_start_date(d)
        except Exception:
            return None

    train_end = parse_date(mapping.get("TRAIN_END"))
    test_start = parse_date(mapping.get("TEST_START"))
    test_end = parse_date(mapping.get("TEST_END"))
    win_start = parse_date(mapping.get("WIN_START"))
    return train_end, test_start, test_end, win_start


def _get_n_clusters_from_db(default: int = 4) -> int:
    sql = """
        SELECT config_value
        FROM forecast_config
        WHERE config_key = 'N_CLUSTERS'
        LIMIT 1
    """
    with engine.begin() as conn:
        row = conn.execute(text(sql)).fetchone()

    if not row or row[0] is None:
        v = int(default)
    else:
        s = str(row[0]).strip()
        m = re.search(r"\d+", s)
        v = int(m.group(0)) if m else int(default)

    if v < 2:
        v = 2
    if v > 12:
        v = 12
    return int(v)


def _get_eligibility_rule_from_db():
    keys = [
        "ELIG_MIN_N_MONTHS",
        "ELIG_MIN_NONZERO",
        "ELIG_MIN_TOTAL_QTY",
        "ELIG_MAX_ZERO_RATIO",
        "ELIG_ALIVE_RECENT_MONTHS",
        "ELIG_REQUIRE_LAST_MONTH",
        "ELIG_REQUIRE_QTY_12M_GT0",
    ]
    placeholders = ",".join([f":k{i}" for i in range(len(keys))])
    params = {f"k{i}": key for i, key in enumerate(keys)}

    sql = f"""
        SELECT config_key, config_value
        FROM forecast_config
        WHERE config_key IN ({placeholders})
    """

    mapping = {}
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).fetchall()
        for row in rows:
            mapping[row.config_key] = row.config_value

    def _to_int(val, default):
        try:
            if val is None:
                return default
            if hasattr(val, "n"):
                return int(getattr(val, "n"))
            if isinstance(val, (np.integer, int)):
                return int(val)
            s = str(val).strip()
            m = re.search(r"-?\d+", s)
            if m:
                return int(m.group(0))
            return default
        except Exception:
            return default

    def _to_float(val, default):
        try:
            if val is None:
                return default
            if isinstance(val, (np.floating, float, int, np.integer)):
                return float(val)
            s = str(val).strip().replace(",", "")
            return float(s)
        except Exception:
            return default

    def _to_bool01(val, default):
        if val is None:
            return default
        s = str(val).strip().lower()
        if s in ["1", "true", "yes", "y", "on"]:
            return True
        if s in ["0", "false", "no", "n", "off"]:
            return False
        return default

    rule = {
        "min_n_months": _to_int(mapping.get("ELIG_MIN_N_MONTHS"), 36),
        "min_nonzero": _to_int(mapping.get("ELIG_MIN_NONZERO"), 10),
        "min_total_qty": _to_float(mapping.get("ELIG_MIN_TOTAL_QTY"), 30.0),
        "max_zero_ratio": _to_float(mapping.get("ELIG_MAX_ZERO_RATIO"), 0.7),
        "alive_recent_months": _to_int(mapping.get("ELIG_ALIVE_RECENT_MONTHS"), 3),
        "require_last_month": _to_bool01(mapping.get("ELIG_REQUIRE_LAST_MONTH"), True),
        "require_qty_12m_gt0": _to_bool01(mapping.get("ELIG_REQUIRE_QTY_12M_GT0"), True),
    }
    return rule


def _summarize_rule(rule: dict) -> str:
    try:
        return (
            f"Minimal bulan data {int(rule.get('min_n_months', 0))}. "
            f"Minimal bulan ada penjualan {int(rule.get('min_nonzero', 0))}. "
            f"Minimal total penjualan {float(rule.get('min_total_qty', 0)):.0f}. "
            f"Maks rasio bulan tanpa penjualan {float(rule.get('max_zero_ratio', 0)):.2f}. "
            f"Maks jarak dari penjualan terakhir {int(rule.get('alive_recent_months', 0))} bulan. "
            f"Wajib ada data di batas latih {'Ya' if bool(rule.get('require_last_month', True)) else 'Tidak'}. "
            f"Wajib total 12 bulan terakhir > 0 {'Ya' if bool(rule.get('require_qty_12m_gt0', True)) else 'Tidak'}."
        )
    except Exception:
        return "Aturan seleksi tersimpan di parameter model."


def _upsert_forecast_config(configs, user_id=None):
    if not configs:
        return

    sql = """
        INSERT INTO forecast_config (config_key, config_value, updated_by)
        VALUES (:k, :v, :u)
        ON DUPLICATE KEY UPDATE
            config_value = :v,
            updated_by   = :u,
            updated_at   = CURRENT_TIMESTAMP
    """
    with engine.begin() as conn:
        for key, value in configs.items():
            conn.execute(text(sql), {"k": key, "v": str(value), "u": user_id})


def _save_model_run_sku_snapshot(run_id: int, df_panel: pd.DataFrame) -> int:
    if df_panel is None or df_panel.empty:
        return 0

    for c in ["cabang", "sku", "cluster"]:
        if c not in df_panel.columns:
            raise ValueError(f"Kolom {c} tidak ditemukan untuk snapshot model_run_sku.")

    snap = df_panel[["cabang", "sku", "cluster"]].drop_duplicates().copy()

    if "demand_level" in df_panel.columns:
        try:
            tmp = df_panel[["cabang", "sku", "demand_level"]].drop_duplicates()
            snap = snap.merge(tmp, on=["cabang", "sku"], how="left")
        except Exception:
            snap["demand_level"] = None
    else:
        snap["demand_level"] = None

    snap = _norm_keys_df(snap)
    snap["model_run_id"] = int(run_id)
    snap["eligible_model"] = 1

    snap["cluster"] = pd.to_numeric(snap["cluster"], errors="coerce").fillna(-1).astype(int)
    snap = snap[snap["cluster"] >= 0].copy()

    if snap.empty:
        return 0

    insert_sql = """
        INSERT INTO model_run_sku (model_run_id, cabang, sku, cluster, demand_level, eligible_model)
        VALUES (:model_run_id, :cabang, :sku, :cluster, :demand_level, :eligible_model)
        ON DUPLICATE KEY UPDATE
            cluster = VALUES(cluster),
            demand_level = VALUES(demand_level),
            eligible_model = VALUES(eligible_model)
    """

    rows = snap.to_dict("records")
    with engine.begin() as conn:
        conn.execute(text(insert_sql), rows)

    return int(len(rows))


def _save_model_run_sku_metrics(run_id: int, df_pred: pd.DataFrame, test_obs_mask: pd.Series) -> int:
    if df_pred is None or df_pred.empty:
        return 0

    need_cols = {"cabang", "sku", "qty", "pred_qty"}
    if not need_cols.issubset(set(df_pred.columns)):
        return 0

    model_run_sku_cols = _get_table_columns("model_run_sku")
    metric_cols = []
    if "test_mae" in model_run_sku_cols:
        metric_cols.append("test_mae")
    if "test_rmse" in model_run_sku_cols:
        metric_cols.append("test_rmse")
    if "test_mape" in model_run_sku_cols:
        metric_cols.append("test_mape")
    if "test_n_obs" in model_run_sku_cols:
        metric_cols.append("test_n_obs")

    if not metric_cols:
        return 0

    dfm = df_pred.loc[test_obs_mask, ["cabang", "sku", "qty", "pred_qty"]].copy()
    if dfm.empty:
        return 0

    dfm = _norm_keys_df(dfm)
    dfm["qty"] = pd.to_numeric(dfm["qty"], errors="coerce")
    dfm["pred_qty"] = pd.to_numeric(dfm["pred_qty"], errors="coerce")
    dfm = dfm.dropna(subset=["qty", "pred_qty"])
    dfm = dfm[dfm["qty"] > 0].copy()
    if dfm.empty:
        return 0

    rows = []
    for (cab, sku), g in dfm.groupby(["cabang", "sku"], sort=False):
        y = g["qty"].values
        p = g["pred_qty"].values
        pack = _metric_pack(y, p)
        rec = {"model_run_id": int(run_id), "cabang": cab, "sku": sku}

        if "test_mae" in metric_cols:
            rec["test_mae"] = pack.get("mae")
        if "test_rmse" in metric_cols:
            rec["test_rmse"] = pack.get("rmse")
        if "test_mape" in metric_cols:
            rec["test_mape"] = pack.get("mape")
        if "test_n_obs" in metric_cols:
            rec["test_n_obs"] = int(pack.get("n") or 0)

        rows.append(rec)

    if not rows:
        return 0

    upd_parts = []
    if "test_mae" in metric_cols:
        upd_parts.append("test_mae = VALUES(test_mae)")
    if "test_rmse" in metric_cols:
        upd_parts.append("test_rmse = VALUES(test_rmse)")
    if "test_mape" in metric_cols:
        upd_parts.append("test_mape = VALUES(test_mape)")
    if "test_n_obs" in metric_cols:
        upd_parts.append("test_n_obs = VALUES(test_n_obs)")

    set_sql = ",\n            ".join(upd_parts)

    insert_cols = ["model_run_id", "cabang", "sku"] + metric_cols
    col_sql = ", ".join(insert_cols)
    val_sql = ", ".join([f":{c}" for c in insert_cols])

    insert_sql = f"""
        INSERT INTO model_run_sku ({col_sql})
        VALUES ({val_sql})
        ON DUPLICATE KEY UPDATE
            {set_sql}
    """

    with engine.begin() as conn:
        conn.execute(text(insert_sql), rows)

    return int(len(rows))


def _get_sales_date_range():
    sql = """
        SELECT MIN(periode) AS min_periode,
               MAX(periode) AS max_periode
        FROM sales_monthly
    """
    with engine.begin() as conn:
        row = conn.execute(text(sql)).fetchone()

    if not row or row.min_periode is None:
        return None, None

    return _to_month_start_any(row.min_periode), _to_month_start_any(row.max_periode)


def _get_default_train_end_for_profile():
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(periode) AS max_periode FROM sales_monthly")).mappings().fetchone()
    if row and row["max_periode"] is not None:
        max_per = _to_month_start_any(row["max_periode"])
        if max_per is not None:
            return max_per

    today = date.today()
    return date(today.year, today.month, 1)


def _load_cluster_map_from_db() -> pd.DataFrame:
    wanted = ["cabang", "sku", "cluster", "demand_level", "eligible_model"]
    cols = _select_existing_cols("sku_profile", wanted)
    if not cols:
        return pd.DataFrame(columns=["cabang", "sku", "cluster", "demand_level", "eligible_model"])

    sql = f"SELECT {', '.join(cols)} FROM sku_profile"
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn)

    df = _norm_keys_df(df)
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(int)
    if "eligible_model" in df.columns:
        df["eligible_model"] = pd.to_numeric(df["eligible_model"], errors="coerce").fillna(0).astype(int)
    return df


def _cast_bool_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "eligible_model" in df.columns:
        df["eligible_model"] = pd.to_numeric(df["eligible_model"], errors="coerce").fillna(0).astype(int).astype(bool)
    if "has_last" in df.columns:
        df["has_last"] = pd.to_numeric(df["has_last"], errors="coerce").fillna(0).astype(int).astype(bool)
    return df


@st.cache_data(ttl=30)
def _load_sku_profile_fast(cabang=None, sku=None, eligible=None, cluster=None, limit=2000, offset=0) -> pd.DataFrame:
    wanted = [
        "id",
        "cabang",
        "sku",
        "demand_level",
        "cluster",
        "eligible_model",
        "n_months",
        "nonzero_months",
        "total_qty",
        "zero_ratio_train",
        "qty_12m",
        "qty_6m",
        "months_since_last_nz",
        "has_last",
        "last_updated",
    ]
    cols = _select_existing_cols("sku_profile", wanted)
    if not cols:
        return pd.DataFrame()

    sql = f"""
        SELECT {', '.join(cols)}
        FROM sku_profile
        WHERE 1=1
          AND (:cabang IS NULL OR cabang = :cabang)
          AND (:sku IS NULL OR sku = :sku)
          AND (:eligible IS NULL OR eligible_model = :eligible)
          AND (:cluster IS NULL OR cluster = :cluster)
        ORDER BY cabang, sku
        LIMIT :lim OFFSET :off
    """
    params = {
        "cabang": cabang,
        "sku": sku,
        "eligible": eligible,
        "cluster": cluster,
        "lim": int(limit),
        "off": int(offset),
    }
    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    df = _norm_keys_df(df)
    df = _cast_bool_cols(df)
    return df


@st.cache_data(ttl=30)
def _load_sku_profile_filters():
    sql = """
        SELECT
            (SELECT COUNT(*) FROM sku_profile) AS n_rows,
            (SELECT COUNT(DISTINCT cabang) FROM sku_profile) AS n_cabang,
            (SELECT COUNT(DISTINCT sku) FROM sku_profile) AS n_sku
    """
    with engine.begin() as conn:
        row = conn.execute(text(sql)).mappings().fetchone()
    n_rows = int(row["n_rows"]) if row and row["n_rows"] is not None else 0
    n_cabang = int(row["n_cabang"]) if row and row["n_cabang"] is not None else 0
    n_sku = int(row["n_sku"]) if row and row["n_sku"] is not None else 0

    with engine.begin() as conn:
        cab = pd.read_sql(text("SELECT DISTINCT cabang FROM sku_profile ORDER BY cabang"), conn)
        sku = pd.read_sql(text("SELECT DISTINCT sku FROM sku_profile ORDER BY sku"), conn)
        clu = pd.read_sql(text("SELECT DISTINCT cluster FROM sku_profile ORDER BY cluster"), conn)

    cabangs = _norm_keys_df(cab)["cabang"].dropna().astype(str).tolist() if not cab.empty else []
    skus = _norm_keys_df(sku)["sku"].dropna().astype(str).tolist() if not sku.empty else []
    clusters = []
    if not clu.empty and "cluster" in clu.columns:
        clusters = pd.to_numeric(clu["cluster"], errors="coerce").dropna().astype(int).tolist()

    return n_rows, n_cabang, n_sku, cabangs, skus, clusters


@st.cache_data(ttl=20)
def _training_ready_counts():
    sql = """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN cluster IS NOT NULL AND cluster >= 0 THEN 1 ELSE 0 END) AS has_cluster,
            SUM(CASE WHEN eligible_model = 1 THEN 1 ELSE 0 END) AS eligible,
            SUM(CASE WHEN eligible_model = 0 THEN 1 ELSE 0 END) AS not_eligible
        FROM sku_profile
    """
    with engine.begin() as conn:
        r = conn.execute(text(sql)).mappings().fetchone()
    if not r:
        return 0, 0, 0, 0
    return int(r["total"] or 0), int(r["has_cluster"] or 0), int(r["eligible"] or 0), int(r["not_eligible"] or 0)


ELIG_COLCFG = {
    "cabang": st.column_config.TextColumn("Cabang"),
    "sku": st.column_config.TextColumn("Kode Produk"),
    "n_months": st.column_config.NumberColumn("Jumlah Bulan Data", format="%d"),
    "nonzero_months": st.column_config.NumberColumn("Bulan Ada Penjualan", format="%d"),
    "total_qty": st.column_config.NumberColumn("Total Penjualan", format="%.0f"),
    "zero_ratio_train": st.column_config.NumberColumn("Rasio Bulan 0 (Train)", format="%.2f"),
    "qty_12m": st.column_config.NumberColumn("Total 12 Bulan Terakhir", format="%.0f"),
    "qty_6m": st.column_config.NumberColumn("Total 6 Bulan Terakhir", format="%.0f"),
    "months_since_last_nz": st.column_config.NumberColumn("Jarak dari Penjualan Terakhir (Bulan)", format="%d"),
    "has_last": st.column_config.CheckboxColumn("Ada Data di Batas Latih"),
    "eligible_model": st.column_config.CheckboxColumn("Layak Dilatih"),
}

PROFILE_COLCFG = {
    "cabang": st.column_config.TextColumn("Cabang"),
    "sku": st.column_config.TextColumn("Kode Produk"),
    "demand_level": st.column_config.TextColumn("Level Permintaan"),
    "eligible_model": st.column_config.CheckboxColumn("Layak Dilatih"),
    "n_months": st.column_config.NumberColumn("Jumlah Bulan Data", format="%d"),
    "nonzero_months": st.column_config.NumberColumn("Bulan Ada Penjualan", format="%d"),
    "total_qty": st.column_config.NumberColumn("Total Penjualan", format="%.0f"),
    "zero_ratio_train": st.column_config.NumberColumn("Rasio Bulan 0", format="%.2f"),
    "qty_12m": st.column_config.NumberColumn("Qty 12 Bulan", format="%.0f"),
    "qty_6m": st.column_config.NumberColumn("Qty 6 Bulan", format="%.0f"),
    "months_since_last_nz": st.column_config.NumberColumn("Bulan sejak penjualan terakhir", format="%d"),
    "has_last": st.column_config.CheckboxColumn("Ada Data di Batas Latih"),
    "cluster": st.column_config.NumberColumn("Kelompok Produk", format="%d"),
    "last_updated": st.column_config.DatetimeColumn("Update Terakhir", format="YYYY-MM-DD HH:mm"),
}


st.set_page_config(
    page_title="Admin · Training Forecast",
    layout="wide",
    initial_sidebar_state="collapsed",
)

require_login()
inject_global_theme()
render_sidebar_user_and_logout()

if "user" not in st.session_state:
    st.error("Silakan login dulu.")
    st.stop()

user = st.session_state["user"]
user_id = user.get("user_id")
role = user.get("role", "user")

if role != "admin":
    st.error("Halaman ini hanya untuk Admin.")
    st.stop()

if "page_uid_admin_train" not in st.session_state:
    st.session_state["page_uid_admin_train"] = str(uuid.uuid4())
page_uid = st.session_state["page_uid_admin_train"]

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {font-family:'Poppins', sans-serif;}
    [data-testid="stVerticalBlockBorderWrapper"]{
        background:#ffffff !important;
        border:1px solid #e5e7eb !important;
        border-radius:16px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div{
        padding: 6px 6px;
        border-radius: 16px;
    }
    .hdr { margin-bottom: 16px; }
    .hdr .t { font-size: 26px; font-weight: 900; color:#111827; margin:0; }
    .pill {
      display:inline-block; margin-left:10px;
      background:#111827; color:#fff; border:1px solid #1f2937;
      font-size:11px; font-weight:900; padding:4px 12px; border-radius:999px;
      letter-spacing:.4px; text-transform:uppercase;
    }
    .topcards { display:grid; grid-template-columns: 1fr 1.2fr 1fr; gap:12px; margin-top: 10px; }
    .topcard {
      background:#000; border:1px solid #333; border-radius:20px;
      padding:18px; color:#fff; box-shadow:0 4px 20px rgba(0,0,0,.35);
      min-height: 92px;
    }
    .k { font-size: 11px; opacity:.65; font-weight:900; text-transform:uppercase; letter-spacing:.5px; }
    .v { font-size: 20px; font-weight:950; margin-top:8px; line-height:1.2; }
    .stButton > button[kind="primary"] {border-radius:999px; padding:0.60rem 1.35rem; font-weight:900;}
    [data-testid="stDataFrame"] {font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_OUT_DIR = PROJECT_ROOT / "outputs" / "Light_Gradient_Boosting_Machine"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

with engine.begin() as conn:
    row = conn.execute(text("SELECT COUNT(*) AS n_sku FROM sku_profile")).mappings().fetchone()
    n_sku_profile = int(row["n_sku"]) if row and row["n_sku"] is not None else 0

min_periode, max_periode = _get_sales_date_range()

models_for_header = get_all_model_runs()
active_model = None
if models_for_header:
    for m in models_for_header:
        if m.get("active_flag") == 1:
            active_model = m
            break

st.markdown(
    """
    <div class="hdr">
      <div class="t">Panel Forecast <span class="pill">Admin</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _fmt_month(d):
    if not d:
        return "-"
    return pd.Timestamp(d).strftime("%b %Y")


period_text = "-"
if min_periode and max_periode:
    period_text = f"{_fmt_month(min_periode)} sampai {_fmt_month(max_periode)}"

model_text = "Belum ada"
model_sub = "Belum dipilih"
if active_model:
    model_name = str(active_model.get("model_type", "Unknown")).upper()
    model_id = active_model.get("id")
    model_text = f"{model_name} #{model_id}"
    model_sub = "Sedang dipakai"

st.markdown(
    f"""
    <div class="topcards">
      <div class="topcard">
        <div class="k">Total cabang x SKU</div>
        <div class="v">{n_sku_profile:,}</div>
      </div>
      <div class="topcard">
        <div class="k">Rentang data penjualan</div>
        <div class="v">{period_text}</div>
      </div>
      <div class="topcard">
        <div class="k">Model aktif</div>
        <div class="v">{model_text}</div>
        <div style="font-size:11px; opacity:.60; margin-top:8px; font-weight:800;">{model_sub}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

TAB_LABELS = ["1. Update Profil", "2. Aturan Produk Layak", "3. Latih Model", "4. Riwayat Model"]
if "admin_train_tab" not in st.session_state:
    st.session_state["admin_train_tab"] = TAB_LABELS[0]

picked = st.segmented_control("", options=TAB_LABELS, key="admin_train_tab")
st.write("")


if picked == "1. Update Profil":
    st.markdown("### Update profil produk dan kelompok")

    n_rows, n_cabang, n_sku, cabangs, skus, clusters = _load_sku_profile_filters()

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Total baris profil", f"{n_rows:,}")
        m2.metric("Jumlah cabang", f"{n_cabang:,}")
        m3.metric("Jumlah produk", f"{n_sku:,}")

    with st.container(border=True):
        st.markdown("Filter cepat")
        f1, f2, f3, f4 = st.columns([1, 1, 1, 1], gap="large")

        with f1:
            cab_opt = ["(Semua cabang)"] + cabangs
            selected_cabang = st.selectbox("Cabang", options=cab_opt, index=0, key="f_prof_cab")

        with f2:
            sku_opt = ["(Semua produk)"] + skus
            selected_sku = st.selectbox("Produk", options=sku_opt, index=0, key="f_prof_sku")

        with f3:
            elig_opt = ["(Semua)", "Layak", "Tidak layak"]
            selected_elig = st.selectbox("Status", options=elig_opt, index=0, key="f_prof_elig")

        with f4:
            clu_opt = ["(Semua)"] + [str(x) for x in clusters]
            selected_cluster = st.selectbox("Kelompok", options=clu_opt, index=0, key="f_prof_cluster")

        p1, p2 = st.columns([1, 1], gap="large")
        with p1:
            page_size = st.selectbox("Baris per halaman", options=[200, 500, 1000, 2000], index=1, key="f_prof_ps")
        with p2:
            max_page = max(1, (int(n_rows) // int(page_size)) + 1)
            page = st.number_input(
                "Halaman",
                min_value=1,
                max_value=max_page,
                value=1,
                step=1,
                key="f_prof_page",
            )

    cab_param = None if selected_cabang == "(Semua cabang)" else selected_cabang
    sku_param = None if selected_sku == "(Semua produk)" else selected_sku

    if selected_elig == "Layak":
        elig_param = 1
    elif selected_elig == "Tidak layak":
        elig_param = 0
    else:
        elig_param = None

    cluster_param = None if selected_cluster == "(Semua)" else int(selected_cluster)

    offset = (int(page) - 1) * int(page_size)
    df_view = _load_sku_profile_fast(
        cabang=cab_param,
        sku=sku_param,
        eligible=elig_param,
        cluster=cluster_param,
        limit=int(page_size),
        offset=int(offset),
    )

    with st.container(border=True):
        if df_view.empty:
            st.info("Tidak ada data yang cocok dengan filter.")
        else:
            st.dataframe(
                df_view,
                use_container_width=True,
                height=560,
                hide_index=True,
                column_config=PROFILE_COLCFG,
            )

    with st.container(border=True):
        st.markdown("Jalankan update profil")
        st.caption("Klik sekali. Tunggu selesai. Setelah itu lanjut ke Aturan Produk Layak.")

        default_train_end = _get_default_train_end_for_profile()
        train_end_profile = st.date_input(
            "Batas data yang dihitung",
            value=default_train_end,
            help="Data setelah tanggal ini tidak ikut dihitung untuk profil.",
            key="profile_train_end",
        )

        default_n_clusters = _get_n_clusters_from_db(default=4)
        n_clusters = st.number_input(
            "Jumlah kelompok produk",
            min_value=2,
            max_value=12,
            value=int(default_n_clusters),
            step=1,
            key="profile_n_clusters",
            help="Kelompok dibuat otomatis dari pola penjualan produk.",
        )

        run_refresh = st.button(
            "Update profil sekarang",
            type="primary",
            use_container_width=True,
            key="btn_refresh_profile_cluster",
        )

        if run_refresh:
            with st.status("Proses berjalan...", expanded=True) as status:
                try:
                    train_end_ms = _to_month_start_date(train_end_profile)
                    train_end_ts = pd.Timestamp(train_end_ms)

                    cfg2 = _get_forecast_config_from_db()
                    _, _, _, win_start_cfg2 = cfg2[:4]
                    win_start_ms = win_start_cfg2 if win_start_cfg2 else dt.date(2021, 1, 1)
                    win_start_ts = pd.Timestamp(_to_month_start_date(win_start_ms))

                    _upsert_forecast_config({"N_CLUSTERS": int(n_clusters)}, user_id=user_id)

                    st.write("Hitung profil produk")
                    n_rows_updated = build_and_store_sku_profile(train_end=train_end_ts, win_start=win_start_ts)

                    st.write("Susun kelompok produk")
                    n_cluster_updated = cluster_and_update_from_sku_profile(engine, n_clusters=int(n_clusters))

                    st.cache_data.clear()
                    status.update(label="Selesai", state="complete")
                    st.success(
                        f"Selesai. Profil diperbarui: {int(n_rows_updated):,} baris. "
                        f"Kelompok diperbarui: {int(n_cluster_updated):,} baris."
                    )
                    st.rerun()
                except Exception as e:
                    status.update(label="Gagal", state="error")
                    st.error(f"Update profil gagal: {e}")


elif picked == "2. Aturan Produk Layak":
    st.markdown("### Aturan produk layak dilatih")

    def _to_int_safe(v, default: int) -> int:
        try:
            if v is None:
                return int(default)
            s = str(v).strip()
            m = re.search(r"-?\d+", s)
            return int(m.group(0)) if m else int(default)
        except Exception:
            return int(default)

    def _get_train_window_months() -> int:
        train_end_cfg, _, _, win_start_cfg = _get_forecast_config_from_db()
        if train_end_cfg and win_start_cfg:
            a = _to_month_start_date(win_start_cfg)
            b = _to_month_start_date(train_end_cfg)
            return max(1, (b.year - a.year) * 12 + (b.month - a.month) + 1)

        a, b = _get_sales_date_range()
        if a and b:
            a = _to_month_start_date(a)
            b = _to_month_start_date(b)
            return max(1, (b.year - a.year) * 12 + (b.month - a.month) + 1)

        return 0

    def _get_company_policy() -> dict:
        defaults = {"fast_per_month": -1, "special_per_month": -1, "dead_months": -1}

        keys = ["FAST_PER_MONTH", "SPECIAL_PER_MONTH", "DEAD_MONTHS"]
        placeholders = ",".join([f":k{i}" for i in range(len(keys))])
        params = {f"k{i}": keys[i] for i in range(len(keys))}
        sql = f"""
            SELECT config_key, config_value
            FROM forecast_config
            WHERE config_key IN ({placeholders})
        """

        got = {}
        try:
            with engine.begin() as conn:
                rows = conn.execute(text(sql), params).fetchall()
            for r in rows:
                got[str(r[0])] = r[1]
        except Exception:
            got = {}

        out = {
            "fast_per_month": _to_int_safe(got.get("FAST_PER_MONTH"), defaults["fast_per_month"]),
            "special_per_month": _to_int_safe(got.get("SPECIAL_PER_MONTH"), defaults["special_per_month"]),
            "dead_months": _to_int_safe(got.get("DEAD_MONTHS"), defaults["dead_months"]),
        }
        return out

    @st.cache_data(ttl=60)
    def _load_stats_source():
        wanted = [
            "n_months",
            "nonzero_months",
            "total_qty",
            "qty_mean",
            "zero_ratio_train",
            "months_since_last_nz",
            "qty_12m",
            "qty_6m",
            "has_last",
        ]
        cols = _select_existing_cols("sku_profile", wanted)
        if not cols:
            return pd.DataFrame()

        with engine.begin() as conn:
            df = pd.read_sql(text(f"SELECT {', '.join(cols)} FROM sku_profile"), conn)

        for c in [
            "n_months",
            "nonzero_months",
            "total_qty",
            "qty_mean",
            "zero_ratio_train",
            "months_since_last_nz",
            "qty_12m",
            "qty_6m",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "has_last" in df.columns:
            df["has_last"] = pd.to_numeric(df["has_last"], errors="coerce").fillna(0).astype(int)

        return df

    def _pick_alive_mask(df: pd.DataFrame, dead_months: int | None) -> pd.Series:
        if df is None or df.empty:
            return pd.Series([], dtype=bool)

        m = pd.Series([True] * len(df), index=df.index)

        if "months_since_last_nz" in df.columns and dead_months is not None and dead_months >= 0:
            ms = pd.to_numeric(df["months_since_last_nz"], errors="coerce")
            m = m & ms.notna() & (ms <= int(dead_months))

        if "qty_6m" in df.columns:
            q6 = pd.to_numeric(df["qty_6m"], errors="coerce").fillna(0.0)
            m = m & (q6 > 0)

        return m

    def _quantile_safe(s: pd.Series, q: float, default: float) -> float:
        try:
            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                return float(default)
            return float(s.quantile(float(q)))
        except Exception:
            return float(default)

    def _suggest_from_data(preset: str) -> dict:
        preset_u = str(preset or "").strip().upper()
        n_win = int(_get_train_window_months())

        df = _load_stats_source()
        policy = _get_company_policy()

        dead_cfg = int(policy.get("dead_months", -1))
        dead_months = dead_cfg if dead_cfg >= 0 else 6

        alive_mask = _pick_alive_mask(df, dead_months)
        df_alive = df.loc[alive_mask].copy() if (df is not None and not df.empty and alive_mask.any()) else df.copy()

        if not df_alive.empty and "qty_mean" not in df_alive.columns:
            if ("total_qty" in df_alive.columns) and ("n_months" in df_alive.columns):
                n = pd.to_numeric(df_alive["n_months"], errors="coerce").replace(0, np.nan)
                t = pd.to_numeric(df_alive["total_qty"], errors="coerce")
                df_alive["qty_mean"] = (t / n).fillna(0.0)
            else:
                df_alive["qty_mean"] = 0.0

        fast_cfg = int(policy.get("fast_per_month", -1))
        special_cfg = int(policy.get("special_per_month", -1))

        fast_per_month = fast_cfg if fast_cfg >= 0 else 5
        special_per_month = special_cfg if special_cfg >= 0 else 30

        zr_default = float(st.session_state.get("elig_max_zero_ratio", 0.70))
        if df_alive.empty or "zero_ratio_train" not in df_alive.columns:
            zr_suggest = float(np.clip(zr_default, 0.0, 1.0))
        else:
            zr = pd.to_numeric(df_alive["zero_ratio_train"], errors="coerce").dropna()
            if zr.empty:
                zr_suggest = float(np.clip(zr_default, 0.0, 1.0))
            else:
                if preset_u == "SPECIAL PRODUCTS":
                    q = 0.50
                elif preset_u == "FAST MOVING":
                    q = 0.60
                elif preset_u == "ALIVE GENERAL":
                    q = 0.75
                else:
                    q = 0.70
                zr_suggest = float(np.clip(float(zr.quantile(q)), 0.0, 1.0))

        min_n_months = int(n_win) if n_win > 0 else int(st.session_state.get("elig_min_n_months", 36))
        min_nonzero = int(np.clip(int(np.ceil((1.0 - zr_suggest) * float(min_n_months))), 0, int(min_n_months)))

        if preset_u == "FAST MOVING":
            min_total_qty = int(max(0, min_n_months * int(fast_per_month)))
        elif preset_u == "SPECIAL PRODUCTS":
            min_total_qty = int(max(0, min_n_months * int(special_per_month)))
        elif preset_u == "ALIVE GENERAL":
            if df_alive.empty or "total_qty" not in df_alive.columns:
                min_total_qty = int(st.session_state.get("elig_min_total_qty", 30))
            else:
                tq = pd.to_numeric(df_alive["total_qty"], errors="coerce").dropna()
                base = float(st.session_state.get("elig_min_total_qty", 30))
                q50 = _quantile_safe(tq, 0.50, base)
                min_total_qty = int(max(0, round(q50)))
        else:
            min_total_qty = int(st.session_state.get("elig_min_total_qty", 30))

        alive_recent_months = int(max(0, dead_months))

        if preset_u in ["FAST MOVING", "SPECIAL PRODUCTS"]:
            require_last_month = True
            require_qty_12m_gt0 = True
        elif preset_u == "ALIVE GENERAL":
            require_last_month = False
            require_qty_12m_gt0 = True
        else:
            require_last_month = bool(st.session_state.get("elig_require_last_month", True))
            require_qty_12m_gt0 = bool(st.session_state.get("elig_require_qty_12m_gt0", True))

        return {
            "min_n_months": int(min_n_months),
            "min_nonzero": int(min_nonzero),
            "min_total_qty": int(min_total_qty),
            "max_zero_ratio": float(zr_suggest),
            "alive_recent_months": int(alive_recent_months),
            "require_last_month": bool(require_last_month),
            "require_qty_12m_gt0": bool(require_qty_12m_gt0),
            "meta": {
                "window_months": int(n_win),
                "fast_per_month": int(fast_per_month),
                "special_per_month": int(special_per_month),
                "dead_months": int(dead_months),
            },
        }

    def _rule_hash(rule: dict) -> str:
        keys = [
            "min_n_months",
            "min_nonzero",
            "min_total_qty",
            "max_zero_ratio",
            "alive_recent_months",
            "require_last_month",
            "require_qty_12m_gt0",
        ]
        return "|".join([str(rule.get(k)) for k in keys])

    def _sync_state_from_rule(rule_cfg: dict):
        if "elig_nonce" not in st.session_state:
            st.session_state["elig_nonce"] = 0.0

        current_hash = _rule_hash(rule_cfg)
        last_hash = st.session_state.get("elig_rule_hash")

        if (last_hash is None) or (last_hash != current_hash):
            st.session_state["elig_rule_hash"] = current_hash
            st.session_state["elig_min_n_months"] = int(rule_cfg["min_n_months"])
            st.session_state["elig_min_nonzero"] = int(rule_cfg["min_nonzero"])
            st.session_state["elig_min_total_qty"] = int(float(rule_cfg["min_total_qty"]))
            st.session_state["elig_max_zero_ratio"] = float(rule_cfg["max_zero_ratio"])
            st.session_state["elig_alive_recent_months"] = int(rule_cfg["alive_recent_months"])
            st.session_state["elig_require_last_month"] = bool(rule_cfg["require_last_month"])
            st.session_state["elig_require_qty_12m_gt0"] = bool(rule_cfg["require_qty_12m_gt0"])

    def _apply_preset_to_inputs(preset_label: str):
        preset_u = str(preset_label or "").strip().upper()
        if preset_u == "FAST MOVING":
            rec = _suggest_from_data("FAST MOVING")
        elif preset_u == "SPECIAL PRODUCTS":
            rec = _suggest_from_data("SPECIAL PRODUCTS")
        elif preset_u == "ALIVE GENERAL":
            rec = _suggest_from_data("ALIVE GENERAL")
        else:
            return

        st.session_state["elig_min_n_months"] = int(rec["min_n_months"])
        st.session_state["elig_min_nonzero"] = int(rec["min_nonzero"])
        st.session_state["elig_min_total_qty"] = int(rec["min_total_qty"])
        st.session_state["elig_max_zero_ratio"] = float(rec["max_zero_ratio"])
        st.session_state["elig_alive_recent_months"] = int(rec["alive_recent_months"])
        st.session_state["elig_require_last_month"] = bool(rec["require_last_month"])
        st.session_state["elig_require_qty_12m_gt0"] = bool(rec["require_qty_12m_gt0"])
        st.session_state["elig_last_meta"] = rec.get("meta", {})

    def _on_preset_change():
        p = st.session_state.get("elig_preset_pick", "Manual")
        if p == "Fast Moving":
            _apply_preset_to_inputs("FAST MOVING")
        elif p == "Special Products":
            _apply_preset_to_inputs("SPECIAL PRODUCTS")
        elif p == "Alive General":
            _apply_preset_to_inputs("ALIVE GENERAL")

    # ambil rule dari DB lalu sinkron ke state
    rule_cfg = _get_eligibility_rule_from_db()
    _sync_state_from_rule(rule_cfg)

    # inisialisasi default kalau belum ada (penting supaya widget tanpa value= aman)
    st.session_state.setdefault("elig_min_n_months", 36)
    st.session_state.setdefault("elig_min_nonzero", 10)
    st.session_state.setdefault("elig_min_total_qty", 30)
    st.session_state.setdefault("elig_max_zero_ratio", 0.70)
    st.session_state.setdefault("elig_alive_recent_months", 6)
    st.session_state.setdefault("elig_require_last_month", True)
    st.session_state.setdefault("elig_require_qty_12m_gt0", True)

    with st.container(border=True):
        st.markdown("Pengaturan aturan")

        preset_opts = ["Manual", "Fast Moving", "Special Products", "Alive General"]
        st.selectbox(
            "Preset",
            options=preset_opts,
            index=0,
            key="elig_preset_pick",
            on_change=_on_preset_change,
            help="Preset mengisi angka otomatis berdasarkan data sku_profile dan window WIN_START sampai TRAIN_END.",
        )

        meta = st.session_state.get("elig_last_meta") or {}
        if meta:
            st.caption(
                f"Sumber hitung: window_months={meta.get('window_months')}, "
                f"fast_per_month={meta.get('fast_per_month')}, "
                f"special_per_month={meta.get('special_per_month')}, "
                f"dead_months={meta.get('dead_months')}"
            )

        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            st.number_input(
                "Minimal bulan data",
                min_value=1,
                max_value=240,
                step=1,
                format="%d",
                key="elig_min_n_months",
            )
            st.number_input(
                "Minimal bulan ada penjualan",
                min_value=0,
                max_value=240,
                step=1,
                format="%d",
                key="elig_min_nonzero",
            )

        with c2:
            st.number_input(
                "Minimal total penjualan",
                min_value=0,
                max_value=10**12,
                step=1,
                format="%d",
                key="elig_min_total_qty",
            )
            st.number_input(
                "Maksimal rasio bulan tanpa penjualan (train)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key="elig_max_zero_ratio",
            )
            st.number_input(
                "Maksimal jarak dari penjualan terakhir (bulan)",
                min_value=0,
                max_value=36,
                step=1,
                format="%d",
                key="elig_alive_recent_months",
            )

        with c3:
            st.checkbox(
                "Wajib ada data di batas latih",
                key="elig_require_last_month",
            )
            st.checkbox(
                "Wajib total 12 bulan terakhir lebih dari 0",
                key="elig_require_qty_12m_gt0",
            )

        apply_now = st.button("Terapkan aturan", type="primary", use_container_width=True, key="btn_apply_elig")

        if apply_now:
            min_n_months = int(st.session_state.get("elig_min_n_months", 36))
            min_nonzero = int(st.session_state.get("elig_min_nonzero", 10))
            min_total_qty = int(st.session_state.get("elig_min_total_qty", 30))
            max_zero_ratio = float(st.session_state.get("elig_max_zero_ratio", 0.70))
            alive_recent_months = int(st.session_state.get("elig_alive_recent_months", 6))
            require_last_month = bool(st.session_state.get("elig_require_last_month", True))
            require_qty_12m_gt0 = bool(st.session_state.get("elig_require_qty_12m_gt0", True))

            with st.status("Proses berjalan...", expanded=True) as status:
                try:
                    _upsert_forecast_config(
                        {
                            "ELIG_MIN_N_MONTHS": int(min_n_months),
                            "ELIG_MIN_NONZERO": int(min_nonzero),
                            "ELIG_MIN_TOTAL_QTY": int(min_total_qty),
                            "ELIG_MAX_ZERO_RATIO": float(max_zero_ratio),
                            "ELIG_ALIVE_RECENT_MONTHS": int(alive_recent_months),
                            "ELIG_REQUIRE_LAST_MONTH": int(bool(require_last_month)),
                            "ELIG_REQUIRE_QTY_12M_GT0": int(bool(require_qty_12m_gt0)),
                        },
                        user_id=user_id,
                    )

                    n_updated = apply_eligibility_to_sku_profile_fast(
                        engine,
                        selection_rule={
                            "min_n_months": int(min_n_months),
                            "min_nonzero": int(min_nonzero),
                            "min_total_qty": float(min_total_qty),
                            "max_zero_ratio": float(max_zero_ratio),
                            "alive_recent_months": int(alive_recent_months),
                            "require_last_month": bool(require_last_month),
                            "require_qty_12m_gt0": bool(require_qty_12m_gt0),
                        },
                    )

                    st.session_state["elig_nonce"] = dt.datetime.now().timestamp()
                    st.cache_data.clear()

                    status.update(label="Selesai", state="complete")
                    st.success(f"Aturan diterapkan. Baris ter-update: {int(n_updated):,}")

                except Exception as e:
                    status.update(label="Gagal", state="error")
                    st.error(f"Terapkan aturan gagal: {e}")

    st.write("")
    st.markdown("### Preview hasil kelayakan")

    elig_nonce = float(st.session_state.get("elig_nonce", 0.0))

    @st.cache_data(ttl=120)
    def _load_distinct_cabangs(_nonce: float):
        with engine.begin() as conn:
            cab = pd.read_sql(text("SELECT DISTINCT cabang FROM sku_profile ORDER BY cabang"), conn)
        cab = _norm_keys_df(cab)
        return cab["cabang"].dropna().astype(str).tolist() if not cab.empty else []

    @st.cache_data(ttl=120)
    def _load_distinct_skus(_nonce: float, cabang):
        sql = """
            SELECT DISTINCT sku
            FROM sku_profile
            WHERE (:cabang IS NULL OR cabang = :cabang)
            ORDER BY sku
        """
        with engine.begin() as conn:
            df = pd.read_sql(text(sql), conn, params={"cabang": cabang})
        df = _norm_keys_df(df)
        return df["sku"].dropna().astype(str).tolist() if not df.empty else []

    @st.cache_data(ttl=60)
    def _counts_filtered(cabang, sku, elig, _nonce: float):
        sql = """
            SELECT
                COUNT(*) total,
                SUM(eligible_model=1) eligible,
                SUM(eligible_model=0) not_eligible
            FROM sku_profile
            WHERE 1=1
              AND (:cabang IS NULL OR cabang = :cabang)
              AND (:sku IS NULL OR sku = :sku)
              AND (:elig IS NULL OR eligible_model = :elig)
        """
        params = {"cabang": cabang, "sku": sku, "elig": elig}
        with engine.begin() as conn:
            r = conn.execute(text(sql), params).fetchone()
        return int(r[0] or 0), int(r[1] or 0), int(r[2] or 0)

    @st.cache_data(ttl=60)
    def _load_table_filtered(cabang, sku, elig, limit: int, offset: int, _nonce: float):
        wanted = [
            "cabang",
            "sku",
            "n_months",
            "nonzero_months",
            "total_qty",
            "zero_ratio_train",
            "qty_12m",
            "qty_6m",
            "months_since_last_nz",
            "has_last",
            "eligible_model",
        ]
        cols = _select_existing_cols("sku_profile", wanted)
        if not cols:
            return pd.DataFrame()

        sql = f"""
            SELECT {', '.join(cols)}
            FROM sku_profile
            WHERE 1=1
              AND (:cabang IS NULL OR cabang = :cabang)
              AND (:sku IS NULL OR sku = :sku)
              AND (:elig IS NULL OR eligible_model = :elig)
            ORDER BY eligible_model DESC, cabang, sku
            LIMIT :lim OFFSET :off
        """
        params = {"cabang": cabang, "sku": sku, "elig": elig, "lim": int(limit), "off": int(offset)}
        with engine.begin() as conn:
            df = pd.read_sql(text(sql), conn, params=params)

        df = _norm_keys_df(df)
        df = _cast_bool_cols(df)
        return df

    cabangs2 = _load_distinct_cabangs(elig_nonce)

    with st.container(border=True):
        f1, f2, f3 = st.columns([1, 1, 1], gap="large")

        with f1:
            cab_opt = ["(Semua cabang)"] + cabangs2
            sel_cab = st.selectbox("Cabang", options=cab_opt, index=0, key="elig_f_cab")

        cab_param2 = None if sel_cab == "(Semua cabang)" else sel_cab
        skus_for_dropdown = _load_distinct_skus(elig_nonce, cab_param2)

        with f2:
            sel_sku = st.selectbox(
                "Produk",
                options=["(Semua produk)"] + skus_for_dropdown,
                index=0,
                key="elig_f_sku",
            )

        with f3:
            sel_elig = st.selectbox(
                "Status",
                options=["(Semua)", "Layak", "Tidak layak"],
                index=0,
                key="elig_f_elig",
            )

    sku_param2 = None if sel_sku == "(Semua produk)" else sel_sku

    if sel_elig == "Layak":
        elig_param2 = 1
    elif sel_elig == "Tidak layak":
        elig_param2 = 0
    else:
        elig_param2 = None

    total_f, eligible_f, not_eligible_f = _counts_filtered(cab_param2, sku_param2, elig_param2, elig_nonce)

    with st.container(border=True):
        a1, a2, a3 = st.columns(3, gap="large")
        a1.metric("Total", f"{total_f:,}")
        a2.metric("Layak", f"{eligible_f:,}")
        a3.metric("Tidak layak", f"{not_eligible_f:,}")

    with st.container(border=True):
        t1, t2 = st.columns([1, 1], gap="large")
        with t1:
            page_size = st.selectbox("Baris tabel", options=[500, 1000, 2000, 5000], index=2, key="elig_prev_ps")
        with t2:
            max_page = max(1, (int(total_f) // int(page_size)) + 1)
            page = st.number_input("Halaman", min_value=1, max_value=max_page, value=1, step=1, key="elig_prev_page")

    offset = (int(page) - 1) * int(page_size)
    info = _load_table_filtered(cab_param2, sku_param2, elig_param2, int(page_size), int(offset), elig_nonce)

    with st.container(border=True):
        if info.empty:
            st.info("Tidak ada data sesuai filter.")
        else:
            show_cols = [
                "cabang",
                "sku",
                "n_months",
                "nonzero_months",
                "total_qty",
                "zero_ratio_train",
                "qty_12m",
                "qty_6m",
                "months_since_last_nz",
                "has_last",
                "eligible_model",
            ]
            show_cols = [c for c in show_cols if c in info.columns]

            st.dataframe(
                info[show_cols],
                use_container_width=True,
                height=580,
                hide_index=True,
                column_config=ELIG_COLCFG,
            )

elif picked == "3. Latih Model":
    st.markdown("### Latih model")

    st.markdown(
        """
        <style>
          .t3-space { height: 10px; }
          .stButton > button[kind="primary"] { border-radius:999px; padding:0.70rem 1.35rem; font-weight:900; }
          .stButton > button { border-radius:14px; padding:0.65rem 1.10rem; font-weight:800; }
          div[data-testid="stCaptionContainer"] p { margin-bottom: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if min_periode is None or max_periode is None:
        st.error("Data penjualan tidak ditemukan. Cek tabel sales_monthly.")
        st.stop()

    total_prof, has_cluster_cnt, eligible_cnt, not_eligible_cnt = _training_ready_counts()

    if total_prof == 0:
        st.warning("Profil masih kosong. Jalankan Update profil dulu.")
        st.stop()
    if has_cluster_cnt == 0:
        st.warning("Kelompok produk belum ada. Jalankan Update profil dulu.")
        st.stop()
    if (eligible_cnt + not_eligible_cnt) == 0:
        st.warning("Status layak belum ada. Jalankan Aturan Produk Layak dulu.")
        st.stop()
    if eligible_cnt == 0:
        st.warning("Tidak ada produk yang layak. Longgarkan aturan di Aturan Produk Layak.")
        st.stop()

    if "training_running" not in st.session_state:
        st.session_state["training_running"] = False
    if "training_requested" not in st.session_state:
        st.session_state["training_requested"] = False

    def _add_month(d: dt.date) -> dt.date:
        y, m = d.year, d.month + 1
        if m == 13:
            y += 1
            m = 1
        return dt.date(y, m, 1)

    key_prefix = f"t3_{page_uid}"

    with st.container(border=True):
        st.markdown("#### Atur periode pelatihan")
        st.caption("Pilih batas latih, lama uji, dan awal window evaluasi. Ini dipakai untuk builder panel.")

        train_end_cfg3, test_start_cfg3, test_end_cfg3, win_start_cfg3 = _get_forecast_config_from_db()

        default_train_end = _to_month_start_date(train_end_cfg3) if train_end_cfg3 else _to_month_start_date(max_periode)
        if not win_start_cfg3:
            win_start_cfg3 = dt.date(2021, 1, 1)

        default_n_test = 5
        if test_start_cfg3 and test_end_cfg3:
            default_n_test = (
                (test_end_cfg3.year - test_start_cfg3.year) * 12
                + (test_end_cfg3.month - test_start_cfg3.month)
                + 1
            )
            default_n_test = int(np.clip(default_n_test, 1, 6))

        c1, c2, c3 = st.columns([1.15, 0.9, 1.15], gap="large")
        with c1:
            train_end_input = st.date_input(
                "Batas akhir data latih",
                value=default_train_end,
                min_value=min_periode,
                max_value=max_periode,
                key=f"{key_prefix}_train_end",
            )
        with c2:
            n_test_months = st.selectbox(
                "Lama periode uji (bulan)",
                options=[1, 2, 3, 4, 5, 6],
                index=[1, 2, 3, 4, 5, 6].index(default_n_test) if default_n_test in [1, 2, 3, 4, 5, 6] else 3,
                key=f"{key_prefix}_n_test",
            )
        with c3:
            win_start_input = st.date_input(
                "Mulai hitung penilaian dari",
                value=_to_month_start_date(win_start_cfg3),
                min_value=min_periode,
                max_value=_to_month_start_date(train_end_input),
                key=f"{key_prefix}_win_start",
            )

        train_end_ms = _to_month_start_date(train_end_input)
        win_start_ms = _to_month_start_date(win_start_input)

        test_start_ms = _add_month(train_end_ms)
        test_end_ms = test_start_ms
        for _ in range(int(n_test_months) - 1):
            test_end_ms = _add_month(test_end_ms)

        max_ms = _to_month_start_any(max_periode)
        if max_ms and test_end_ms > max_ms:
            test_end_ms = max_ms

        st.write("")
        if st.button("Simpan periode", type="primary", use_container_width=True, key=f"{key_prefix}_save_period"):
            _upsert_forecast_config(
                {
                    "TRAIN_END": train_end_ms.isoformat(),
                    "TEST_START": test_start_ms.isoformat(),
                    "TEST_END": test_end_ms.isoformat(),
                    "WIN_START": win_start_ms.isoformat(),
                },
                user_id=user_id,
            )
            st.success("Periode tersimpan.")
            st.cache_data.clear()
            st.rerun()

    st.write("")
    train_end, test_start, test_end, win_start = _get_forecast_config_from_db()
    n_clusters_cfg = _get_n_clusters_from_db(default=4)
    selection_rule_now = _get_eligibility_rule_from_db()

    def _iso(d):
        return d.isoformat() if d else "-"

    summary_text = (
        f"Latih sampai {_iso(train_end)}. "
        f"Uji {_iso(test_start)} sampai {_iso(test_end)}. "
        f"Kelompok produk {int(n_clusters_cfg) if n_clusters_cfg is not None else '-'}. "
        f"Produk yang dipakai hanya yang statusnya layak."
    )

    with st.container(border=True):
        st.markdown("#### Ringkasan yang dipakai")
        st.write(summary_text)
        st.write("Aturan seleksi yang dipakai")
        st.write(_summarize_rule(selection_rule_now))

    if (train_end is None) or (test_start is None) or (test_end is None):
        st.warning("Periode belum lengkap. Simpan periode dulu.")
        st.stop()

    def _run_training():
        with st.status("Proses berjalan...", expanded=True) as status:
            try:
                st.write("Ambil data")

                train_end_ts = pd.Timestamp(_to_month_start_date(train_end))
                test_start_ts = pd.Timestamp(_to_month_start_date(test_start))
                test_end_ts = pd.Timestamp(_to_month_start_date(test_end))
                win_start_ts = pd.Timestamp(_to_month_start_date(win_start)) if win_start else pd.Timestamp("2021-01-01")

                selection_rule = _get_eligibility_rule_from_db()
                rule_summary = _summarize_rule(selection_rule)

                df = build_lgbm_full_fullfeat_from_db(
                    engine,
                    train_end=train_end_ts,
                    test_start=test_start_ts,
                    test_end=test_end_ts,
                    win_start=win_start_ts,
                    selection_rule=selection_rule,
                )
                if df.empty:
                    status.update(label="Gagal", state="error")
                    st.error("Data pelatihan kosong. Cek periode atau aturan produk layak.")
                    return

                df = _norm_keys_df(df)

                if "qty" in df.columns:
                    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
                else:
                    status.update(label="Gagal", state="error")
                    st.error("Kolom qty tidak ditemukan dari builder.")
                    return

                df = df.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)

                clu_map = _load_cluster_map_from_db().drop_duplicates(subset=["cabang", "sku"], keep="last")
                if clu_map.empty:
                    status.update(label="Gagal", state="error")
                    st.error("Kelompok produk tidak ditemukan. Jalankan Update profil dulu.")
                    return

                df = df.merge(
                    clu_map[[c for c in ["cabang", "sku", "cluster", "demand_level"] if c in clu_map.columns]],
                    on=["cabang", "sku"],
                    how="left",
                )
                df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(int)

                if (df["cluster"] == -1).all():
                    status.update(label="Gagal", state="error")
                    st.error("Kelompok produk belum terisi. Jalankan Update profil dulu.")
                    return

                st.write("Latih model")

                df = add_hierarchy_features(df)
                df = add_stabilizer_features(df)
                df = winsorize_outliers(df)
                df = df.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)

                drop_cols = [
                    "area", "cabang", "sku", "periode", "qty", "qty_wins",
                    "is_train", "is_test", "sample_weight", "family",
                ]
                feature_cols = [c for c in df.columns if c not in drop_cols]

                run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                run_label = f"lgbm_{run_ts}"

                run_dir = MODEL_OUT_DIR / run_label
                model_dir = run_dir / "models"
                metric_dir = run_dir / "metrics"
                run_dir.mkdir(parents=True, exist_ok=True)
                model_dir.mkdir(parents=True, exist_ok=True)
                metric_dir.mkdir(parents=True, exist_ok=True)

                cluster_ids = sorted(df["cluster"].dropna().unique())
                models = {}

                for cid in cluster_ids:
                    if int(cid) == -1:
                        continue
                    model = train_lgbm_per_cluster(
                        df=df,
                        cluster_id=int(cid),
                        feature_cols=feature_cols,
                        log_target=True,
                        n_trials=40,
                    )
                    if model is None:
                        continue
                    models[int(cid)] = model
                    model.save_model(str(model_dir / f"lgbm_cluster_{int(cid)}.txt"))

                if not models:
                    status.update(label="Gagal", state="error")
                    st.error("Tidak ada model yang berhasil dilatih.")
                    return

                st.write("Simpan prediksi panel")

                df_pred_list = []
                for cid, model in models.items():
                    df_c = df[df["cluster"] == cid].copy()
                    if df_c.empty:
                        continue
                    X_c = df_c[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                    pred_log = model.predict(X_c)
                    df_c["pred_qty"] = np.expm1(pred_log)
                    df_c["pred_run_label"] = run_label
                    df_pred_list.append(df_c)

                df_pred = pd.concat(df_pred_list, axis=0).sort_values(["cabang", "sku", "periode"])
                pred_path = run_dir / "panel_with_predictions.csv"
                df_pred.to_csv(pred_path, index=False)

                train_mask = (df_pred["is_train"] == 1)
                test_mask = (df_pred["is_test"] == 1)

                train_obs_mask = (
                    train_mask
                    & (df_pred["qty"].notna())
                    & (df_pred["qty"] > 0)
                    & (df_pred["pred_qty"].notna())
                )
                test_obs_mask = (
                    test_mask
                    & (df_pred["qty"].notna())
                    & (df_pred["qty"] > 0)
                    & (df_pred["pred_qty"].notna())
                )

                n_train_points = int(train_obs_mask.sum())
                n_test_points = int(test_obs_mask.sum())

                train_metrics = _metric_pack(
                    df_pred.loc[train_obs_mask, "qty"].values,
                    df_pred.loc[train_obs_mask, "pred_qty"].values
                ) if n_train_points > 0 else _metric_pack([], [])

                test_metrics = _metric_pack(
                    df_pred.loc[test_obs_mask, "qty"].values,
                    df_pred.loc[test_obs_mask, "pred_qty"].values
                ) if n_test_points > 0 else _metric_pack([], [])

                global_train_mae = train_metrics["mae"]
                global_train_mse = train_metrics["mse"]
                global_train_rmse = train_metrics["rmse"]
                global_train_mape = train_metrics["mape"]

                global_test_mae = test_metrics["mae"]
                global_test_mse = test_metrics["mse"]
                global_test_rmse = test_metrics["rmse"]
                global_test_mape = test_metrics["mape"]

                train_start_dt = None
                train_end_dt = None
                test_start_dt = None
                test_end_dt = None

                if train_mask.any():
                    pmin = pd.to_datetime(df_pred.loc[train_mask, "periode"], errors="coerce").min()
                    pmax = pd.to_datetime(df_pred.loc[train_mask, "periode"], errors="coerce").max()
                    train_start_dt = pmin.date() if not pd.isna(pmin) else None
                    train_end_dt = pmax.date() if not pd.isna(pmax) else None

                if test_mask.any():
                    pmin = pd.to_datetime(df_pred.loc[test_mask, "periode"], errors="coerce").min()
                    pmax = pd.to_datetime(df_pred.loc[test_mask, "periode"], errors="coerce").max()
                    test_start_dt = pmin.date() if not pd.isna(pmin) else None
                    test_end_dt = pmax.date() if not pd.isna(pmax) else None

                n_test_months_calc = (test_end.year - test_start.year) * 12 + (test_end.month - test_start.month) + 1
                n_clusters_cfg2 = _get_n_clusters_from_db(default=4)

                params_payload = {
                    "run_dir": str(run_dir),
                    "panel_path": str(pred_path),
                    "train_end": train_end.isoformat(),
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "n_test_months": int(n_test_months_calc),
                    "n_clusters": int(len(list(models.keys()))),
                    "selection_rule": selection_rule,
                    "selection_rule_summary": rule_summary,
                    "win_start": win_start.isoformat() if win_start else None,
                    "cluster_source": "sku_profile",
                    "n_clusters_config": int(n_clusters_cfg2),
                    "trainer": {"log_target": True, "n_trials": 40, "winsorize": True},
                    "eval_points": {
                        "n_train_points_obs": int(n_train_points),
                        "n_test_points_obs": int(n_test_points),
                        "rule_obs_qty_gt0": True,
                    },
                    "metrics_global": {
                        "train": train_metrics,
                        "test": test_metrics,
                        "note": "Metrik dihitung hanya pada titik qty > 0",
                    },
                }

                existing_cols = _get_existing_model_run_cols()

                base_cols = [
                    "model_type", "description", "trained_at", "trained_by",
                    "train_start", "train_end", "test_start", "test_end",
                    "n_test_months", "n_clusters", "params_json", "feature_cols_json",
                ]

                metric_candidates = [
                    ("global_train_rmse", global_train_rmse),
                    ("global_test_rmse", global_test_rmse),
                    ("global_train_mae", global_train_mae),
                    ("global_test_mae", global_test_mae),
                    ("global_train_mse", global_train_mse),
                    ("global_test_mse", global_test_mse),
                    ("global_train_mape", global_train_mape),
                    ("global_test_mape", global_test_mape),
                ]

                cols_to_insert = base_cols.copy()
                values_dict = {
                    "model_type": "lgbm",
                    "description": f"Pelatihan {train_start_dt}-{train_end_dt} · uji {test_start_dt}-{test_end_dt}",
                    "trained_at": dt.datetime.now(),
                    "trained_by": user_id,
                    "train_start": train_start_dt,
                    "train_end": train_end_dt,
                    "test_start": test_start_dt,
                    "test_end": test_end_dt,
                    "n_test_months": int(n_test_months_calc),
                    "n_clusters": int(len(list(models.keys()))),
                    "params_json": json.dumps(params_payload),
                    "feature_cols_json": json.dumps(feature_cols),
                }

                for col_name, col_val in metric_candidates:
                    if col_name in existing_cols:
                        cols_to_insert.append(col_name)
                        values_dict[col_name] = col_val

                col_list_sql = ", ".join(cols_to_insert)
                val_list_sql = ", ".join([f":{c}" for c in cols_to_insert])

                insert_sql = f"INSERT INTO model_run ({col_list_sql}) VALUES ({val_list_sql})"

                with engine.begin() as conn:
                    result = conn.execute(text(insert_sql), values_dict)
                    run_id = result.lastrowid

                snap_rows = _save_model_run_sku_snapshot(int(run_id), df_pred)
                sku_metric_rows = _save_model_run_sku_metrics(int(run_id), df_pred, test_obs_mask)

                status.update(label="Selesai", state="complete")
                st.success(
                    f"Pelatihan selesai. ID riwayat: {run_id}. "
                    f"Snapshot produk: {snap_rows:,} baris. "
                    f"Metrix per SKU tersimpan: {sku_metric_rows:,} baris. "
                    f"Titik uji (qty>0): {int(n_test_points):,}. "
                    f"MAPE uji: {(f'{global_test_mape:.2f}%' if global_test_mape is not None else '-')}"
                )
                st.cache_data.clear()

            except Exception as e:
                status.update(label="Gagal", state="error")
                st.error(f"Pelatihan gagal: {e}")

    mode = "run" if st.session_state["training_requested"] else "idle"
    key_reset = f"t3_{page_uid}_reset_{mode}"
    key_start = f"t3_{page_uid}_start_{mode}"

    with st.container(border=True):
        st.markdown("#### Mulai pelatihan")
        st.caption("Sistem hanya melatih produk yang statusnya layak dan menyimpan snapshot produk per model.")

        left, right = st.columns([1.0, 2.2], gap="large")
        with left:
            if st.button("Reset pelatihan", use_container_width=True, key=key_reset):
                st.session_state["training_running"] = False
                st.session_state["training_requested"] = False
                st.cache_data.clear()
                st.rerun()

        with right:
            st.info("Kalau tombol Mulai pelatihan mati padahal tidak ada proses, tekan Reset pelatihan.")

        clicked = st.button(
            "Mulai pelatihan",
            type="primary",
            use_container_width=True,
            key=key_start,
            disabled=bool(st.session_state["training_running"]),
        )

    if clicked:
        st.session_state["training_running"] = True
        st.session_state["training_requested"] = True
        st.rerun()

    if st.session_state["training_requested"]:
        try:
            _run_training()
        finally:
            st.session_state["training_running"] = False
            st.session_state["training_requested"] = False
            st.cache_data.clear()

else:
    st.markdown("### Riwayat model")

    models = get_all_model_runs()
    if not models:
        st.info("Belum ada riwayat pelatihan.")
        st.stop()

    df_models = pd.DataFrame(models)
    if "trained_at" in df_models.columns:
        df_models = df_models.sort_values("trained_at", ascending=False)
    df_models = df_models.reset_index(drop=True)

    df_models["status_display"] = np.where(df_models.get("active_flag", 0) == 1, "Aktif", "Arsip")

    display_cols = [
        "status_display",
        "id",
        "model_type",
        "n_test_months",
        "global_train_rmse",
        "global_test_rmse",
        "global_test_mape",
        "trained_at",
    ]
    display_cols = [c for c in display_cols if c in df_models.columns]
    df_display = df_models[display_cols].copy()

    n_rows = int(len(df_display))
    header_px = 38
    row_px = 30
    height_px = header_px + (n_rows * row_px)
    height_px = int(max(92, min(260, height_px)))

    selection = st.dataframe(
        df_display,
        hide_index=True,
        use_container_width=True,
        height=height_px,
        on_select="rerun",
        selection_mode="single-row",
    )

    if not selection.selection["rows"]:
        st.info("Pilih satu baris untuk lihat detail.")
        st.stop()

    selected_index = selection.selection["rows"][0]
    selected_row = df_models.iloc[selected_index]
    mid = int(selected_row.get("id"))

    params_obj = {}
    rule_obj = None
    eval_points = None
    metrics_obj = None
    try:
        params_raw = selected_row.get("params_json")
        params_obj = json.loads(params_raw) if isinstance(params_raw, str) and params_raw else {}
        rule_obj = params_obj.get("selection_rule")
        eval_points = params_obj.get("eval_points")
        metrics_obj = params_obj.get("metrics_global")
    except Exception:
        params_obj = {}
        rule_obj = None
        eval_points = None
        metrics_obj = None

    snap_cnt = 0
    try:
        with engine.begin() as conn:
            r = conn.execute(
                text("SELECT COUNT(*) AS n FROM model_run_sku WHERE model_run_id = :mid"),
                {"mid": mid},
            ).mappings().fetchone()
        snap_cnt = int(r["n"]) if r and r.get("n") is not None else 0
    except Exception:
        snap_cnt = 0

    def _yn(v) -> str:
        return "Ya" if bool(v) else "Tidak"

    def _rule_logistik(rule: dict) -> list[str]:
        if not isinstance(rule, dict) or not rule:
            return ["Aturan: -"]
        return [
            f"Minimal data: {int(rule.get('min_n_months', 0))} bulan",
            f"Minimal bulan ada jual: {int(rule.get('min_nonzero', 0))} bulan",
            f"Minimal total jual: {int(float(rule.get('min_total_qty', 0)))} unit",
            f"Maks bulan kosong: {int(round(float(rule.get('max_zero_ratio', 0)) * 100))}%",
            f"Maks stop jual: {int(rule.get('alive_recent_months', 0))} bulan",
            f"Ada data di bulan batas latih: {_yn(rule.get('require_last_month', True))}",
            f"Total 12 bulan terakhir > 0: {_yn(rule.get('require_qty_12m_gt0', True))}",
        ]

    def _fmt_metric(v, suffix=""):
        if v is None:
            return "-"
        try:
            return f"{float(v):,.2f}{suffix}"
        except Exception:
            return "-"

    mape_test_display = None
    try:
        if isinstance(metrics_obj, dict) and isinstance(metrics_obj.get("test"), dict):
            mape_test_display = metrics_obj["test"].get("mape")
    except Exception:
        mape_test_display = None

    with st.container(border=True):
        left, right = st.columns([3, 1], gap="large")

        with left:
            st.markdown(f"#### Detail model ID: {mid}")
            st.write(f"Latih: {selected_row.get('train_start')} sampai {selected_row.get('train_end')}")
            st.write(f"Uji: {selected_row.get('test_start')} sampai {selected_row.get('test_end')}")

            is_active = (selected_row.get("active_flag", 0) == 1)
            st.write(f"Status: {'aktif' if is_active else 'arsip'}")
            st.write(f"Jumlah SKU yang diforecast: {snap_cnt:,}")

            if isinstance(eval_points, dict):
                st.write(f"Titik evaluasi uji (qty>0): {int(eval_points.get('n_test_points_obs', 0)):,}")

            if "global_test_mape" in selected_row.index and selected_row.get("global_test_mape") is not None:
                st.write(f"MAPE uji: {_fmt_metric(selected_row.get('global_test_mape'), '%')}")
            else:
                st.write(f"MAPE uji: {_fmt_metric(mape_test_display, '%')}")

            st.write("Aturan produk yang masuk model:")
            for ln in (_rule_logistik(rule_obj) if rule_obj else ["Aturan: -"]):
                st.write(f"- {ln}")

        with right:
            st.write("")
            if not is_active:
                if st.button("Aktifkan", type="primary", use_container_width=True):
                    ok = activate_model(mid)
                    if ok:
                        st.toast("Model diaktifkan.", icon="✅")
                        st.rerun()
                    st.error("Gagal mengaktifkan model.")
            else:
                st.button("Sedang aktif", disabled=True, use_container_width=True)
