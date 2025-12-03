# app/services/safety_stock_service.py

from datetime import date
from typing import Optional

import pandas as pd
from sqlalchemy import text

from app.db import engine

# =========================
# LEAD TIME & SALES TOOLS
# (ini anggap sudah kamu punya, kalau belum ya tambahkan)
# =========================

def load_leadtime_df() -> pd.DataFrame:
    """
    Ambil tabel leadtime_index (cabang, index_value).
    """
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT cabang, index_value FROM leadtime_index"),
            conn,
        )
    return df


def load_sales_last_6_months(
    as_of: Optional[pd.Timestamp] = None,
    area_filter: Optional[list[str]] = None,
    cabang_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    if as_of is None:
        as_of = pd.Timestamp(date.today())

    start_date = (as_of - pd.DateOffset(months=6)).replace(day=1)

    sql = """
        SELECT area, cabang, sku, periode, qty
        FROM sales_monthly
        WHERE periode >= :start_date
          AND periode <= :as_of
    """
    params: dict = {"start_date": start_date, "as_of": as_of}

    if area_filter:
        sql += " AND area IN :areas"
        params["areas"] = tuple(area_filter)

    if cabang_filter:
        sql += " AND cabang IN :cabangs"
        params["cabangs"] = tuple(cabang_filter)

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params, parse_dates=["periode"])

    return df


def compute_avg_6m(df_sales: pd.DataFrame) -> pd.DataFrame:
    if df_sales.empty:
        return pd.DataFrame(columns=["area", "cabang", "sku", "avg_qty_6m"])

    df = df_sales.sort_values(["cabang", "sku", "periode"]).copy()

    last6 = (
        df.groupby(["area", "cabang", "sku"], as_index=False)
          .tail(6)
    )

    agg = (
        last6.groupby(["area", "cabang", "sku"], as_index=False)
             .agg(avg_qty_6m=("qty", "mean"))
    )
    return agg

# =========================
# SAFETY STOCK SNAPSHOT
# =========================

def ensure_safety_stock_snapshot_table():
    """
    Pastikan tabel safety_stock_snapshot ada.
    Pakai definisi yang kamu kirim.
    """
    create_sql = """
    CREATE TABLE IF NOT EXISTS safety_stock_snapshot (
        id               BIGINT AUTO_INCREMENT PRIMARY KEY,
        cabang           VARCHAR(10) NOT NULL,
        sku              VARCHAR(30) NOT NULL,
        periode_forecast DATE,
        forecast_qty     FLOAT,
        avg_6            FLOAT,
        max_6            FLOAT,
        safety_stock     INT,
        stok_akhir       INT,
        reorder_qty      INT,
        status           VARCHAR(50),
        generated_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
        generated_by     INT,
        FOREIGN KEY (generated_by) REFERENCES user_account(user_id),
        INDEX idx_ss_cs (cabang, sku)
    ) ENGINE=InnoDB;
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))


def load_max_lama_from_history(
    as_of: pd.Timestamp,
    area_filter: list[str] | None = None,
    cabang_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    max_lama diambil dari history:
    max_lama = max_6 dari periode_forecast terakhir < as_of
    di tabel safety_stock_snapshot.
    """
    ensure_safety_stock_snapshot_table()

    base_sql = """
    SELECT s.cabang, s.sku, s.periode_forecast, s.max_6 AS max_lama
    FROM safety_stock_snapshot s
    JOIN (
        SELECT cabang, sku, MAX(periode_forecast) AS last_periode
        FROM safety_stock_snapshot
        WHERE periode_forecast < :as_of
        GROUP BY cabang, sku
    ) x
      ON s.cabang = x.cabang
     AND s.sku    = x.sku
     AND s.periode_forecast = x.last_periode
    WHERE 1=1
    """
    params: dict = {"as_of": as_of.date()}

    # area tidak ada di tabel snapshot, jadi filter area dilakukan nanti lewat join ke avg_6
    if cabang_filter:
        base_sql += " AND s.cabang IN :cabangs"
        params["cabangs"] = tuple(cabang_filter)

    with engine.connect() as conn:
        df = pd.read_sql(text(base_sql), conn)

    return df


def growth_rule(max_lama: float, proyeksi_max_baru: float) -> float:
    """
    Placeholder growth: ganti dengan rule Excel kamu nanti.
    """
    if max_lama is None or pd.isna(max_lama) or max_lama == 0:
        return 1.0
    return (proyeksi_max_baru - max_lama) / max_lama


def decide_new_max(max_lama: float, proyeksi_max_baru: float, growth: float) -> float:
    """
    Placeholder penentuan max baru: ganti logic sesuai rule merah.
    """
    if pd.isna(max_lama) or max_lama == 0:
        return proyeksi_max_baru

    if growth >= 0.2:
        return proyeksi_max_baru
    elif growth <= -0.2:
        return (max_lama + proyeksi_max_baru) / 2
    else:
        return max_lama


def compute_safety_stock(
    as_of: pd.Timestamp,
    area_filter: list[str] | None = None,
    cabang_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Hitung:
      avg_6   = avg qty 6 bulan terakhir
      max_6   = max baru (setelah growth rule)
      safety_stock = 0.8 * max_6

    max_lama diambil dari history tabel safety_stock_snapshot (kolom max_6).
    """

    # 1. sales 6 bulan terakhir
    df_sales = load_sales_last_6_months(
        as_of=as_of,
        area_filter=area_filter,
        cabang_filter=cabang_filter,
    )

    df_avg = compute_avg_6m(df_sales)
    if df_avg.empty:
        return pd.DataFrame(
            columns=[
                "area",
                "cabang",
                "sku",
                "avg_qty_6m",
                "leadtime_index",
                "proyeksi_max_baru",
                "max_lama",
                "growth",
                "max_baru",
                "safety_stock",
            ]
        )

    # 2. lead time index
    df_lead = load_leadtime_df()
    df_lead = df_lead.rename(columns={"index_value": "leadtime_index"})

    merged = df_avg.merge(df_lead, on="cabang", how="left")
    merged = merged[~merged["leadtime_index"].isna()].copy()

    # 3. proyeksi max baru
    merged["proyeksi_max_baru"] = merged["avg_qty_6m"] * merged["leadtime_index"]

    # 4. max_lama dari safety_stock_snapshot.max_6
    df_hist = load_max_lama_from_history(
        as_of=as_of,
        area_filter=area_filter,
        cabang_filter=cabang_filter,
    )

    if not df_hist.empty:
        merged = merged.merge(
            df_hist[["cabang", "sku", "max_lama"]],
            on=["cabang", "sku"],
            how="left",
        )
    else:
        merged["max_lama"] = pd.NA

    # 5. growth + max_baru
    merged["growth"] = merged.apply(
        lambda r: growth_rule(r["max_lama"], r["proyeksi_max_baru"]),
        axis=1,
    )
    merged["max_baru"] = merged.apply(
        lambda r: decide_new_max(r["max_lama"], r["proyeksi_max_baru"], r["growth"]),
        axis=1,
    )

    # 6. safety stock = 80% x max_baru
    merged["safety_stock"] = 0.8 * merged["max_baru"]

    merged = merged.sort_values(["area", "cabang", "sku"]).reset_index(drop=True)
    return merged


def update_safety_stock(
    as_of: pd.Timestamp,
    area_filter: list[str] | None = None,
    cabang_filter: list[str] | None = None,
    generated_by: int | None = None,
) -> pd.DataFrame:
    """
    Hitung safety stock dan simpan ke safety_stock_snapshot.
    Kolom di tabel:
      periode_forecast = as_of.date()
      avg_6            = avg_qty_6m
      max_6            = max_baru
      safety_stock     = dibulatkan ke int
    forecast_qty, stok_akhir, reorder_qty, status dibiarkan null/dummy.
    """
    ensure_safety_stock_snapshot_table()

    df_result = compute_safety_stock(
        as_of=as_of,
        area_filter=area_filter,
        cabang_filter=cabang_filter,
    )

    if df_result.empty:
        return df_result

    records = []
    for _, row in df_result.iterrows():
        records.append(
            {
                "cabang": row["cabang"],
                "sku": row["sku"],
                "periode_forecast": as_of.date(),
                "forecast_qty": None,  # nanti bisa diisi forecast beneran
                "avg_6": float(row["avg_qty_6m"]),
                "max_6": float(row["max_baru"]),
                "safety_stock": int(round(row["safety_stock"])),
                "stok_akhir": None,
                "reorder_qty": None,
                "status": None,
                "generated_by": generated_by,
            }
        )

    insert_sql = """
    INSERT INTO safety_stock_snapshot (
        cabang,
        sku,
        periode_forecast,
        forecast_qty,
        avg_6,
        max_6,
        safety_stock,
        stok_akhir,
        reorder_qty,
        status,
        generated_by
    )
    VALUES (
        :cabang,
        :sku,
        :periode_forecast,
        :forecast_qty,
        :avg_6,
        :max_6,
        :safety_stock,
        :stok_akhir,
        :reorder_qty,
        :status,
        :generated_by
    )
    """

    with engine.begin() as conn:
        conn.execute(text(insert_sql), records)

    return df_result
