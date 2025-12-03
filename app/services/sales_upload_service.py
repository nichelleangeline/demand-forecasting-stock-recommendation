# app/services/sales_upload_service.py

from typing import Literal
import pandas as pd
import numpy as np
from sqlalchemy import text

from app.db import engine

AREA_MAP = {
    "02A": "1A",
    "05A": "1B",
    "13A": "02",
    "13I": "3",
    "14A": "04",
    "16C": "05",
    "17A": "08",
    "23A": "07",
    "29A": "06",
}


def _preprocess_daily_to_monthly(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : df_raw hasil read_excel / read_csv dari file upload harian.
    Output : DataFrame bulanan [area, cabang, sku, periode, qty]
             qty sudah bersih sebagai demand positif (logic NETTO).
    """

    rename_map = {
        "posting date": "date",
        "postingdate": "date",
        "posting_date": "date",

        "location code": "cabang_raw",
        "locationcode": "cabang_raw",
        "location_code": "cabang_raw",

        "sku": "sku",
        "item no.": "sku",

        "quantity": "qty_raw",
        "qty": "qty_raw",
    }

    # normalisasi nama kolom jadi lowercase dulu
    cols_lower = {c: c.lower().strip() for c in df_raw.columns}
    df_raw = df_raw.rename(columns=cols_lower)
    df = df_raw.rename(columns={c: rename_map.get(c, c) for c in df_raw.columns})

    required = ["date", "cabang_raw", "sku", "qty_raw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang di data upload: {missing}")

    # qty numeric
    df["qty"] = pd.to_numeric(df["qty_raw"], errors="coerce")
    df = df.dropna(subset=["qty"])

    # cabang dan area
    df["cabang"] = (
        df["cabang_raw"]
        .astype(str)
        .str.split("-")
        .str[0]
        .str.strip()
        .str.upper()
    )
    df["area"] = df["cabang"].map(AREA_MAP).fillna(df["cabang"])

    # tanggal ke periode bulanan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["periode"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # agregasi NETTO per bulan (shipment + retur)
    g = (
        df.groupby(["periode", "area", "cabang", "sku"], as_index=False)
          .agg(qty_net=("qty", "sum"))
    )

    # qty positif pakai logika pipeline offline:
    #   - qty_net = -40 -> demand 40
    #   - qty_net = +10 -> demand 0
    g["qty"] = -g["qty_net"]
    g.loc[g["qty"] < 0, "qty"] = 0.0
    g = g.drop(columns=["qty_net"])

    g = g.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)
    return g[["area", "cabang", "sku", "periode", "qty"]]


def _preprocess_monthly_direct(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Untuk file yang memang sudah bulanan:
    ekspektasi ada kolom: periode, cabang, sku, qty
    area akan diisi dari cabang (AREA_MAP) kalau belum ada.
    """
    cols_lower = {c: c.lower().strip() for c in df_raw.columns}
    df = df_raw.rename(columns=cols_lower)

    rename_map = {
        "periode": "periode",
        "period": "periode",
        "bulan": "periode",

        "cabang": "cabang",
        "location code": "cabang",
        "locationcode": "cabang",
        "location_code": "cabang",

        "sku": "sku",
        "item no.": "sku",

        "qty": "qty",
        "quantity": "qty",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required = ["periode", "cabang", "sku", "qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang di data bulanan: {missing}")

    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df.dropna(subset=["qty"])

    df["cabang"] = (
        df["cabang"]
        .astype(str)
        .str.split("-")
        .str[0]
        .str.strip()
        .str.upper()
    )
    df["area"] = df["cabang"].map(AREA_MAP).fillna(df["cabang"])

    df["periode"] = pd.to_datetime(df["periode"], errors="coerce")
    df = df.dropna(subset=["periode"])
    df["periode"] = df["periode"].dt.to_period("M").dt.to_timestamp()

    df = df.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)
    return df[["area", "cabang", "sku", "periode", "qty"]]


def preprocess_sales_df(
    df_raw: pd.DataFrame,
    mode: Literal["harian", "bulanan"] = "harian",
) -> pd.DataFrame:
    """
    Wrapper yang dipanggil dari halaman Streamlit.
    - mode='harian'  -> pakai _preprocess_daily_to_monthly
    - mode='bulanan' -> pakai _preprocess_monthly_direct
    Output selalu: [area, cabang, sku, periode, qty]
    """
    if mode == "harian":
        return _preprocess_daily_to_monthly(df_raw)
    elif mode == "bulanan":
        return _preprocess_monthly_direct(df_raw)
    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")


def save_sales_monthly_to_db(df_monthly: pd.DataFrame,
                             source_filename: str,
                             uploaded_by: int) -> int:
    """
    Simpan ke sales_monthly.
    - Kalau (cabang, sku, periode) sudah ada -> qty diganti dengan yang baru.
    - Kalau belum ada -> insert baris baru.
    """

    if df_monthly.empty:
        return 0

    df = df_monthly.copy()

    # standar: strip & upper supaya key konsisten
    df["area"] = df["area"].astype(str).str.strip().str.upper()
    df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    df["sku"] = df["sku"].astype(str).str.strip().str.upper()

    df["periode"] = pd.to_datetime(df["periode"]).dt.date
    df["qty"] = df["qty"].astype(float)

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "area": row["area"],
                "cabang": row["cabang"],
                "sku": row["sku"],
                "periode": row["periode"],
                "qty": row["qty"],
            }
        )

    sql = """
        INSERT INTO sales_monthly
            (area, cabang, sku, periode, qty)
        VALUES
            (:area, :cabang, :sku, :periode, :qty)
        ON DUPLICATE KEY UPDATE
            qty = VALUES(qty)
    """

    with engine.begin() as conn:
        conn.execute(text(sql), records)

    return len(records)


# =====================================================
# Tambahan: ambil stok terakhir dari file HARIAN
# =====================================================

def _extract_latest_stock_from_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Dari file harian NAV, ambil stok terakhir per area-cabang-sku.
    Asumsi ada kolom:
      - Posting Date / posting date / posting_date
      - Location Code / location code
      - SKU / Item No.
      - Stock
    Output: [area, cabang, sku, last_txn_date, last_stock]
    """

    cols_lower = {c: c.lower().strip() for c in df_raw.columns}
    df = df_raw.rename(columns=cols_lower)

    rename_map = {
        "posting date": "date",
        "postingdate": "date",
        "posting_date": "date",

        "location code": "cabang_raw",
        "locationcode": "cabang_raw",
        "location_code": "cabang_raw",

        "sku": "sku",
        "item no.": "sku",

        "stock": "stock_raw",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required = ["date", "cabang_raw", "sku", "stock_raw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Kalau tidak ada kolom stock, anggap saja tidak bisa ambil latest stock
        return pd.DataFrame(columns=["area", "cabang", "sku", "last_txn_date", "last_stock"])

    df["stock_raw"] = pd.to_numeric(df["stock_raw"], errors="coerce")
    df = df.dropna(subset=["stock_raw"])

    df["cabang"] = (
        df["cabang_raw"]
        .astype(str)
        .str.split("-")
        .str[0]
        .str.strip()
        .str.upper()
    )
    df["area"] = df["cabang"].map(AREA_MAP).fillna(df["cabang"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df.sort_values(["area", "cabang", "sku", "date"])

    last_df = (
        df.groupby(["area", "cabang", "sku"], as_index=False)
          .tail(1)[["area", "cabang", "sku", "date", "stock_raw"]]
    )

    last_df = last_df.rename(
        columns={
            "date": "last_txn_date",
            "stock_raw": "last_stock",
        }
    )

    return last_df[["area", "cabang", "sku", "last_txn_date", "last_stock"]]


def preprocess_sales_and_stock_df(
    df_raw: pd.DataFrame,
    mode: Literal["harian", "bulanan"] = "harian",
):
    """
    Versi kombo:
      - Kembalikan df_monthly (seperti preprocess_sales_df)
      - DAN df_latest_stock (kalau mode='harian' & ada kolom stock)
    """

    if mode == "harian":
        df_monthly = _preprocess_daily_to_monthly(df_raw)
        df_latest = _extract_latest_stock_from_daily(df_raw)
        return df_monthly, df_latest
    elif mode == "bulanan":
        df_monthly = _preprocess_monthly_direct(df_raw)
        # File bulanan biasanya tidak punya kolom stock
        df_latest = pd.DataFrame(columns=["area", "cabang", "sku", "last_txn_date", "last_stock"])
        return df_monthly, df_latest
    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")


def save_latest_stock_to_db(df_latest: pd.DataFrame,
                            source_filename: str,
                            uploaded_by: int) -> int:
    """
    Simpan stok terakhir per cabang+sku ke tabel latest_stock.
    Struktur tabel (contoh):

    CREATE TABLE IF NOT EXISTS latest_stock (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        area VARCHAR(10) NOT NULL,
        cabang VARCHAR(10) NOT NULL,
        sku VARCHAR(100) NOT NULL,
        last_txn_date DATE NOT NULL,
        last_stock DOUBLE NOT NULL,
        source_filename VARCHAR(255),
        uploaded_by INT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                     ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_latest_stock (cabang, sku)
    );
    """

    if df_latest.empty:
        return 0

    df = df_latest.copy()

    df["area"] = df["area"].astype(str).str.strip().str.upper()
    df["cabang"] = df["cabang"].astype(str).str.strip().str.upper()
    df["sku"] = df["sku"].astype(str).str.strip().str.upper()
    df["last_txn_date"] = pd.to_datetime(df["last_txn_date"]).dt.date
    df["last_stock"] = df["last_stock"].astype(float)

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "area": row["area"],
                "cabang": row["cabang"],
                "sku": row["sku"],
                "last_txn_date": row["last_txn_date"],
                "last_stock": row["last_stock"],
                "source_filename": source_filename,
                "uploaded_by": uploaded_by,
            }
        )

    sql = """
        INSERT INTO latest_stock (
            area, cabang, sku,
            last_txn_date, last_stock,
            source_filename, uploaded_by
        )
        VALUES (
            :area, :cabang, :sku,
            :last_txn_date, :last_stock,
            :source_filename, :uploaded_by
        )
        ON DUPLICATE KEY UPDATE
            last_txn_date   = VALUES(last_txn_date),
            last_stock      = VALUES(last_stock),
            source_filename = VALUES(source_filename),
            uploaded_by     = VALUES(uploaded_by),
            updated_at      = CURRENT_TIMESTAMP
    """

    with engine.begin() as conn:
        conn.execute(text(sql), records)

    return len(records)
