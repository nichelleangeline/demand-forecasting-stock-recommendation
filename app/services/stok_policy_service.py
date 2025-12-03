# app/services/stok_policy_service.py

from typing import Optional

import pandas as pd
from sqlalchemy import text
from app.db import engine

TABLE_NAME = "stok_policy"

# Index lead time per cabang
INDEX_LT_MAP = {
    "02A": 1.48,  # 1A 02A
    "05A": 1.64,  # 1B 05A
    "13A": 1.04,  # 2  13A
    "13I": 1.04,  # 3  13I
    "14A": 1.08,  # 4  14A
    "16C": 1.04,  # 5  16C
    "29A": 1.48,  # 6  29A
    "23A": 1.40,  # 7  23A
    "17A": 1.16,  # 8  17A
}


def build_stok_policy_from_sales_monthly(cabang: Optional[str] = None) -> None:
    """
    Seed / update stok_policy dari tabel sales_monthly dengan logika:

    - Data penjualan bulanan per (cabang, sku)
    - Policy dihitung tiap 3 bulan (bulan ke-3, 6, 9, 12)
    - Di tiap titik policy:
        window 6 bulan terakhir -> avg_6, proyeksi_max_baru
        growth = proyeksi / max_lama_sebelumnya
        max_baru:
          - growth = 0   -> 2
          - growth < 2   -> proyeksi_max_baru
          - growth >= 2  -> 2 * max_lama_sebelumnya

    - Untuk periode terakhir (last_period):
        avg_qty   = rata-rata 6 bulan terakhir sampai last_period
        max_lama  = max_baru dari policy 3 bulanan terakhir sebelum/tepat last_period
    """

    base_sql = """
        SELECT cabang, sku, periode, qty
        FROM sales_monthly
        WHERE 1=1
    """
    params: dict = {}
    if cabang:
        base_sql += " AND cabang = :cabang"
        params["cabang"] = cabang

    with engine.connect() as conn:
        df = pd.read_sql(
            text(base_sql),
            conn,
            params=params,
            parse_dates=["periode"],
        )

    if df.empty:
        return

    df = df.dropna(subset=["cabang", "sku", "periode", "qty"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df.dropna(subset=["qty"])

    # Normalisasi ke awal bulan
    df["periode"] = pd.to_datetime(df["periode"]).dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["cabang", "sku", "periode"])

    records: list[dict] = []

    for (cab, sku), g in df.groupby(["cabang", "sku"]):
        # Agregasi qty per bulan
        g_month = (
            g.groupby("periode", as_index=False)["qty"]
             .sum()
             .sort_values("periode")
             .reset_index(drop=True)
        )

        if g_month.empty:
            continue

        index_lt = INDEX_LT_MAP.get(cab, 1.0)

        periods = g_month["periode"].tolist()
        qtys = g_month["qty"].tolist()
        n = len(g_month)

        prev_max: Optional[float] = None  # MAX BARU terakhir (jadi max_lama berikutnya)

        # Simulasi policy tiap 3 bulan
        for i in range(n):
            p = periods[i]
            month = p.month

            # Hanya bulan ke-3,6,9,12
            if month not in (3, 6, 9, 12):
                continue

            # Ambil 6 bulan terakhir sampai p
            start_idx = max(0, i - 5)
            window = qtys[start_idx : i + 1]
            if len(window) < 6:
                # belum cukup 6 bulan histori
                continue

            avg_6 = sum(window) / len(window)
            proy = avg_6 * index_lt

            if prev_max is None or prev_max == 0:
                growth = 0
                max_baru = 2.0
            else:
                growth = proy / prev_max if prev_max != 0 else 0
                if growth == 0:
                    max_baru = 2.0
                elif growth < 2:
                    max_baru = proy
                else:
                    max_baru = 2.0 * prev_max

            prev_max = max_baru

        # State di periode terakhir (misal Mei 2024)
        last_period = periods[-1]
        last_idx = n - 1
        start_last = max(0, last_idx - 5)
        window_last = qtys[start_last : last_idx + 1]

        avg_last = sum(window_last) / len(window_last) if window_last else 0.0
        max_lama_final = prev_max if prev_max is not None else 0.0

        records.append(
            {
                "cabang": cab,
                "sku": sku,
                "avg_qty": float(avg_last),
                "max_lama": float(max_lama_final),
                "index_lt": float(index_lt),
            }
        )

    if not records:
        return

    upsert_sql = f"""
    INSERT INTO {TABLE_NAME} (
        cabang,
        sku,
        avg_qty,
        max_lama,
        index_lt
    )
    VALUES (
        :cabang,
        :sku,
        :avg_qty,
        :max_lama,
        :index_lt
    )
    ON DUPLICATE KEY UPDATE
        avg_qty  = VALUES(avg_qty),
        max_lama = VALUES(max_lama),
        index_lt = VALUES(index_lt);
    """

    with engine.begin() as conn:
        conn.execute(text(upsert_sql), records)


def recompute_stok_policy(cabang: Optional[str] = None) -> None:
    """
    Recompute stok policy di tabel stok_policy.

    Rumus:
      proyeksi_max_baru = avg_qty * index_lt

      growth =
        0 jika max_lama NULL / 0
        (proyeksi_max_baru / max_lama) jika max_lama > 0

      max_baru:
        - growth = 0      -> 2
        - growth < 2      -> proyeksi_max_baru
        - growth >= 2     -> 2 * max_lama

    Jika `cabang` diisi, hanya baris untuk cabang tersebut yang dihitung ulang.
    Kalau None, semua cabang akan di-update.
    """

    where_clause = ""
    params: dict = {}

    if cabang:
        where_clause = "WHERE t.cabang = :cabang"
        params["cabang"] = cabang

    # STEP 1: hitung proyeksi_max_baru dan growth
    sql_step1 = f"""
    UPDATE {TABLE_NAME} AS t
    SET
      proyeksi_max_baru = t.avg_qty * t.index_lt,
      growth = CASE
                 WHEN t.max_lama IS NULL OR t.max_lama = 0
                   THEN 0
                 ELSE (t.avg_qty * t.index_lt) / t.max_lama
               END
    {where_clause};
    """

    # STEP 2: hitung max_baru pakai aturan growth
    sql_step2 = f"""
    UPDATE {TABLE_NAME} AS t
    SET
      max_baru = CASE
                   WHEN t.growth = 0
                     THEN 2
                   WHEN t.growth < 2
                     THEN t.proyeksi_max_baru
                   ELSE 2 * t.max_lama
                 END
    {where_clause};
    """

    with engine.begin() as conn:
        conn.execute(text(sql_step1), params)
        conn.execute(text(sql_step2), params)


if __name__ == "__main__":
    # Contoh jalan manual:
    # python -m app.services.stok_policy_service
    build_stok_policy_from_sales_monthly()
    recompute_stok_policy()
    print("Build + recompute stok policy selesai.")
