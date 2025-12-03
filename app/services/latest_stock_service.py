# app/services/latest_stock_service.py

import pandas as pd
from sqlalchemy import text
from app.db import engine


def build_latest_stock_from_transactions():
    """
    Ambil data transaksi dari sales_raw,
    lalu hitung stok terakhir per cabang+sku
    berdasarkan baris transaksi terakhir.
    Hasilnya di-upsert ke tabel latest_stock.
    """

    # 1. Load data transaksi dari DB
    sql = """
        SELECT
            id,
            cabang,
            sku,
            posting_date,
            stock
        FROM sales_raw
        WHERE stock IS NOT NULL
    """
    with engine.begin() as conn:
        df = pd.read_sql(sql, conn)

    if df.empty:
        print("Tidak ada data di sales_raw.")
        return 0

    # pastikan datetime
    df["posting_date"] = pd.to_datetime(df["posting_date"])

    # 2. Sort supaya tail(1) bener-bener baris terakhir
    df = df.sort_values(["cabang", "sku", "posting_date", "id"])

    # 3. Ambil baris terakhir per cabang+sku
    last_df = (
        df.groupby(["cabang", "sku"], as_index=False)
          .tail(1)[["cabang", "sku", "posting_date", "stock"]]
    )

    last_df = last_df.rename(
        columns={
            "posting_date": "last_txn_date",
            "stock": "last_stock"
        }
    )

    # 4. Upsert ke tabel latest_stock
    upsert_sql = text("""
        INSERT INTO latest_stock (
            cabang,
            sku,
            last_txn_date,
            last_stock
        )
        VALUES (
            :cabang,
            :sku,
            :last_txn_date,
            :last_stock
        )
        ON DUPLICATE KEY UPDATE
            last_txn_date = VALUES(last_txn_date),
            last_stock    = VALUES(last_stock);
    """)

    with engine.begin() as conn:
        for _, row in last_df.iterrows():
            conn.execute(
                upsert_sql,
                {
                    "cabang":        row["cabang"],
                    "sku":           row["sku"],
                    "last_txn_date": row["last_txn_date"].date(),
                    "last_stock":    float(row["last_stock"]),
                }
            )

    print(f"Selesai. Baris latest_stock ter-update: {len(last_df)}")
    return len(last_df)


if __name__ == "__main__":
    build_latest_stock_from_transactions()
