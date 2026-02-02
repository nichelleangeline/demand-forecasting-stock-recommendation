import pandas as pd
from sqlalchemy import text
from app.db import engine

def load_dashboard_data():
    sql = text("""
        SELECT 
            fc.cabang,
            fc.sku,
            fc.periode,
            fc.qty_forecast,
            ls.stock_akhir,
            cc.lead_time_bulan
        FROM forecast_result fc
        LEFT JOIN latest_stock ls
            ON fc.cabang = ls.cabang AND fc.sku = ls.sku
        LEFT JOIN config_cabang cc
            ON fc.cabang = cc.cabang
        ORDER BY fc.cabang, fc.sku, fc.periode
    """)

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    df["periode"] = pd.to_datetime(df["periode"])
    return df

def compute_safety_stock(df):
    out_rows = []

    for (cab, sku), g in df.groupby(["cabang", "sku"]):
        g = g.sort_values("periode")

        # ambil 6 bulan terakhir
        last6 = g.tail(6)

        if len(last6) < 6:
            continue  # skip series pendek

        qty_vals = last6["qty_forecast"].values
        avg_6 = qty_vals.mean()
        max_6 = qty_vals.max()

        stock_akhir = g["stock_akhir"].iloc[-1] if "stock_akhir" in g else 0
        lead_time = g["lead_time_bulan"].iloc[-1] or 1.0

        # safety stock resmi
        needed = (max_6 * lead_time) - (avg_6 * lead_time)
        safety_stock = max(0, round(needed))

        reorder_qty = max(0, safety_stock - stock_akhir)

        out_rows.append({
            "cabang": cab,
            "sku": sku,
            "avg_6": avg_6,
            "max_6": max_6,
            "lead_time": lead_time,
            "safety_stock": safety_stock,
            "stock_akhir": stock_akhir,
            "reorder_qty": reorder_qty,
            "status": status_label(safety_stock, stock_akhir)
        })

    return pd.DataFrame(out_rows)


def status_label(safety_stock, stock_akhir):
    if stock_akhir < safety_stock:
        return "Risiko Stok Kosong"
    elif stock_akhir > (safety_stock * 1.5):
        return "Kelebihan Stok"
    return "Normal"

def compute_kpi(df_forecast, df_safety):
    total_pred = df_forecast["qty_forecast"].sum()

    over = df_safety[df_safety["status"] == "Kelebihan Stok"]
    under = df_safety[df_safety["status"] == "Risiko Stok Kosong"]

    return {
        "total_forecast": round(total_pred, 2),
        "count_overstock": len(over),
        "count_stockout": len(under),
        "avg_safety_stock": df_safety["safety_stock"].mean().round(2),
    }

def get_dashboard_metrics():
    df = load_dashboard_data()
    safety = compute_safety_stock(df)
    kpi = compute_kpi(df, safety)

    return {
        "forecast_df": df,
        "safety_df": safety,
        "kpi": kpi
    }
