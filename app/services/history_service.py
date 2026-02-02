from typing import Dict
import pandas as pd
from sqlalchemy import text


def load_history(engine, cabang: str, sku: str) -> pd.DataFrame:
    sql = text(
        """
        SELECT
            s.area,
            s.cabang,
            s.sku,
            s.periode,
            s.qty,
            COALESCE(e.event_flag, 0)      AS event_flag,
            COALESCE(e.holiday_count, 0)   AS holiday_count,
            COALESCE(e.rainfall, 0.0)      AS rainfall
        FROM sales_monthly s
        LEFT JOIN external_data e
          ON s.cabang = e.cabang
         AND s.periode = e.periode
        WHERE s.cabang = :cabang
          AND s.sku    = :sku
        ORDER BY s.periode
        """
    )

    with engine.begin() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={"cabang": cabang, "sku": sku},
        )

    if df.empty:
        return df

    df["periode"] = pd.to_datetime(df["periode"])
    df = df.sort_values("periode").reset_index(drop=True)
    return df
