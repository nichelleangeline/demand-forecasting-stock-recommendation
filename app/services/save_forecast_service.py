from sqlalchemy import text

def save_future_to_db(engine, df_future, model_run_id):
    sql = """
        INSERT INTO forecast_future
        (model_run_id, area, cabang, sku, periode, pred_qty)
        VALUES (:mid, :area, :cabang, :sku, :periode, :pred)
    """

    with engine.begin() as conn:
        for _, r in df_future.iterrows():
            conn.execute(
                text(sql),
                {
                    "mid": model_run_id,
                    "area": r["area"],
                    "cabang": r["cabang"],
                    "sku": r["sku"],
                    "periode": r["periode"].date(),
                    "pred": float(r["pred_qty"]),
                }
            )
