# app/register_model_lgbm.py

from pathlib import Path
from sqlalchemy import text
import pandas as pd

from app.db import engine

PROJECT_ROOT = Path(r"D:\Documents\Skripsi\demand-forecasting")
OUT_DIR = PROJECT_ROOT / "outputs" / "lgbm_full_global_tweedie"

model_path     = OUT_DIR / "lgbm_full_global_tweedie_model.txt"
featcols_path  = OUT_DIR / "lgbm_full_global_tweedie_feature_cols.txt"
global_metrics = OUT_DIR / "lgbm_full_global_tweedie_global_metrics.csv"


def main():
    # baca global metrics hasil training terbaru
    gm = pd.read_csv(global_metrics)
    row = gm.iloc[0]

    train_start = row["train_start_date"]
    train_end   = row["train_end_date"]

    with engine.begin() as conn:
        # non-aktifkan semua model lama
        conn.execute(text("UPDATE model_registry SET is_active = 0"))

        # daftarkan model baru
        conn.execute(
            text("""
            INSERT INTO model_registry
            (model_name, model_type,
             model_file_path, feature_cols_path,
             train_start, train_end,
             created_by, is_active)
            VALUES
            (:model_name, :model_type,
             :model_file_path, :feature_cols_path,
             :train_start, :train_end,
             :created_by, :is_active)
            """),
            {
                "model_name": "LGBM_FULL_GLOBAL_TWEEDIE_V1",   # versi kamu sekarang
                "model_type": "LGBM",
                "model_file_path": str(model_path),
                "feature_cols_path": str(featcols_path),
                "train_start": train_start,
                "train_end": train_end,
                "created_by": 1,   # admin user_id = 1
                "is_active": 1,
            },
        )

    print("Model registered ke model_registry.")


if __name__ == "__main__":
    main()
