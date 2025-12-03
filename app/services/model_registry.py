# app/services/model_registry.py

import json
from typing import Dict, Any, Optional

from sqlalchemy import text

from app.db import engine


# =========================
# Helper: ambil run aktif
# =========================

def _get_active_run() -> Optional[Dict[str, Any]]:
    """
    Ambil 1 baris model_run yang active_flag = 1, paling baru (trained_at desc).
    Kolom disesuaikan dengan schema model_run versi DB kamu.
    """
    sql = """
        SELECT
            id,
            model_type,
            description,
            trained_at,
            train_start,
            train_end,
            test_start,
            test_end,
            n_test_months,
            n_clusters,
            params_json,
            feature_cols_json,
            global_train_rmse,
            global_test_rmse,
            global_train_mae,
            global_test_mae
        FROM model_run
        WHERE active_flag = 1
        ORDER BY trained_at DESC
        LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql)).mappings().first()

    if not row:
        return None

    return dict(row)


# =========================
# Public: load_active_models
# =========================

def load_active_models() -> Optional[Dict[str, Any]]:
    """
    Dipakai di forecast_page:
    - ambil info run aktif (model_run)
    - ambil daftar model per cluster (model_run_cluster)
    - parse feature_cols_json

    Return:
        {
          "run_info": {...},
          "cluster_models": {
              0: {
                  "model_path": "...",
                  "train_rmse": ...,
                  "test_rmse": ...
              },
              ...
          },
          "feature_cols": [...string kolom fitur...]  # kalau ada
        }
    atau None kalau belum ada model aktif.
    """
    run = _get_active_run()
    if not run:
        return None

    run_id = run["id"]

    # Parse feature_cols_json (kalau diisi)
    feature_cols = None
    if run.get("feature_cols_json"):
        try:
            feature_cols = json.loads(run["feature_cols_json"])
        except Exception:
            feature_cols = None

    # Ambil mapping cluster -> model_path dari model_run_cluster
    sql_clusters = """
        SELECT
            cluster_id,
            model_path,
            train_rmse,
            test_rmse
        FROM model_run_cluster
        WHERE model_run_id = :run_id
        ORDER BY cluster_id
    """

    cluster_models: Dict[int, Dict[str, Any]] = {}
    with engine.connect() as conn:
        rows = conn.execute(text(sql_clusters), {"run_id": run_id}).mappings().all()

    for r in rows:
        cid = int(r["cluster_id"])
        cluster_models[cid] = {
            "model_path": r["model_path"],
            "train_rmse": r.get("train_rmse"),
            "test_rmse": r.get("test_rmse"),
        }

    if not cluster_models:
        # Secara logika harusnya ada, tapi kalau belum ke-insert ya anggap belum siap
        return None

    return {
        "run_info": run,
        "cluster_models": cluster_models,
        "feature_cols": feature_cols,
    }
