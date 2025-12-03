# app/services/model_service.py

from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import json

import lightgbm as lgb
from sqlalchemy import text

from app.db import engine


def get_all_model_runs() -> List[Dict]:
    """
    Ambil daftar semua model_run untuk admin & dashboard.
    Pastikan include params_json & feature_cols_json.
    """
    sql = """
        SELECT
            id,
            model_type,
            description,
            trained_at,
            trained_by,
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
            global_test_mae,
            active_flag
        FROM model_run
        ORDER BY trained_at DESC
    """
    # pakai .mappings() biar setiap row bisa langsung jadi dict(r)
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()

    return [dict(r) for r in rows]


def get_active_model_run() -> Optional[Dict]:
    """
    Ambil 1 model_run aktif:
    - kalau ada active_flag = 1 → ambil yang trained_at paling baru
    - kalau tidak ada → ambil yang trained_at paling baru overall
    """
    runs = get_all_model_runs()
    if not runs:
        return None

    active = [r for r in runs if r.get("active_flag") == 1]
    if active:
        active_sorted = sorted(
            active,
            key=lambda r: r.get("trained_at") or datetime.min,
        )
        return active_sorted[-1]

    runs_sorted = sorted(
        runs,
        key=lambda r: r.get("trained_at") or datetime.min,
    )
    return runs_sorted[-1]


def activate_model(model_run_id: int) -> bool:
    """
    Set model_run_id ini jadi aktif.
    - Semua model_run lain active_flag = 0
    - Yang dipilih active_flag = 1
    """
    with engine.begin() as conn:
        # matikan semua model aktif
        conn.execute(
            text("UPDATE model_run SET active_flag = 0 WHERE active_flag = 1")
        )

        # aktifkan yang dipilih
        result = conn.execute(
            text(
                """
                UPDATE model_run
                SET active_flag = 1, updated_at = :updated_at
                WHERE id = :id
                """
            ),
            {
                "id": model_run_id,
                "updated_at": datetime.now(),
            },
        )

    return result.rowcount > 0


def _parse_params_json(raw) -> Dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def load_cluster_models(run: Dict) -> Dict[int, lgb.Booster]:
    """
    Load semua model cluster LightGBM untuk satu model_run.

    Asumsi struktur:
      params_json['run_dir'] / 'models' / 'lgbm_full_cluster_{cid}.txt'
    """
    params = _parse_params_json(run.get("params_json"))

    # 1) cari run_dir dari params_json
    run_dir_str = params.get("run_dir")

    # 2) fallback: kalau tidak ada, scan outputs/<model_type>/subdir terbaru
    if not run_dir_str:
        project_root = Path(__file__).resolve().parents[2]  # .../demand-forecasting
        base_out = project_root / "outputs" / run["model_type"]
        if base_out.exists():
            subdirs = [p for p in base_out.iterdir() if p.is_dir()]
            if subdirs:
                latest = sorted(subdirs, key=lambda p: p.stat().st_mtime)[-1]
                run_dir_str = str(latest)

    if not run_dir_str:
        raise ValueError(
            "Tidak bisa menentukan run_dir dari params_json / folder outputs."
        )

    run_dir = Path(run_dir_str)
    model_dir = run_dir / "models"

    # jumlah cluster dari kolom n_clusters atau params_json
    n_clusters = run.get("n_clusters") or params.get("n_clusters") or 4
    n_clusters = int(n_clusters)

    models: Dict[int, lgb.Booster] = {}
    for cid in range(n_clusters):
        model_path = model_dir / f"lgbm_full_cluster_{cid}.txt"
        if not model_path.exists():
            # cluster ini bisa saja kosong, skip saja
            continue
        models[cid] = lgb.Booster(model_file=str(model_path))

    if not models:
        raise ValueError(f"Tidak ada file model cluster di {model_dir}")

    return models
