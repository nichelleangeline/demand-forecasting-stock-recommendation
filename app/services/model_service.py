import json
from sqlalchemy import text
from app.db import engine


def get_all_model_runs() -> list[dict]:
    sql = """
        SELECT
            id,
            model_type,
            active_flag,
            trained_at,
            trained_by,
            n_test_months,
            n_clusters,
            global_train_rmse,
            global_test_rmse,
            global_train_mae,
            global_test_mae,
            params_json,
            feature_cols_json,
            description,
            train_start, train_end, test_start, test_end
        FROM model_run
        ORDER BY trained_at DESC, id DESC
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return [dict(r) for r in rows]


def get_active_model_run() -> dict | None:
    sql = """
        SELECT
            id,
            model_type,
            active_flag,
            trained_at,
            trained_by,
            n_test_months,
            n_clusters,
            global_train_rmse,
            global_test_rmse,
            global_train_mae,
            global_test_mae,
            params_json,
            feature_cols_json,
            description,
            train_start, train_end, test_start, test_end
        FROM model_run
        WHERE active_flag = 1
        ORDER BY trained_at DESC, id DESC
        LIMIT 1
    """
    with engine.begin() as conn:
        row = conn.execute(text(sql)).mappings().fetchone()
    return dict(row) if row else None


def activate_model(model_run_id: int) -> bool:
    mid = int(model_run_id)
    with engine.begin() as conn:
        conn.execute(text("UPDATE model_run SET active_flag = 0 WHERE active_flag = 1"))
        res = conn.execute(
            text("UPDATE model_run SET active_flag = 1 WHERE id = :mid"),
            {"mid": mid},
        )
        try:
            return int(res.rowcount) > 0
        except Exception:
            return True
