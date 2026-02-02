from typing import List, Dict, Optional
from sqlalchemy import text
from app.db import engine


def get_all_cabang_config() -> List[Dict]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT
                    id,
                    cabang,
                    area,
                    lead_time_bulan,
                    min_months_hist,
                    min_nonzero_months,
                    min_total_qty,
                    max_zero_ratio,
                    is_active,
                    updated_at
                FROM config_cabang
                ORDER BY area, cabang
            """)
        ).mappings().all()
    return [dict(r) for r in rows]


def get_cabang_config_by_id(cfg_id: int) -> Optional[Dict]:
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT
                    id,
                    cabang,
                    area,
                    lead_time_bulan,
                    min_months_hist,
                    min_nonzero_months,
                    min_total_qty,
                    max_zero_ratio,
                    is_active,
                    updated_at
                FROM config_cabang
                WHERE id = :id
            """),
            {"id": cfg_id},
        ).mappings().first()
    return dict(row) if row else None


def create_cabang_config(
    cabang: str,
    area: str,
    lead_time_bulan: float = 1.0,
    min_months_hist: int = 36,
    min_nonzero_months: int = 10,
    min_total_qty: int = 30,
    max_zero_ratio: float = 0.7,
    is_active: bool = True,
    updated_by: Optional[int] = None,
) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO config_cabang (
                    cabang,
                    area,
                    lead_time_bulan,
                    min_months_hist,
                    min_nonzero_months,
                    min_total_qty,
                    max_zero_ratio,
                    is_active,
                    updated_by
                )
                VALUES (
                    :cabang,
                    :area,
                    :lt,
                    :min_hist,
                    :min_nz,
                    :min_qty,
                    :max_zero,
                    :active,
                    :upd
                )
            """),
            {
                "cabang": cabang,
                "area": area,
                "lt": float(lead_time_bulan),
                "min_hist": int(min_months_hist),
                "min_nz": int(min_nonzero_months),
                "min_qty": int(min_total_qty),
                "max_zero": float(max_zero_ratio),
                "active": 1 if is_active else 0,
                "upd": updated_by,
            },
        )
        return result.lastrowid


def update_cabang_config(
    cfg_id: int,
    area: str,
    lead_time_bulan: float,
    min_months_hist: int,
    min_nonzero_months: int,
    min_total_qty: int,
    max_zero_ratio: float,
    is_active: bool,
    updated_by: Optional[int] = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE config_cabang
                SET
                    area              = :area,
                    lead_time_bulan   = :lt,
                    min_months_hist   = :min_hist,
                    min_nonzero_months = :min_nz,
                    min_total_qty     = :min_qty,
                    max_zero_ratio    = :max_zero,
                    is_active         = :active,
                    updated_by        = :upd
                WHERE id = :id
            """),
            {
                "id": cfg_id,
                "area": area,
                "lt": float(lead_time_bulan),
                "min_hist": int(min_months_hist),
                "min_nz": int(min_nonzero_months),
                "min_qty": int(min_total_qty),
                "max_zero": float(max_zero_ratio),
                "active": 1 if is_active else 0,
                "upd": updated_by,
            },
        )
