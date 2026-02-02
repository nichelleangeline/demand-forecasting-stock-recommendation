# app/services/stok_policy_service.py

from sqlalchemy import text
from app.db import engine


def ensure_stok_policy_table():
    create_sql = """
    CREATE TABLE IF NOT EXISTS stok_policy (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        cabang VARCHAR(10) NOT NULL,
        sku VARCHAR(100) NOT NULL,

        avg_qty DOUBLE NULL,
        max_lama DOUBLE NULL,
        index_lt DOUBLE NULL,
        proyeksi_max_baru DOUBLE NULL,
        growth DOUBLE NULL,
        max_baru DOUBLE NULL,

        updated_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP
            ON UPDATE CURRENT_TIMESTAMP,

        UNIQUE KEY uq_stok_policy_cs (cabang, sku)
    ) ENGINE=InnoDB;
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))


def upsert_stok_policy_from_sales_monthly(cabang=None) -> int:
    ensure_stok_policy_table()

    params = {}
    where_cabang = ""
    if cabang:
        where_cabang = " AND TRIM(UPPER(cabang)) = TRIM(UPPER(:cabang)) "
        params["cabang"] = cabang

   
    upsert_sql = f"""
    INSERT INTO stok_policy (
        cabang, sku,
        avg_qty, max_lama, index_lt,
        proyeksi_max_baru, growth, max_baru,
        updated_at
    )
    SELECT
        TRIM(UPPER(sm.cabang)) AS cabang,
        TRIM(UPPER(sm.sku))    AS sku,

        AVG(sm.qty)            AS avg_qty,
        MAX(sm.qty)            AS max_lama,

        1.0                    AS index_lt,

        (MAX(sm.qty) * 
            (
                CASE
                    WHEN AVG(CASE
                              WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 12 MONTH)
                              THEN sm.qty END) IS NULL
                         OR ABS(AVG(CASE
                              WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 12 MONTH)
                              THEN sm.qty END)) < 1e-8
                    THEN 1.0
                    ELSE
                        AVG(CASE
                              WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 3 MONTH)
                              THEN sm.qty END)
                        /
                        AVG(CASE
                              WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 12 MONTH)
                              THEN sm.qty END)
                END
            )
        ) AS proyeksi_max_baru,

        (
            CASE
                WHEN AVG(CASE
                          WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 12 MONTH)
                          THEN sm.qty END) IS NULL
                     OR ABS(AVG(CASE
                          WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 12 MONTH)
                          THEN sm.qty END)) < 1e-8
                THEN 1.0
                ELSE
                    AVG(CASE
                          WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 3 MONTH)
                          THEN sm.qty END)
                    /
                    AVG(CASE
                          WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 12 MONTH)
                          THEN sm.qty END)
            END
        ) AS growth,

        (MAX(sm.qty) *
            (
                CASE
                    WHEN AVG(CASE
                              WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 12 MONTH)
                              THEN sm.qty END) IS NULL
                         OR ABS(AVG(CASE
                              WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 12 MONTH)
                              THEN sm.qty END)) < 1e-8
                    THEN 1.0
                    ELSE
                        AVG(CASE
                              WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 3 MONTH)
                              THEN sm.qty END)
                        /
                        AVG(CASE
                              WHEN sm.periode >= DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 12 MONTH)
                              THEN sm.qty END)
                END
            )
        ) AS max_baru,

        NOW() AS updated_at
    FROM sales_monthly sm
    WHERE 1=1
      {where_cabang}
    GROUP BY TRIM(UPPER(sm.cabang)), TRIM(UPPER(sm.sku))
    ON DUPLICATE KEY UPDATE
        avg_qty           = VALUES(avg_qty),
        max_lama          = VALUES(max_lama),
        index_lt          = VALUES(index_lt),
        proyeksi_max_baru = VALUES(proyeksi_max_baru),
        growth            = VALUES(growth),
        max_baru          = VALUES(max_baru),
        updated_at        = NOW();
    """

    with engine.begin() as conn:
        res = conn.execute(text(upsert_sql), params)
    return int(res.rowcount or 0)
