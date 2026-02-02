CREATE DATABASE IF NOT EXISTS tktw_db
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'tktw_app'@'127.0.0.1' IDENTIFIED BY 'tktw123';
ALTER USER 'tktw_app'@'127.0.0.1' IDENTIFIED BY 'tktw123';

GRANT
    SELECT, INSERT, UPDATE, DELETE,
    CREATE, DROP, INDEX, ALTER,
    LOCK TABLES, EXECUTE, CREATE VIEW
ON tktw_db.* TO 'tktw_app'@'127.0.0.1';

FLUSH PRIVILEGES;

USE tktw_db;


CREATE TABLE IF NOT EXISTS user_account (
    user_id        INT AUTO_INCREMENT PRIMARY KEY,
    full_name      VARCHAR(150) NOT NULL,
    email          VARCHAR(150) NOT NULL UNIQUE,
    password_hash  VARCHAR(255) NOT NULL,
    role           ENUM('admin','user') NOT NULL DEFAULT 'user',
    is_active      TINYINT(1) DEFAULT 1,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;


USE tktw_db;



CREATE TABLE IF NOT EXISTS sales_monthly (
id BIGINT AUTO_INCREMENT PRIMARY KEY,
area VARCHAR(10) NOT NULL,
cabang VARCHAR(10) NOT NULL,
sku VARCHAR(100) NOT NULL,
periode DATE NOT NULL,
qty DOUBLE NOT NULL,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
UNIQUE KEY uq_sales_monthly (cabang, sku, periode)
)

CREATE TABLE IF NOT EXISTS external_data (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    area VARCHAR(10) NOT NULL,
    cabang VARCHAR(10) NOT NULL,
    periode DATE NOT NULL,
    event_flag TINYINT NOT NULL DEFAULT 0,
    holiday_count INT NOT NULL DEFAULT 0,
    rainfall DOUBLE NOT NULL DEFAULT 0,
    source_filename VARCHAR(255),
    uploaded_by INT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_external (cabang, periode)
);

CREATE TABLE IF NOT EXISTS sku_profile (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    cabang VARCHAR(10) NOT NULL,
    sku VARCHAR(100) NOT NULL,
    n_months INT NOT NULL,
    qty_mean DOUBLE NOT NULL,
    qty_std DOUBLE,
    qty_max DOUBLE,
    qty_min DOUBLE,
    total_qty DOUBLE NOT NULL,
    zero_months INT NOT NULL,
    zero_ratio DOUBLE,
    cv DOUBLE,
    demand_level TINYINT, -- 0,1,2,3 dari qcut
    cluster INT, -- -1,0,1,2,3 dsb
    eligible_model TINYINT(1) NOT NULL DEFAULT 1,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_sku_profile (cabang, sku),
    KEY idx_sku_profile_cluster (cluster),
    KEY idx_sku_profile_eligible (eligible_model)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_run (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_type VARCHAR(100) NOT NULL, -- misal: 'lgbm_full_clusters_tweedie_noleak'
    description TEXT,
    trained_at DATETIME NOT NULL,
    trained_by INT, -- user_id admin
    train_start DATE NOT NULL,
    train_end DATE NOT NULL,
    test_start DATE NULL,
    test_end DATE NULL,
    n_test_months INT DEFAULT 0,
    n_clusters INT NOT NULL,
    params_json JSON NULL, -- hyperparam global / info tambahan
    feature_cols_json JSON NULL, -- daftar fitur yang dipakai waktu training
    global_train_rmse DOUBLE,
    global_test_rmse DOUBLE,
    global_train_mae DOUBLE,
    global_test_mae DOUBLE,
    active_flag TINYINT(1) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_model_run_type (model_type),
    KEY idx_model_run_active (active_flag),
    KEY idx_model_run_trained_at (trained_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_run_cluster (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_run_id BIGINT NOT NULL,
    cluster_id INT NOT NULL,
    model_path VARCHAR(255) NOT NULL, -- path ke file .txt LGBM
    train_rmse DOUBLE,
    test_rmse DOUBLE,
    train_mae DOUBLE,
    test_mae DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_model_run_cluster (model_run_id, cluster_id),
    KEY idx_mrc_model_run (model_run_id),
    CONSTRAINT fk_mrc_model_run
    FOREIGN KEY (model_run_id) REFERENCES model_run(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS latest_stock (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,

    area VARCHAR(10) NOT NULL,
    cabang VARCHAR(10) NOT NULL,
    sku VARCHAR(100) NOT NULL,

    last_txn_date DATE NOT NULL,   -- tanggal transaksi terakhir yang kelihatan di file
    last_stock DOUBLE NOT NULL,    -- stok setelah transaksi terakhir itu

    source_filename VARCHAR(255),  -- nama file upload terakhir yang menyentuh stok SKU ini
    uploaded_by INT,               -- user_id yang upload
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
               ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uq_latest_stock (cabang, sku),
    KEY idx_latest_stock_area (area),
    KEY idx_latest_stock_cabang (cabang),
    KEY idx_latest_stock_sku (sku)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS forecast_config (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL,
    config_value VARCHAR(255) NOT NULL,
    description VARCHAR(255),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    updated_by INT,
    UNIQUE KEY uq_forecast_config_key (config_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;



CREATE TABLE IF NOT EXISTS panel_global_monthly (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    cabang        VARCHAR(10) NOT NULL,
    area          VARCHAR(10) NOT NULL,
    sku           VARCHAR(30) NOT NULL,
    periode       DATE NOT NULL,

    qty           INT NOT NULL,
    imputed_flag  TINYINT(1) DEFAULT 0,

    -- EXOG
    event_flag    INT DEFAULT 0,
    holiday_count INT DEFAULT 0,
    rainfall      FLOAT DEFAULT 0,

    -- GLOBAL STATS PER cabang,sku (SUPAYA SAMA DENGAN MODEL LAMA)
    imputed       TINYINT(1) DEFAULT 0,

    eligible_model TINYINT(1) DEFAULT 1,

    last_updated  DATETIME DEFAULT CURRENT_TIMESTAMP
                           ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uniq_pg (cabang, sku, periode),
    INDEX idx_pg_cs (cabang, sku),
    INDEX idx_pg_periode (periode)
) ENGINE=InnoDB;


CREATE TABLE IF NOT EXISTS series_info (
    id BIGINT NOT NULL AUTO_INCREMENT,
    cabang VARCHAR(10) NOT NULL,
    sku    VARCHAR(30) NOT NULL,

    -- ringkasan histori
    n_months         INT NOT NULL,        -- jumlah bulan dalam window seleksi
    nonzero_months   INT NOT NULL,        -- jumlah bulan qty > 0
    total_qty        FLOAT NOT NULL,      -- total penjualan sepanjang window
    qty_12m          FLOAT NOT NULL,      -- total 12 bulan terakhir
    qty_6m           FLOAT NOT NULL,      -- total 6 bulan terakhir

    zero_ratio_train FLOAT NOT NULL,      -- proporsi bulan zero di periode train
    n_train          INT NOT NULL,        -- jumlah bulan train (is_train==1)

    last_nz              DATE NULL,       -- tanggal periode terakhir qty > 0 di train
    months_since_last_nz INT NOT NULL,    -- selisih bulan dari TRAIN_END ke last_nz
    alive_recent         TINYINT(1) NOT NULL DEFAULT 0,   -- masih hidup menjelang akhir window

    has_train_end   TINYINT(1) NOT NULL DEFAULT 0,   -- ADA data di bulan TRAIN_END
    eligible_model  TINYINT(1) NOT NULL DEFAULT 0,   -- 1 = lolos kriteria global

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
               ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uniq_series (cabang, sku),
    KEY idx_series_cabang (cabang)
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci;

ALTER TABLE user_account
  ADD COLUMN reset_code_used TINYINT(1) NOT NULL DEFAULT 0
  AFTER reset_expiry;

ALTER TABLE user_account
  ADD COLUMN access_all_cabang TINYINT(1) NOT NULL DEFAULT 0
  AFTER must_change_password;



CREATE TABLE stok_policy (
    id           BIGINT PRIMARY KEY AUTO_INCREMENT,
    cabang       VARCHAR(10),
    sku          VARCHAR(100),

    avg_qty      DOUBLE,   
    max_lama     DOUBLE,   
    index_lt     DOUBLE,   

    proyeksi_max_baru DOUBLE,  
    growth            DOUBLE,  
    max_baru          DOUBLE  
);

