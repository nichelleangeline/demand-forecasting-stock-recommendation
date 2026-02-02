import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

from app.modeling.tweedie_params import get_tweedie_params


def train_lgbm_per_cluster(
    df: pd.DataFrame,
    cluster_id: int,
    feature_cols: list,
    log_target: bool = True,
    n_trials: int = 40,
):
 
    # Filter data per cluster
    df_c = df[df["cluster"] == cluster_id].copy()
    if df_c.empty:
        print("Cluster", cluster_id, "kosong. Skip.")
        return None

    df_c = df_c.sort_values(["cabang", "sku", "periode"]).reset_index(drop=True)

    # Target
    if log_target:
        if "qty_wins" not in df_c.columns:
            raise ValueError("Kolom 'qty_wins' tidak ditemukan di df.")
        df_c["tgt"] = np.log1p(df_c["qty_wins"])
    else:
        df_c["tgt"] = df_c["qty_wins"]

    # Inner train/val: hanya is_train == 1
    train_all = df_c[df_c["is_train"] == 1].copy()
    val_cutoff = pd.Timestamp("2024-02-01")

    train_inner = train_all[train_all["periode"] < val_cutoff]
    val_inner   = train_all[train_all["periode"] >= val_cutoff]

    if train_inner.empty or val_inner.empty:
        print(f"Cluster {cluster_id}: inner train/val kosong. Skip.")
        return None

    X_train = train_inner[feature_cols]
    X_val   = val_inner[feature_cols]

    y_train = train_inner["tgt"].values
    y_val   = val_inner["tgt"].values

    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def objective(trial):
        params = get_tweedie_params()
        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 12),
        })

        train_set = lgb.Dataset(X_train, y_train)
        val_set   = lgb.Dataset(X_val, y_val)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[train_set, val_set],
            valid_names=["train","val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False)
            ],
        )

        pred_val = model.predict(X_val, num_iteration=model.best_iteration)

        if log_target:
            pred_val = np.expm1(pred_val)

        true_val = val_inner["qty"].values.astype(float)
        rmse_val = np.sqrt(np.mean((true_val - pred_val)**2))
        return rmse_val

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=1337))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_iter = study.best_trial.user_attrs.get("best_iteration", 200)
    best_params = study.best_params

    final_params = get_tweedie_params()
    final_params.update(best_params)

    train_full = df_c[df_c["is_train"] == 1].copy()
    X_full = train_full[feature_cols]
    X_full = X_full.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_full = train_full["tgt"].values

    model = lgb.train(
        final_params,
        lgb.Dataset(X_full, y_full),
        num_boost_round=best_iter,
    )

    return model
