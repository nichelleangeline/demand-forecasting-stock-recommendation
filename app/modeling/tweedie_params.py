def get_tweedie_params():
    """
    Default parameter dasar untuk LGBM Tweedie.
    Params lain seperti learning_rate, num_leaves, dll
    diisi dari Optuna di train_lgbm_per_cluster.
    """
    return {
        "objective": "tweedie",
        "tweedie_variance_power": 1.25,  
        "metric": "rmse",
        "verbosity": -1,
        "force_row_wise": True,
        "seed": 1337,
    }
