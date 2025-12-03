import numpy as np
import pandas as pd

from app.features.hierarchy_features import add_hierarchy_features
from app.features.stabilizer_features import add_stabilizer_features
from app.features.outlier_handler import winsorize_outliers


def apply_feature_pipeline(df):
    df = df.copy()

    df["is_train"] = 1
    df = add_hierarchy_features(df)
    df = add_stabilizer_features(df)
    df = winsorize_outliers(df)

    df["log_qty"] = np.log1p(df["qty"])
    df["log_qty_wins"] = np.log1p(df["qty_wins"])

    df = add_all_lags_and_rollings(df)
    return df
