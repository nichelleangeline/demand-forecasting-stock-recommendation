import pandas as pd

def add_exog_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cabang","sku","periode"]).copy()

    if "event_flag" in df.columns:
        df["event_flag_lag1"] = (
            df.groupby(["cabang","sku"])["event_flag"].shift(1)
        )
    else:
        df["event_flag_lag1"] = 0

    if "holiday_count" in df.columns:
        df["holiday_count_lag1"] = (
            df.groupby(["cabang","sku"])["holiday_count"].shift(1)
        )
    else:
        df["holiday_count_lag1"] = 0

    return df
