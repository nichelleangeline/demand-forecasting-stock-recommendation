# app/features/hierarchy_features.py

import pandas as pd

def add_hierarchy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def sku_family(s):
        s = str(s).upper().strip()

        if s.endswith("CHAR"):
            return "char"
        if s.endswith("CPOX"):
            return "cpox"
        if s.endswith("CSW"):
            return "csw"
        if s.endswith("CSB") or "KSB" in s:
            return "csb"

        # fallback: prefix 4 huruf
        return s[:4]

    df["family"] = df["sku"].apply(sku_family)
    return df
