from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from sqlalchemy import text
from app.db import engine


AREA_MAP = {
    "02A": "1A",
    "05A": "1B",
    "13A": "02",
    "13I": "3",
    "14A": "04",
    "16C": "05",
    "17A": "08",
    "23A": "07",
    "29A": "06",
}

def _to_month_start(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()


def _get_all_cabang_with_area() -> List[Dict[str, str]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT area, cabang FROM config_cabang ORDER BY cabang")
        ).fetchall()

        if not rows:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT area, cabang "
                    "FROM sales_monthly "
                    "ORDER BY cabang"
                )
            ).fetchall()

    return [{"area": r[0], "cabang": r[1]} for r in rows]


def _process_holiday_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # deteksi kolom periode
    if "periode" in df.columns:
        per_col = "periode"
    elif "bulan" in df.columns:
        per_col = "bulan"
    else:
        # fallback: kolom pertama
        per_col = df.columns[0]
        df = df.rename(columns={per_col: "periode"})
        per_col = "periode"

    # deteksi kolom jumlah libur
    if "holiday_count" in df.columns:
        cnt_col = "holiday_count"
    elif "hari_libur" in df.columns:
        cnt_col = "hari_libur"
    else:
        if len(df.columns) < 2:
            raise ValueError("File hari libur tidak punya kolom jumlah hari libur.")
        cnt_col = df.columns[1]

    df = df.rename(columns={per_col: "periode", cnt_col: "holiday_count"})

    df["periode"] = _to_month_start(df["periode"])
    df = df.dropna(subset=["periode"])

    df["holiday_count"] = (
        pd.to_numeric(df["holiday_count"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 31)
    )

    df = (
        df[["periode", "holiday_count"]]
        .drop_duplicates()
        .sort_values("periode")
        .reset_index(drop=True)
    )
    return df



def _process_event_df(df_raw: pd.DataFrame,
                      allowed_cabangs: List[Dict[str, str]]) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    date_candidates = {
        "gatering date", "gathering date", "periode", "period",
        "month", "bulan", "date", "tanggal"
    }
    date_col = next((c for c in df.columns if c in date_candidates), None) or df.columns[0]
    df = df.rename(columns={date_col: "periode"})

    value_cols = [c for c in df.columns if c != "periode"]
    tmp = df.melt(
        id_vars=["periode"],
        value_vars=value_cols,
        var_name="cabang_raw",
        value_name="event_flag",
    )

    tmp["periode"] = _to_month_start(tmp["periode"])
    tmp = tmp.dropna(subset=["periode"])

    tmp["cabang"] = (
        tmp["cabang_raw"].astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[-_].*$", "", regex=True)
    )

    tmp["event_flag"] = (
        pd.to_numeric(tmp["event_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )

    valid_cab_set = {c["cabang"] for c in allowed_cabangs}
    tmp = tmp[tmp["cabang"].isin(valid_cab_set)]

    ev = (
        tmp[["periode", "cabang", "event_flag"]]
        .drop_duplicates()
        .sort_values(["cabang", "periode"])
        .reset_index(drop=True)
    )
    return ev


def _process_rain_16c_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    rf = df_raw.copy()
    rf.columns = [str(c).strip().lower() for c in rf.columns]

    date_col = next((c for c in rf.columns if "tanggal" in c or "date" in c), rf.columns[0])
    rain_col = next((c for c in rf.columns if "rr" in c or "rain" in c), None)
    if rain_col is None:
        raise ValueError("Kolom curah hujan (RR / rain) tidak ditemukan di file.")

    rf = rf.rename(columns={date_col: "date", rain_col: "rain"})

    rf["date"] = pd.to_datetime(rf["date"], errors="coerce", dayfirst=True)
    if rf["date"].isna().all():
        rf["date"] = pd.to_datetime(
            pd.to_numeric(rf["date"], errors="coerce"),
            origin="1899-12-30",
            unit="D",
            errors="coerce",
        )

    rf = rf.dropna(subset=["date"])
    if rf.empty:
        raise ValueError("Tidak ada tanggal valid di file curah hujan.")

    rf["rain"] = pd.to_numeric(rf["rain"], errors="coerce")
    rf.loc[(rf["rain"] == 8888) | (rf["rain"] < 0), "rain"] = np.nan

    daily = (
        rf.groupby("date", as_index=False)["rain"]
        .mean()
        .sort_values("date")
    )
    if daily.empty:
        raise ValueError("File curah hujan tidak punya data valid setelah dibersihkan.")

    full_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    df1 = daily.set_index("date").reindex(full_days)
    df1.index.name = "date"

    df1["rain_interp"] = df1["rain"].interpolate(
        "linear", limit_direction="both"
    ).clip(lower=0)

    df1["rain"] = df1["rain_interp"]
    df1 = df1.drop(columns=["rain_interp"])

    d2 = df1.reset_index()
    d2["periode"] = d2["date"].dt.to_period("M").dt.to_timestamp()

    rain_16 = (
        d2.groupby("periode", as_index=False)["rain"]
        .sum()
        .rename(columns={"rain": "rainfall"})
        .sort_values("periode")
    )

    rain_16["cabang"] = "16C"
    rain_16 = rain_16[["periode", "cabang", "rainfall"]]
    return rain_16

def handle_external_upload(
    uploaded_file,
    jenis: str,
    uploaded_by: Optional[int] = None,
) -> Tuple[pd.DataFrame, int]:
    if uploaded_file is None:
        raise ValueError("File eksternal kosong.")

    filename = getattr(uploaded_file, "name", "uploaded_external")
    name_lower = filename.lower()

    if name_lower.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    allowed_cabangs = _get_all_cabang_with_area()
    cabang_area_map = {c["cabang"]: c["area"] for c in allowed_cabangs}

    jenis = jenis.lower().strip()

    if jenis == "holiday":
        hol = _process_holiday_df(df_raw)

        if not allowed_cabangs:
            raise ValueError("Tidak ada konfigurasi cabang/area di DB untuk menyebar holiday.")

        records = []
        for _, row in hol.iterrows():
            periode = pd.to_datetime(row["periode"]).date()
            hcnt = int(row["holiday_count"])
            for cabang, area in cabang_area_map.items():
                records.append(
                    {
                        "cabang": cabang,
                        "area": area,
                        "periode": periode,
                        "holiday_count": hcnt,
                        "source_filename": filename,
                        "uploaded_by": uploaded_by,
                    }
                )

        sql = """
            INSERT INTO external_data (
                cabang, area, periode,
                holiday_count,
                source_filename, uploaded_by
            )
            VALUES (
                :cabang, :area, :periode,
                :holiday_count,
                :source_filename, :uploaded_by
            )
            ON DUPLICATE KEY UPDATE
                holiday_count  = VALUES(holiday_count),
                source_filename = VALUES(source_filename),
                uploaded_by    = VALUES(uploaded_by),
                uploaded_at    = CURRENT_TIMESTAMP
        """

        with engine.begin() as conn:
            conn.execute(text(sql), records)

        df_preview = hol.copy()
        n_rows = len(records)

    elif jenis == "event":
        ev = _process_event_df(df_raw, allowed_cabangs)

        records = []
        for _, row in ev.iterrows():
            cabang = row["cabang"]
            periode = pd.to_datetime(row["periode"]).date()
            ev_flag = int(row["event_flag"])
            area = cabang_area_map.get(cabang, AREA_MAP.get(cabang, cabang))

            records.append(
                {
                    "cabang": cabang,
                    "area": area,
                    "periode": periode,
                    "event_flag": ev_flag,
                    "source_filename": filename,
                    "uploaded_by": uploaded_by,
                }
            )

        sql = """
            INSERT INTO external_data (
                cabang, area, periode,
                event_flag,
                source_filename, uploaded_by
            )
            VALUES (
                :cabang, :area, :periode,
                :event_flag,
                :source_filename, :uploaded_by
            )
            ON DUPLICATE KEY UPDATE
                event_flag     = VALUES(event_flag),
                source_filename = VALUES(source_filename),
                uploaded_by    = VALUES(uploaded_by),
                uploaded_at    = CURRENT_TIMESTAMP
        """

        with engine.begin() as conn:
            conn.execute(text(sql), records)

        df_preview = ev.copy()
        n_rows = len(records)

    elif jenis == "rainfall_16c":
        rain_16 = _process_rain_16c_df(df_raw)

        # pastikan 16C ada di config / sales
        if "16C" not in cabang_area_map:
            area_16 = AREA_MAP.get("16C", "05")
            cabang_area_map["16C"] = area_16

        area_16 = cabang_area_map["16C"]

        records = []
        for _, row in rain_16.iterrows():
            periode = pd.to_datetime(row["periode"]).date()
            rainfall = float(row["rainfall"])
            records.append(
                {
                    "cabang": "16C",
                    "area": area_16,
                    "periode": periode,
                    "rainfall": rainfall,
                    "source_filename": filename,
                    "uploaded_by": uploaded_by,
                }
            )

        sql = """
            INSERT INTO external_data (
                cabang, area, periode,
                rainfall,
                source_filename, uploaded_by
            )
            VALUES (
                :cabang, :area, :periode,
                :rainfall,
                :source_filename, :uploaded_by
            )
            ON DUPLICATE KEY UPDATE
                rainfall       = VALUES(rainfall),
                source_filename = VALUES(source_filename),
                uploaded_by    = VALUES(uploaded_by),
                uploaded_at    = CURRENT_TIMESTAMP
        """

        with engine.begin() as conn:
            conn.execute(text(sql), records)

        df_preview = rain_16.copy()
        n_rows = len(records)

    else:
        raise ValueError(f"Jenis data eksternal tidak dikenali: {jenis}")

    return df_preview, n_rows
