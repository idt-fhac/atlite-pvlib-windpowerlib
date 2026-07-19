import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
import zipfile
import csv

# ---------------------------------------------------------------------------
# Package layout (paths resolved from this file — cwd-independent)
# ---------------------------------------------------------------------------
#   structured_research/
#     data/      — parquet input cache
#     cutouts/   — atlite ERA5 cutouts (*.nc)
#     results/   — CSV / PNG study outputs
#     *.py       — study scripts
#
PACKAGE_DIR = Path(__file__).resolve().parent
PAPER_ROOT = PACKAGE_DIR.parent  # repository root (contains run_all.py)
# External archives (e.g. Kelmarsh SCADA zip): prefer repo root, then parent.
WORKSPACE_ROOT = PAPER_ROOT
DATA_DIR = PACKAGE_DIR / "data"
CUTOUTS_DIR = PACKAGE_DIR / "cutouts"
RESULTS_DIR = PACKAGE_DIR / "results"

# Add package dir so `import lib` / `import utils` work from any cwd
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
# Add paper root for optional infrastructure imports
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))


def ensure_results_dir() -> Path:
    """Create and return the canonical results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def cutout_path(filename: str) -> Path:
    """Absolute path to an atlite cutout under structured_research/cutouts/."""
    return CUTOUTS_DIR / filename


def result_path(filename: str) -> Path:
    """Absolute path to a study artifact under structured_research/results/."""
    return RESULTS_DIR / filename


def offline_mode() -> bool:
    """
    True when OFFLINE_MODE=1/true — refuse all DB connections and require
    local parquet/cutout caches (for reproducible cache-only runs).
    """
    return os.getenv("OFFLINE_MODE", "").lower() in ("1", "true", "yes")


def resolve_engine(uri_type: str = "oeds"):
    """Return a DB engine, or None in offline mode (cache-only)."""
    if offline_mode():
        return None
    return get_db_engine(uri_type)


def metrics_by_season(sim, act, include_full_year: bool = True) -> pd.DataFrame:
    """
    Compute calculate_metrics() for each meteorological season (+ optional Full Year).
    """
    sim = pd.Series(sim).copy()
    act = pd.Series(act).copy()
    if not isinstance(sim.index, pd.DatetimeIndex):
        sim.index = pd.to_datetime(sim.index)
    if not isinstance(act.index, pd.DatetimeIndex):
        act.index = pd.to_datetime(act.index)
    if getattr(sim.index, "tz", None) is not None:
        sim.index = sim.index.tz_localize(None)
    if getattr(act.index, "tz", None) is not None:
        act.index = act.index.tz_localize(None)

    rows = []
    for code, months in SEASONS.items():
        s = sim[sim.index.month.isin(months)]
        a = act[act.index.month.isin(months)]
        m = calculate_metrics(s, a)
        rows.append({
            "Season": SEASON_LABELS[code],
            "SeasonCode": code,
            "Ratio_%": round(m["ratio"], 2),
            "Correlation": round(m["corr"], 4) if m["corr"] == m["corr"] else float("nan"),
            "MAE": round(m["mae"], 3),
            "RMSE": round(m["rmse"], 3),
            "Sim_Sum": round(m["sim_sum"], 3),
            "Act_Sum": round(m["act_sum"], 3),
            "n": m["n"],
        })
    if include_full_year:
        m = calculate_metrics(sim, act)
        rows.append({
            "Season": "Full Year",
            "SeasonCode": "FULL",
            "Ratio_%": round(m["ratio"], 2),
            "Correlation": round(m["corr"], 4) if m["corr"] == m["corr"] else float("nan"),
            "MAE": round(m["mae"], 3),
            "RMSE": round(m["rmse"], 3),
            "Sim_Sum": round(m["sim_sum"], 3),
            "Act_Sum": round(m["act_sum"], 3),
            "n": m["n"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Meteorological season definitions (months)
# ---------------------------------------------------------------------------
SEASONS = {
    "DJF": [12, 1, 2],   # Winter
    "MAM": [3, 4, 5],    # Spring
    "JJA": [6, 7, 8],    # Summer
    "SON": [9, 10, 11],  # Autumn
}
SEASON_LABELS = {
    "DJF": "Winter (DJF)",
    "MAM": "Spring (MAM)",
    "JJA": "Summer (JJA)",
    "SON": "Autumn (SON)",
}

# Solar orientation mapping codes
mastr_solar_azimuth = {
    'Ost': 90, 'Süd': 180, 'West': 270, 'Nord': 0,
    'Nord-Ost': 45, 'Süd-Ost': 135, 'Süd-West': 225, 'Nord-West': 315,
    'Ost-West': 180, 'nachgeführt': 180, 'None': 180, 'NaN': 180,
    'nan': 180
}
mastr_solar_tilt = {
    'Unter 20 Grad': 15, '21 - 40 Grad': 30, '41 - 60 Grad': 50,
    'Über 60 Grad': 70, 'Fassadenanlage': 90, 'None': 30, 'NaN': 30,
    'nan': 30
}

def get_db_engine(uri_type='oeds'):
    """
    Returns a SQLAlchemy engine for the specified database type.
    'oeds' for the open-energy-data-server, 'timescale' for regional/actual data.

    Raises RuntimeError when OFFLINE_MODE is set — use resolve_engine() in scripts
    that support cache-only execution.
    """
    if offline_mode():
        raise RuntimeError(
            f"OFFLINE_MODE is active: refusing database connection ({uri_type}). "
            "Use local parquet/cutout caches under structured_research/data/ and cutouts/."
        )
    if uri_type == 'oeds':
        x = os.getenv("INFRASTRUCTURE_SOURCE", "10.26.4.45:6432/opendata")
        y = os.getenv("INFRASTRUCTURE_LOGIN", "readonly:readonly")
        uri = f"postgresql://{y}@{x}"
    elif uri_type == 'timescale':
        uri = os.getenv("TIMESCALE_URI", "postgresql://readonly:readonly@extern.idt.fh-aachen.de:5433/opendata")
    else:
        raise ValueError(f"Unknown uri_type: {uri_type}")
    return sqlalchemy.create_engine(uri)

def load_local_data(filename):
    """
    Helper function to load data from the local Parquet directory if available.
    """
    if os.getenv("BYPASS_LOCAL_DATA", "False").lower() in ("true", "1"):
        raise FileNotFoundError("Local data bypass is active (BYPASS_LOCAL_DATA is set)")
    path = DATA_DIR / filename
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Local research data file not found: {path}")


def load_local_data_season(basename: str, season_code: str | None = None):
    """
    Load a season-specific parquet file if season_code is given (e.g. 'DJF'),
    otherwise load the full-year file (basename ending with '_2023') or the
    legacy file (bare basename).

    File naming convention produced by extract_research_data.py:
      {basename}_2023.parquet        — full year 2023
      {basename}_{season_code}.parquet — seasonal slice (DJF/MAM/JJA/SON)
      {basename}.parquet             — legacy January-only (backward compat)

    Note: callers must still slice to the requested analysis window via
    ``slice_time_index`` / query helpers — full-year files are preferred when present.
    """
    if season_code is not None:
        if season_code not in SEASONS:
            raise ValueError(f"Unknown season_code '{season_code}'. Use one of: {list(SEASONS)}")
        try:
            return load_local_data(f"{basename}_{season_code}.parquet")
        except FileNotFoundError:
            pass
    # Try full-year file first, then legacy
    for suffix in ("_2023", ""):
        try:
            return load_local_data(f"{basename}{suffix}.parquet")
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No local data found for '{basename}' (tried _2023 and legacy).")


def _as_timestamp(value):
    """Parse a date/datetime-like value to tz-naive Timestamp, or None."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def slice_time_index(obj, start_date=None, end_date=None, date_range=None, time_col=None, fill_value=None):
    """
    Restrict a Series/DataFrame to a requested time window.

    Preference order:
      1. ``date_range`` — if ``fill_value`` is set, reindex onto that exact index;
         otherwise label-slice to ``date_range[0]``…``date_range[-1]``.
      2. ``start_date`` / ``end_date`` — inclusive label slice on a DatetimeIndex,
         or filter a ``time_col`` for long-format tables (e.g. weather_nuts3).

    Use ``fill_value=0.0`` for generation series; leave it ``None`` for weather.
    Returns a copy; does not mutate the input.
    """
    if obj is None:
        return obj

    if time_col is not None:
        df = obj.copy()
        df[time_col] = pd.to_datetime(df[time_col]).dt.tz_localize(None)
        if date_range is not None:
            start = pd.Timestamp(date_range[0])
            end = pd.Timestamp(date_range[-1])
            return df[(df[time_col] >= start) & (df[time_col] <= end)].copy()
        start = _as_timestamp(start_date)
        end = _as_timestamp(end_date)
        if start is not None:
            df = df[df[time_col] >= start]
        if end is not None:
            df = df[df[time_col] <= end]
        return df.copy()

    out = obj.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    if date_range is not None:
        if fill_value is not None:
            return out.reindex(date_range, fill_value=fill_value)
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[-1])
        return out.loc[start:end].copy()

    start = _as_timestamp(start_date)
    end = _as_timestamp(end_date)
    if start is not None and end is not None:
        return out.loc[start:end].copy()
    if start is not None:
        return out.loc[start:].copy()
    if end is not None:
        return out.loc[:end].copy()
    return out


def load_plz_nuts(engine=None):
    """
    Loads NUTS3 mappings for postcodes from public.plz table (or local file).
    """
    try:
        return load_local_data("plz_nuts.parquet")
    except FileNotFoundError:
        pass

    if engine is None:
        raise FileNotFoundError(
            "plz_nuts.parquet not found and no DB engine available "
            "(OFFLINE_MODE or resolve_engine returned None)."
        )

    with engine.connect() as conn:
        df = pd.read_sql_query("SELECT code, nuts1, nuts2, nuts3, latitude, longitude FROM plz", conn, index_col="code")
    df.index = df.index.astype(str).str.zfill(5)
    return df

def map_row_to_tso(row):
    """
    Maps a row with 'Bundesland' and postcode to its corresponding TSO control zone.
    """
    state = row.get('Bundesland')
    plz = str(row.get('plzCode', row.get('Postleitzahl', ''))).zfill(5)
    if state == 'Baden-Württemberg':
        return 'DE_TRANSNET'
    elif state in ['Berlin', 'Brandenburg', 'Hamburg', 'Mecklenburg-Vorpommern', 'Sachsen', 'Sachsen-Anhalt', 'Thüringen']:
        return 'DE_50HZ'
    elif state in ['Bayern', 'Bremen', 'Niedersachsen', 'Schleswig-Holstein']:
        return 'DE_TENNET'
    elif state in ['Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland']:
        return 'DE_AMPRION'
    elif state == 'Hessen':
        if plz.startswith(('34', '35', '36')):
            return 'DE_TENNET'
        else:
            return 'DE_AMPRION'
    return 'UNKNOWN'

def classify_wind_turbines(wind_df):
    """
    Classifies wind turbines based on Specific Power (Option 2 implementation).
    """
    df = wind_df.copy()
    rotor_radius = df['diameter'] / 2.0
    rotor_area = np.pi * (rotor_radius ** 2)
    df['sp'] = df['maxPower'] / rotor_area
    df['sp'] = df['sp'].replace([np.inf, -np.inf], np.nan).fillna(0.35)
    
    conditions = [
        (df['sp'] <= 0.32),
        (df['sp'] > 0.32) & (df['sp'] <= 0.38),
        (df['sp'] > 0.38) & (df['sp'] <= 0.44),
        (df['sp'] > 0.44)
    ]
    choices = ['class_low', 'class_med_low', 'class_med', 'class_high']
    df['class'] = np.select(conditions, choices, default='class_med')
    return df


# Fixed hubs used by the legacy four SP-class Windpowerlib stack.
SP_CLASS_HUB_M = {
    "class_low": 120.0,
    "class_med_low": 105.0,
    "class_med": 100.0,
    "class_high": 80.0,
}
SP_CLASS_TURBINE_TYPE = {
    "class_low": "V112/3000",
    "class_med_low": "V90/2000",
    "class_med": "E-82/2300",
    "class_high": "E-70/2000",
}


def _wpl_catalogue_table():
    """Parse windpowerlib oedb turbine types into diameter / rated-power rows."""
    from windpowerlib import get_turbine_types
    import re

    rows = []
    for tt in get_turbine_types(print_out=False)["turbine_type"].astype(str):
        if "/" not in tt:
            continue
        left, right = tt.rsplit("/", 1)
        try:
            p_kw = int(right)
        except ValueError:
            continue
        digits = re.findall(r"\d{2,3}", left)
        diam = int(digits[-1]) if digits else None
        rows.append({"turbine_type": tt, "diam_m": diam, "p_kw": p_kw})
    return pd.DataFrame(rows)


def map_mastr_to_wpl_turbine(
    wind_df,
    *,
    max_d_diam=3.0,
    max_d_p=150.0,
    relax_d_diam=5.0,
    relax_d_p=300.0,
    fallback_sp_class=True,
):
    """
    Map each MaStR unit to a windpowerlib catalogue turbine.

    Primary key: nearest (rotor diameter, rated power) in the oedb catalogue.
    Optional fallback: legacy specific-power class representative.
    Adds columns: wpl_type, wpl_match_score, wpl_match_source, hub_m_used.
    """
    df = wind_df.copy()
    if "class" not in df.columns:
        df = classify_wind_turbines(df)

    cat = _wpl_catalogue_table()
    n = len(df)
    types = np.array([None] * n, dtype=object)
    scores = np.full(n, np.nan)
    sources = np.array(["unmapped"] * n, dtype=object)

    diam = pd.to_numeric(df["diameter"], errors="coerce").to_numpy()
    p_kw = pd.to_numeric(df["maxPower"], errors="coerce").to_numpy()
    cat_d = cat["diam_m"].to_numpy(dtype=float)
    cat_p = cat["p_kw"].to_numpy(dtype=float)
    cat_t = cat["turbine_type"].to_numpy()

    for i in range(n):
        if not (np.isfinite(diam[i]) and np.isfinite(p_kw[i])):
            continue
        d = np.abs(cat_d - diam[i])
        p = np.abs(cat_p - p_kw[i])
        mask = (d <= max_d_diam) & (p <= max_d_p)
        if not mask.any():
            mask = (d <= relax_d_diam) & (p <= relax_d_p)
        if not mask.any():
            continue
        score = d[mask] + p[mask] / 1000.0
        j = int(np.argmin(score))
        idx = np.flatnonzero(mask)[j]
        types[i] = cat_t[idx]
        scores[i] = float(score[j])
        sources[i] = "diam_power"

    if fallback_sp_class:
        need = types == None  # noqa: E711
        cls = df["class"].to_numpy()
        for i in np.flatnonzero(need):
            types[i] = SP_CLASS_TURBINE_TYPE.get(cls[i], "E-82/2300")
            sources[i] = "sp_class_fallback"

    df["wpl_type"] = types
    df["wpl_match_score"] = scores
    df["wpl_match_source"] = sources

    if "hub_m" in df.columns:
        hub = pd.to_numeric(df["hub_m"], errors="coerce")
    else:
        hub = pd.Series(np.nan, index=df.index)
    fallback_hub = df["class"].map(SP_CLASS_HUB_M)
    df["hub_m_used"] = hub.fillna(fallback_hub)
    return df

def parse_solar_orientation(solar_df):
    """
    Maps MaStR solar orientation codes to azimuth degrees and tilt degrees.
    """
    df = solar_df.copy()
    df['azimuth'] = df['azimuthCode'].map(mastr_solar_azimuth).fillna(180).astype(int)
    df['tilt'] = df['tiltCode'].map(mastr_solar_tilt).fillna(30).astype(int)
    return df

def query_mastr_wind(engine=None, start_date='2023-01-01', end_date='2023-01-31'):
    """
    Queries wind turbines from the MaStR database (or local file) with postcode coordinate fallback.

    Prefer an enriched parquet (hub_m / manufacturer / type_name). If the cache is the
    older 6-column extract, fall through to the DB when an engine is available.
    """
    try:
        cached = load_local_data("mastr_wind.parquet")
        if "hub_m" in cached.columns or engine is None or offline_mode():
            return cached
    except FileNotFoundError:
        cached = None

    if engine is None:
        if cached is not None:
            return cached
        raise FileNotFoundError("mastr_wind.parquet missing and no DB engine (offline/cache-only).")

    query = f"""
        SELECT
            COALESCE(w."Laengengrad", p.longitude) as lon,
            COALESCE(w."Breitengrad", p.latitude) as lat,
            w."Bruttoleistung" as "maxPower",
            w."Bundesland",
            w."Postleitzahl" as "plzCode",
            w."Rotordurchmesser" as "diameter",
            w."Nabenhoehe" as "hub_m",
            w."Hersteller" as "manufacturer",
            w."Typenbezeichnung" as "type_name",
            w."Inbetriebnahmedatum" as "commission"
        FROM mastr.wind_extended w
        LEFT JOIN public.plz p
            ON w."Postleitzahl" = LPAD(p.code::text, 5, '0')
        WHERE w."EinheitBetriebsstatus" = 'In Betrieb'
          AND w."WindAnLandOderAufSee" = 'Windkraft an Land'
          AND w."Inbetriebnahmedatum" <= '{end_date}'
          AND (w."DatumEndgueltigeStilllegung" IS NULL OR w."DatumEndgueltigeStilllegung" > '{start_date}')
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    df = df.dropna(subset=['lon', 'lat'])
    return df

def query_mastr_solar(engine=None, start_date='2023-01-01', end_date='2023-01-31'):
    """
    Queries solar installations from the MaStR database (or local file) with postcode coordinate fallback.
    """
    try:
        return load_local_data("mastr_solar.parquet")
    except FileNotFoundError:
        pass

    if engine is None:
        raise FileNotFoundError("mastr_solar.parquet missing and no DB engine (offline/cache-only).")

    query = f"""
        SELECT 
            COALESCE(s."Laengengrad", p.longitude) as lon,
            COALESCE(s."Breitengrad", p.latitude) as lat,
            s."Bruttoleistung" as "maxPower",
            s."Bundesland",
            s."Postleitzahl" as "plzCode",
            COALESCE(s."Hauptausrichtung", 'Süd') as "azimuthCode",
            COALESCE(s."HauptausrichtungNeigungswinkel", '21 - 40 Grad') as "tiltCode"
        FROM "mastr-2025".solar_extended s
        LEFT JOIN public.plz p 
            ON s."Postleitzahl" = LPAD(p.code::text, 5, '0')
        WHERE s."EinheitBetriebsstatus" = 'In Betrieb'
          AND s."Inbetriebnahmedatum" <= '{end_date}'
          AND (s."DatumEndgueltigeStilllegung" IS NULL OR s."DatumEndgueltigeStilllegung" > '{start_date}')
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    df = df.dropna(subset=['lon', 'lat'])
    return df

def query_entsoe_generation(engine=None, start_date=None, end_date=None, country_zones=None, date_range=None, season_code=None):
    """
    Fetches ENTSO-E generation data (or local file) for specified country zones.

    Always returns frames aligned to ``date_range`` when provided, otherwise sliced
    to ``start_date``/``end_date``. Local parquet caches may contain the full year;
    this function is responsible for restricting them to the analysis window.
    """
    if country_zones is None:
        country_zones = ['DE_50HZ', 'DE_AMPRION', 'DE_TENNET', 'DE_TRANSNET']

    try:
        entsoe_wind = load_local_data_season("entsoe_generation_wind", season_code)
        entsoe_solar = load_local_data_season("entsoe_generation_solar", season_code)
        # Keep only requested zones when present
        wind_cols = [z for z in country_zones if z in entsoe_wind.columns]
        solar_cols = [z for z in country_zones if z in entsoe_solar.columns]
        entsoe_wind = entsoe_wind[wind_cols]
        entsoe_solar = entsoe_solar[solar_cols]
        entsoe_wind = slice_time_index(
            entsoe_wind, start_date, end_date, date_range, fill_value=0.0
        )
        entsoe_solar = slice_time_index(
            entsoe_solar, start_date, end_date, date_range, fill_value=0.0
        )
        return entsoe_wind, entsoe_solar
    except FileNotFoundError:
        pass

    if engine is None:
        raise FileNotFoundError("ENTSO-E parquet cache missing and no DB engine (offline/cache-only).")

    if date_range is None:
        if start_date is None or end_date is None:
            raise ValueError("query_entsoe_generation requires date_range or start_date/end_date")
        date_range = pd.date_range(start_date, end_date, freq='h')

    entsoe_wind = pd.DataFrame(index=date_range)
    entsoe_solar = pd.DataFrame(index=date_range)

    for zone in country_zones:
        query = f"""
            SELECT "index" as time, solar, wind_onshore
            FROM entsoe.query_generation
            WHERE index BETWEEN '{start_date}' AND '{end_date}'
              AND country = '{zone}'
            ORDER BY time ASC
        """
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, index_col='time')

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df_hourly = df.groupby(df.index).first().resample('h').mean()

        entsoe_wind[zone] = df_hourly['wind_onshore'].reindex(date_range, fill_value=0.0)
        entsoe_solar[zone] = df_hourly['solar'].reindex(date_range, fill_value=0.0)

    return entsoe_wind, entsoe_solar

def query_entsoe_installed_capacity(engine=None, date_str='2022-12-31 23:00:00', zones=None):
    """
    Queries ENTSO-E installed generation capacity (or local file) for specified zones.
    """
    try:
        return load_local_data("entsoe_capacity.parquet")
    except FileNotFoundError:
        pass
        
    if zones is None:
        zones = ['DE_50HZ', 'DE_AMPRION', 'DE_TENNET', 'DE_TRANSNET']
    zones_str = "', '".join(zones)
    query = f"""
        SELECT 
            country, 
            MAX(solar) as solar_cap_mw, 
            MAX(wind_onshore) as wind_cap_mw
        FROM entsoe.query_installed_generation_capacity
        WHERE index = '{date_str}'
          AND country IN ('{zones_str}')
        GROUP BY country
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, index_col='country')
    return df

def query_ecmwf_weather_nuts3(engine=None, start_date=None, end_date=None, nuts_prefix='DE', season_code=None, date_range=None):
    """
    Queries county-level (NUTS3) ECMWF weather data (or local file).

    Local caches are filtered to ``date_range`` or ``start_date``/``end_date`` and
    to ``nuts_prefix`` (e.g. 'DE' → nuts_id LIKE 'DE___').
    """
    try:
        df = load_local_data_season("weather_ecmwf_nuts3", season_code)
        if "nuts_id" in df.columns and nuts_prefix:
            df = df[df["nuts_id"].astype(str).str.startswith(nuts_prefix)].copy()
        return slice_time_index(df, start_date, end_date, date_range, time_col="time")
    except FileNotFoundError:
        pass

    if engine is None:
        raise FileNotFoundError("weather_ecmwf_nuts3 parquet missing and no DB engine (offline/cache-only).")

    query = f"""
        SELECT time, nuts_id,
               avg(temp_air) as temp_air,
               avg(ghi) as ghi,
               avg(wind_speed) as wind_speed
        FROM ecmwf.ecmwf_eu
        WHERE time BETWEEN '{start_date}' AND '{end_date}'
          AND nuts_id LIKE '{nuts_prefix}___'
        GROUP BY time, nuts_id
        ORDER BY time ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    return df

def query_ecmwf_weather(engine=None, start_date=None, end_date=None, lat=None, lon=None, nuts_id=None, date_range=None):
    """
    Queries ECMWF weather data (or local file) for either a specific lat/lon or nuts_id.
    Local caches are sliced to the requested time window.
    """
    local_df = None
    if nuts_id == 'UKF2' or (lat is not None and lon is not None and abs(lat - 52.4) < 0.1 and abs(lon - -0.88) < 0.1):
        try:
            local_df = load_local_data("weather_kelmarsh.parquet")
        except FileNotFoundError:
            pass
    elif lat is not None and lon is not None:
        if abs(lat - 50.9089) < 0.1 and abs(lon - 6.4135) < 0.1:
            try:
                local_df = load_local_data("weather_juelich.parquet")
            except FileNotFoundError:
                pass
        # Campus Jülich DB grid point used by validate_juelich_solar
        elif abs(lat - 50.845454545454544) < 0.05 and abs(lon - 6.454545454545454) < 0.05:
            try:
                local_df = load_local_data("weather_juelich.parquet")
            except FileNotFoundError:
                pass

    if local_df is not None:
        if not isinstance(local_df.index, pd.DatetimeIndex):
            if "time" in local_df.columns:
                local_df = local_df.set_index("time")
            local_df.index = pd.to_datetime(local_df.index)
        return slice_time_index(local_df, start_date, end_date, date_range)

    if nuts_id is not None:
        query = f"""
            SELECT time, temp_air, wind_speed, ghi 
            FROM ecmwf.ecmwf_eu 
            WHERE nuts_id = '{nuts_id}'
              AND time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY time ASC
        """
    elif lat is not None and lon is not None:
        query = f"""
            SELECT time, temp_air, wind_speed, ghi 
            FROM ecmwf.ecmwf_eu 
            WHERE latitude = {lat} AND longitude = {lon}
              AND time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY time ASC
        """
    else:
        raise ValueError("Must specify either nuts_id or both lat and lon")

    if engine is None:
        raise FileNotFoundError("Point weather parquet missing and no DB engine (offline/cache-only).")

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, index_col='time')
    return slice_time_index(df, start_date, end_date, date_range)

def query_juelich_actual(engine=None, start_date='2023-01-01 00:00:00', end_date='2023-12-31 23:59:59', date_range=None):
    """
    Queries Jülich actual PV generation data (or local file) from Timescale DB.
    """
    try:
        df = load_local_data("juelich_actuals.parquet")
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df = df.set_index("time")
            df.index = pd.to_datetime(df.index)
        return slice_time_index(df, start_date, end_date, date_range)
    except FileNotFoundError:
        pass

    if engine is None:
        raise FileNotFoundError("juelich_actuals.parquet missing and no DB engine (offline/cache-only).")

    query = f"""
        SELECT time_bucket('3600s', datetime AT TIME ZONE 'Europe/Berlin') AT TIME ZONE 'UTC' AS "time", AVG(value) AS "generation"
        FROM eview.eview
        WHERE plant = 'FP-JUEL'
          AND datetime BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 1
        ORDER BY 1
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, index_col='time')
    return slice_time_index(df, start_date, end_date, date_range)

def process_kelmarsh_scada(zip_path, output_csv_path, turbines, date_range):
    """
    Processes Kelmarsh raw SCADA files from zip and aggregates hourly average power.
    """
    hourly_df = pd.DataFrame(index=date_range)
    
    with zipfile.ZipFile(zip_path) as z:
        for t_name, csv_filename in turbines.items():
            hourly_sums = {}
            hourly_counts = {}
            with z.open(csv_filename) as f:
                # Need to decode the file stream line by line
                csv_reader = csv.reader((line.decode('utf-8') for line in f))
                for row in csv_reader:
                    if not row or row[0].startswith('#'):
                        continue
                    try:
                        timestamp_str = row[0]
                        power_kw_str = row[62]
                        if power_kw_str and power_kw_str != 'NaN':
                            power_kw = float(power_kw_str)
                            hour_str = timestamp_str[:13] + ":00:00"
                            hourly_sums[hour_str] = hourly_sums.get(hour_str, 0.0) + power_kw
                            hourly_counts[hour_str] = hourly_counts.get(hour_str, 0) + 1
                    except Exception:
                        continue
            hourly_avg = {}
            for hr, val_sum in hourly_sums.items():
                count = hourly_counts[hr]
                hourly_avg[hr] = val_sum / count if count > 0 else 0.0
            t_series = pd.Series(hourly_avg)
            t_series.index = pd.to_datetime(t_series.index)
            t_series = t_series.reindex(date_range, fill_value=0.0)
            hourly_df[f"{t_name}_actual"] = t_series
            
    hourly_df.to_csv(output_csv_path)
    return hourly_df

def calculate_metrics(sim, act):
    """
    Calculates MAE, RMSE, Pearson Correlation, Sums, and Ratio (Sim/Act * 100).

    Series are inner-aligned on their indexes before *all* metrics (including sums),
    so a January simulation cannot be accidentally ratioed against a full-year actual.
    """
    sim = pd.Series(sim).copy()
    act = pd.Series(act).copy()
    if not isinstance(sim.index, pd.DatetimeIndex):
        try:
            sim.index = pd.to_datetime(sim.index)
        except (TypeError, ValueError):
            pass
    if not isinstance(act.index, pd.DatetimeIndex):
        try:
            act.index = pd.to_datetime(act.index)
        except (TypeError, ValueError):
            pass

    sim, act = sim.align(act, join="inner")
    mask = sim.notna() & act.notna()
    sim = sim[mask]
    act = act[mask]

    if len(sim) == 0:
        return {
            'corr': float("nan"),
            'mae': float("nan"),
            'rmse': float("nan"),
            'sim_sum': 0.0,
            'act_sum': 0.0,
            'ratio': 0.0,
            'n': 0,
        }

    corr = float(sim.corr(act)) if len(sim) > 1 else float("nan")
    mae = float((sim - act).abs().mean())
    rmse = float(np.sqrt(((sim - act) ** 2).mean()))
    sim_sum = float(sim.sum())
    act_sum = float(act.sum())
    ratio = (sim_sum / act_sum * 100) if act_sum > 0 else 0.0
    return {
        'corr': corr,
        'mae': mae,
        'rmse': rmse,
        'sim_sum': sim_sum,
        'act_sum': act_sum,
        'ratio': ratio,
        'n': int(len(sim)),
    }

def plot_duration_curves(df_dict, title, ylabel, save_path, colors=None):
    """
    Plots the generation duration curves for compared datasets.
    """
    plt.figure(figsize=(10, 6))
    for idx, (label, series) in enumerate(df_dict.items()):
        sorted_vals = series.sort_values(ascending=False).values
        x_percent = np.linspace(0, 100, len(sorted_vals))
        color = colors[idx] if colors and idx < len(colors) else None
        plt.plot(x_percent, sorted_vals, label=label, linewidth=2, color=color)
    plt.title(title)
    plt.xlabel("Percentage of Time (%)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_scatter_comparison(actual, predicted_dict, title, ylabel, save_path, colors=None):
    """
    Plots the scatter comparison between actual and simulated outputs.
    """
    plt.figure(figsize=(8, 8))
    for idx, (label, pred) in enumerate(predicted_dict.items()):
        color = colors[idx] if colors and idx < len(colors) else None
        plt.scatter(actual, pred, alpha=0.15, s=10, label=label, color=color)
    max_val = max(actual.max(), max(p.max() for p in predicted_dict.values()))
    plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=2, label='Perfect Match (y = x)')
    plt.title(title)
    plt.xlabel("Actual Measured Power (kW)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_timeseries_comparison(df, columns_dict, title, ylabel, save_path, sample_range=None, figsize=(12, 6), colors=None, linestyles=None):
    """
    Plots comparison timeseries for a slice of dates.
    """
    plt.figure(figsize=figsize)
    plot_df = df.loc[sample_range] if sample_range else df
    for idx, (col, label) in enumerate(columns_dict.items()):
        color = colors[idx] if colors and idx < len(colors) else None
        linestyle = linestyles[idx] if linestyles and idx < len(linestyles) else '-'
        plt.plot(plot_df[col], label=label, alpha=0.8, color=color, linestyle=linestyle)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tso_timeseries_comparison(actual_df, sim_dfs_dict, zones, title, ylabel, save_path, figsize=(15, 12), colors_dict=None, linestyles_dict=None):
    """
    Plots multi-subplot timeseries comparisons for TSO zones.
    """
    fig, axes = plt.subplots(len(zones), 1, figsize=figsize, sharex=True)
    for idx, zone in enumerate(zones):
        ax = axes[idx]
        ax.plot(actual_df[zone], label='ENTSO-E Actual', color='black', linewidth=1.5)
        for label, sim_df in sim_dfs_dict.items():
            color = colors_dict.get(label) if colors_dict else None
            linestyle = linestyles_dict.get(label) if linestyles_dict else '--'
            ax.plot(sim_df[zone], label=label, linestyle=linestyle, alpha=0.8, color=color)
        ax.set_title(f"{title} - {zone} Control Area")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
