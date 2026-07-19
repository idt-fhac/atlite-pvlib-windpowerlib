from pathlib import Path
import os
import sys
import logging
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    get_db_engine,
    query_mastr_wind,
    query_mastr_solar,
    load_plz_nuts,
    query_entsoe_generation,
    query_entsoe_installed_capacity,
    query_ecmwf_weather_nuts3,
    query_ecmwf_weather,
    query_juelich_actual,
    DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("extract_data")

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


def main():
    from utils import offline_mode
    if offline_mode():
        raise RuntimeError(
            "extract_research_data.py requires database access. "
            "Unset OFFLINE_MODE (or omit --offline) to refresh parquet caches."
        )

    os.environ["BYPASS_LOCAL_DATA"] = "True"
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing all local data to: {data_dir}")

    # Establish DB engines
    oeds_engine = get_db_engine('timescale')
    timescale_engine = get_db_engine('timescale')

    # -----------------------------------------------------------------------
    # Date ranges
    # -----------------------------------------------------------------------
    year_start = '2023-01-01'
    year_end = '2023-12-31'
    year_date_range = pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='h')

    # Legacy January-only range
    jan_start = '2023-01-01'
    jan_end = '2023-01-31'

    zones = ['DE_50HZ', 'DE_AMPRION', 'DE_TENNET', 'DE_TRANSNET']

    # -----------------------------------------------------------------------
    # 1. Postcode NUTS3 mappings (static)
    # -----------------------------------------------------------------------
    logger.info("Extracting plz_nuts...")
    plz_nuts = load_plz_nuts(oeds_engine)
    plz_nuts.to_parquet((data_dir / "plz_nuts.parquet"))

    # -----------------------------------------------------------------------
    # 2. MaStR Wind Onshore (snapshot: active during 2023)
    # -----------------------------------------------------------------------
    logger.info("Extracting mastr_wind...")
    wind = query_mastr_wind(oeds_engine, year_start, year_end)
    wind.to_parquet((data_dir / "mastr_wind.parquet"))

    # -----------------------------------------------------------------------
    # 3. MaStR Solar PV (snapshot: active during 2023)
    # -----------------------------------------------------------------------
    logger.info("Extracting mastr_solar...")
    solar = query_mastr_solar(oeds_engine, year_start, year_end)
    solar.to_parquet((data_dir / "mastr_solar.parquet"))

    # -----------------------------------------------------------------------
    # 4. ENTSO-E installed capacity (as of end of 2022)
    # -----------------------------------------------------------------------
    logger.info("Extracting entsoe_capacity...")
    capacity = query_entsoe_installed_capacity(oeds_engine, '2022-12-31 23:00:00')
    capacity.to_parquet((data_dir / "entsoe_capacity.parquet"))

    # -----------------------------------------------------------------------
    # 5. ENTSO-E generation — full year 2023 + per-season + legacy Jan
    # -----------------------------------------------------------------------
    logger.info("Extracting entsoe_generation (full year 2023)...")
    entsoe_wind_annual, entsoe_solar_annual = query_entsoe_generation(
        oeds_engine, year_start, year_end, zones, year_date_range
    )
    entsoe_wind_annual.to_parquet((data_dir / "entsoe_generation_wind_2023.parquet"))
    entsoe_solar_annual.to_parquet((data_dir / "entsoe_generation_solar_2023.parquet"))

    # Legacy January-only files (backward compatibility)
    mask_jan = year_date_range.month == 1
    entsoe_wind_annual[mask_jan].to_parquet((data_dir / "entsoe_generation_wind.parquet"))
    entsoe_solar_annual[mask_jan].to_parquet((data_dir / "entsoe_generation_solar.parquet"))

    # Per-season slices
    for season_code, months in SEASONS.items():
        mask = year_date_range.month.isin(months)
        entsoe_wind_annual[mask].to_parquet(
            (data_dir / f"entsoe_generation_wind_{season_code}.parquet")
        )
        entsoe_solar_annual[mask].to_parquet(
            (data_dir / f"entsoe_generation_solar_{season_code}.parquet")
        )
        logger.info(f"  Saved ENTSO-E {SEASON_LABELS[season_code]} ({season_code}) slices.")

    # -----------------------------------------------------------------------
    # 6. Germany NUTS3 ECMWF weather — full year + per-season + legacy Jan
    # -----------------------------------------------------------------------
    logger.info("Extracting weather_ecmwf_nuts3 (full year 2023) — this may take a few minutes...")
    weather_nuts3 = query_ecmwf_weather_nuts3(oeds_engine, year_start, year_end)
    weather_nuts3.to_parquet((data_dir / "weather_ecmwf_nuts3_2023.parquet"))
    logger.info(f"  Saved {len(weather_nuts3):,} NUTS3 weather records.")

    # Determine whether 'time' is a column or index for slicing
    if 'time' in weather_nuts3.columns:
        time_series = pd.to_datetime(weather_nuts3['time'])
    else:
        time_series = pd.to_datetime(weather_nuts3.index)

    # Legacy January-only file
    weather_nuts3[time_series.dt.month == 1].to_parquet(
        (data_dir / "weather_ecmwf_nuts3.parquet")
    )

    # Per-season slices
    for season_code, months in SEASONS.items():
        mask = time_series.dt.month.isin(months)
        weather_nuts3[mask].to_parquet(
            (data_dir / f"weather_ecmwf_nuts3_{season_code}.parquet")
        )
        logger.info(f"  Saved ECMWF NUTS3 {SEASON_LABELS[season_code]} ({season_code}) slice.")

    # -----------------------------------------------------------------------
    # 7. Localized weather for Jülich — full 2023
    # -----------------------------------------------------------------------
    logger.info("Extracting weather_juelich (full year 2023)...")
    db_lat, db_lon = 50.845454545454544, 6.454545454545454
    weather_juelich = query_ecmwf_weather(
        oeds_engine, '2023-01-01 00:00:00', '2023-12-31 23:00:00', lat=db_lat, lon=db_lon
    )
    weather_juelich.to_parquet((data_dir / "weather_juelich.parquet"))

    # -----------------------------------------------------------------------
    # 8. Localized weather for Kelmarsh — full 2023
    # -----------------------------------------------------------------------
    logger.info("Extracting weather_kelmarsh (full year 2023)...")
    weather_kelmarsh = query_ecmwf_weather(
        oeds_engine, '2023-01-01 00:00:00', '2023-12-31 23:00:00', nuts_id='UKF2'
    )
    weather_kelmarsh.to_parquet((data_dir / "weather_kelmarsh.parquet"))

    # -----------------------------------------------------------------------
    # 9. Jülich actual solar generation — full 2023
    # -----------------------------------------------------------------------
    logger.info("Extracting juelich_actuals (full year 2023)...")
    juelich_act = query_juelich_actual(timescale_engine, '2023-01-01 00:00:00', '2023-12-31 23:59:59')
    juelich_act.to_parquet((data_dir / "juelich_actuals.parquet"))

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    logger.info("All local research data successfully extracted!")
    logger.info("")
    logger.info("Files written:")
    for f in sorted(os.listdir(data_dir)):
        size_mb = (data_dir / f).stat().st_size / 1e6
        logger.info(f"  {f:60s}  {size_mb:7.2f} MB")


if __name__ == "__main__":
    main()
