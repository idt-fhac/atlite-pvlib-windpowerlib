from pathlib import Path
"""
test_aggregation_baseline.py
============================
Compares high-granularity OEDS wind simulation against single-location baselines
that scale one turbine profile to national capacity.

Reference high-granularity metrics are loaded from:
  - annual_seasonal_national.csv  (default z0=0.20)
  - wind_z0_sensitivity.csv       (TSO-tuned z0 → national aggregate estimate)

Locations tested for the single-turbine baseline:
  1. Geographic Center (Niederdorla / DEG0H)
  2. Hilly Inland (Feldberg / DE132)
  3. Coastal High-Wind (Nordfriesland / DEF07)
"""

import sys
import logging
import numpy as np
import pandas as pd
from windpowerlib import ModelChain, WindTurbine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    get_db_engine,
    load_plz_nuts,
    map_row_to_tso,
    query_mastr_wind,
    query_entsoe_generation,
    query_ecmwf_weather_nuts3,
    calculate_metrics,
    ensure_results_dir,
    result_path,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("aggregation")

ZONES = ['DE_50HZ', 'DE_AMPRION', 'DE_TENNET', 'DE_TRANSNET']
TURBINE_MODEL = WindTurbine(hub_height=100.0, turbine_type="E-82/2300")

# Optimal z0 per TSO from sensitivity_roughness (≈90% yield target)
# Discrete sweep optima nearest ~90% yield ratio (see wind_z0_sensitivity.csv).
TUNED_Z0 = {
    'DE_TENNET': 0.15,
    'DE_50HZ': 0.40,
    'DE_AMPRION': 0.50,
    'DE_TRANSNET': 0.80,
}


def _load_reference_metrics():
    """Load published high-granularity metrics from prior study outputs."""
    nat_path = result_path("annual_seasonal_national.csv")
    z0_path = result_path("wind_z0_sensitivity.csv")

    default = {"ratio": float("nan"), "corr": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    tuned = dict(default)

    if nat_path.exists():
        nat = pd.read_csv(nat_path)
        fy = nat[nat["Season"].astype(str).str.contains("Full Year", case=False)].iloc[0]
        default = {
            "ratio": float(fy["Wind_Ratio_%"]),
            "corr": float(fy["Wind_Corr"]),
            "mae": float(fy["Wind_MAE_MW"]),
            "rmse": float(fy["Wind_RMSE_MW"]),
        }

    # Prefer a true national recompute if available
    tuned_path = result_path("tso_roughness_national_comparison.csv")
    if tuned_path.exists():
        tr = pd.read_csv(tuned_path)
        row = tr[tr["Configuration"].astype(str).str.contains("TSO-specific", case=False)]
        if len(row) == 1:
            r = row.iloc[0]
            tuned = {
                "ratio": float(r["Yield_Ratio_%"]),
                "corr": float(r["Correlation"]),
                "mae": float(r["MAE_MW"]),
                "rmse": float(r["RMSE_MW"]),
            }
    elif z0_path.exists():
        z0 = pd.read_csv(z0_path)
        # Yield-weighted ratio/corr only (zone MAE/RMSE ≠ national MAE/RMSE)
        rows = []
        for tso, z in TUNED_Z0.items():
            sub = z0[(z0["TSO"] == tso) & (np.isclose(z0["z0"], z))]
            if len(sub) == 1:
                rows.append(sub.iloc[0])
        if rows:
            ref = pd.DataFrame(rows)
            w = ref["Actual_GWh"].astype(float)
            tuned = {
                "ratio": float((ref["Ratio_%"] * w).sum() / w.sum()),
                "corr": float((ref["Correlation"] * w).sum() / w.sum()),
                "mae": float("nan"),
                "rmse": float("nan"),
            }

    return default, tuned


def main():
    ensure_results_dir()
    date_range = pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='h')
    engine = get_db_engine('timescale')

    logger.info("Loading wind registry capacity and ENTSO-E actuals...")
    wind_df = query_mastr_wind(engine, '2023-01-01', '2023-12-31')
    plz_nuts = load_plz_nuts(engine)

    wind_df['nuts3'] = wind_df['plzCode'].map(plz_nuts['nuts3'])
    wind_df = wind_df.dropna(subset=['nuts3'])
    wind_df['tso'] = wind_df.apply(map_row_to_tso, axis=1)
    wind_df = wind_df[wind_df['tso'] != 'UNKNOWN']

    total_capacity_mw = wind_df['maxPower'].sum() / 1e3
    logger.info(f"Total active wind capacity in Germany: {total_capacity_mw:.2f} MW")

    entsoe_wind_annual, _ = query_entsoe_generation(
        engine, '2023-01-01', '2023-12-31', ZONES, date_range
    )
    assert len(entsoe_wind_annual) == len(date_range)
    act_w_nat = entsoe_wind_annual.sum(axis=1)

    logger.info("Loading weather data for the baseline simulations...")
    weather_raw = query_ecmwf_weather_nuts3(
        engine, '2023-01-01', '2023-12-31', nuts_prefix='DE', date_range=date_range
    )
    weather_dict = {nuts: grp.set_index('time') for nuts, grp in weather_raw.groupby('nuts_id')}

    baselines = {
        "Geographic Center (DEG0H)": "DEG0H",
        "Hilly Inland (DE132)": "DE132",
        "Coastal High-Wind (DEF07)": "DEF07",
    }
    results = {}

    for label, nuts in baselines.items():
        logger.info(f"Simulating baseline: {label}...")
        weather = weather_dict[nuts].reindex(date_range, method='nearest')
        ww = pd.DataFrame(
            np.asarray([
                0.2 * np.ones(len(date_range)),
                weather["temp_air"].values,
                weather["wind_speed"].values,
            ]).T,
            index=date_range,
            columns=[["roughness_length", "temperature", "wind_speed"], [0, 2, 10]],
        )
        mc = ModelChain(TURBINE_MODEL).run_model(ww)
        sim_mw = (mc.power_output / TURBINE_MODEL.nominal_power) * total_capacity_mw
        results[label] = calculate_metrics(sim_mw, act_w_nat)

    default_ref, tuned_ref = _load_reference_metrics()

    rows = [
        {
            "Configuration": "OEDS high-granularity (default z0=0.20, from annual_seasonal_national.csv)",
            "Yield_Ratio_%": default_ref["ratio"],
            "Correlation": default_ref["corr"],
            "MAE_MW": default_ref["mae"],
            "RMSE_MW": default_ref["rmse"],
        },
        {
            "Configuration": "OEDS high-granularity (TSO-tuned z0, from wind_z0_sensitivity.csv)",
            "Yield_Ratio_%": tuned_ref["ratio"],
            "Correlation": tuned_ref["corr"],
            "MAE_MW": tuned_ref["mae"],
            "RMSE_MW": tuned_ref["rmse"],
        },
    ]
    for label, m in results.items():
        rows.append({
            "Configuration": f"Single-location baseline: {label}",
            "Yield_Ratio_%": m["ratio"],
            "Correlation": m["corr"],
            "MAE_MW": m["mae"],
            "RMSE_MW": m["rmse"],
        })

    out = pd.DataFrame(rows)
    out_path = result_path("aggregation_baseline_comparison.csv")
    out.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("COMPARISON: HIGH-GRANULARITY MODEL VS. SINGLE-LOCATION BASELINES (FULL YEAR 2023)")
    print("=" * 80)
    print(out.to_string(index=False))
    print("=" * 80)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
