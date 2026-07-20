from pathlib import Path
"""
test_tso_roughness.py
=====================
Computes national Germany onshore wind generation for 2023 when using TSO-specific
roughness length parameters vs uniform z0=0.20 m.
Discrete sweep optima nearest ~90% yield (see wind_z0_sensitivity.csv):
  - TenneT: 0.15 m
  - 50Hertz: 0.40 m
  - Amprion: 0.50 m
  - TransnetBW: 0.80 m
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
    classify_wind_turbines,
    query_mastr_wind,
    query_entsoe_generation,
    query_ecmwf_weather_nuts3,
    calculate_metrics,
    ensure_results_dir,
    result_path,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_tso")

ZONES = ['DE_50HZ', 'DE_AMPRION', 'DE_TENNET', 'DE_TRANSNET']
ROUGHNESS_TSO = {
    'DE_TENNET': 0.15,
    'DE_50HZ': 0.40,
    'DE_AMPRION': 0.50,
    'DE_TRANSNET': 0.80,
}

TURBINE_MODELS = {
    'class_low':     WindTurbine(hub_height=120.0, turbine_type="V112/3000"),
    'class_med_low': WindTurbine(hub_height=105.0, turbine_type="V90/2000"),
    'class_med':     WindTurbine(hub_height=100.0, turbine_type="E-82/2300"),
    'class_high':    WindTurbine(hub_height=80.0,  turbine_type="E-70/2000"),
}


def run_national_wind(z0_by_tso, wind_groups, weather_dict, nuts3_to_tso, date_range):
    sim_w_tso = {tso: pd.Series(0.0, index=date_range) for tso in ZONES}
    for nuts in wind_groups.index:
        if nuts not in weather_dict or nuts not in nuts3_to_tso:
            continue
        tso = nuts3_to_tso[nuts]
        z0 = z0_by_tso[tso]
        weather_df = weather_dict[nuts].reindex(date_range, method='nearest')
        row = wind_groups.loc[nuts]
        ww = pd.DataFrame(
            np.asarray([
                z0 * np.ones(len(date_range)),
                weather_df["temp_air"].values,
                weather_df["wind_speed"].values,
            ]).T,
            index=date_range,
            columns=[["roughness_length", "temperature", "wind_speed"], [0, 2, 10]],
        )
        for cls_name, capacity_kw in row.items():
            if capacity_kw > 0:
                wt = TURBINE_MODELS[cls_name]
                mc = ModelChain(wt).run_model(ww)
                cls_power_mw = mc.power_output / wt.nominal_power * (capacity_kw * 1e3) / 1e6
                sim_w_tso[tso] += cls_power_mw
    return sum(sim_w_tso.values())


def main():
    ensure_results_dir()
    date_range = pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='h')
    engine = get_db_engine('timescale')

    wind_df = query_mastr_wind(engine, '2023-01-01', '2023-12-31')
    plz_nuts = load_plz_nuts(engine)

    wind_df['nuts3'] = wind_df['plzCode'].map(plz_nuts['nuts3'])
    wind_df = wind_df.dropna(subset=['nuts3'])
    wind_df['tso'] = wind_df.apply(map_row_to_tso, axis=1)
    wind_df = wind_df[wind_df['tso'] != 'UNKNOWN']
    wind_df = classify_wind_turbines(wind_df)

    wind_groups = wind_df.groupby(['nuts3', 'class'])['maxPower'].sum().unstack(fill_value=0.0)
    nuts3_to_tso = {nuts3: grp['tso'].iloc[0] for nuts3, grp in wind_df.groupby('nuts3')}

    weather_raw = query_ecmwf_weather_nuts3(
        engine, '2023-01-01', '2023-12-31', nuts_prefix='DE', date_range=date_range
    )
    weather_dict = {nuts: grp.set_index('time') for nuts, grp in weather_raw.groupby('nuts_id')}

    entsoe_wind_annual, _ = query_entsoe_generation(
        engine, '2023-01-01', '2023-12-31', ZONES, date_range
    )
    assert len(entsoe_wind_annual) == len(date_range)
    act_w_nat = entsoe_wind_annual.sum(axis=1)

    logger.info("Running uniform z0=0.20 simulation...")
    sim_default = run_national_wind(
        {tso: 0.20 for tso in ZONES},
        wind_groups, weather_dict, nuts3_to_tso, date_range,
    )
    m_default = calculate_metrics(sim_default, act_w_nat)

    logger.info("Running TSO-specific roughness simulation...")
    sim_tuned = run_national_wind(
        ROUGHNESS_TSO, wind_groups, weather_dict, nuts3_to_tso, date_range,
    )
    m_tuned = calculate_metrics(sim_tuned, act_w_nat)

    out = pd.DataFrame([
        {
            "Configuration": "Uniform z0=0.20",
            "Yield_GWh": m_default["sim_sum"] / 1e3,
            "Yield_Ratio_%": m_default["ratio"],
            "Correlation": m_default["corr"],
            "MAE_MW": m_default["mae"],
            "RMSE_MW": m_default["rmse"],
            "n": m_default["n"],
        },
        {
            "Configuration": "TSO-specific z0",
            "Yield_GWh": m_tuned["sim_sum"] / 1e3,
            "Yield_Ratio_%": m_tuned["ratio"],
            "Correlation": m_tuned["corr"],
            "MAE_MW": m_tuned["mae"],
            "RMSE_MW": m_tuned["rmse"],
            "n": m_tuned["n"],
        },
    ])
    out_path = result_path("tso_roughness_national_comparison.csv")
    out.to_csv(out_path, index=False)

    print("\n=== NATIONAL WIND ONSHORE METRICS COMPARISON ===")
    print(out.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
