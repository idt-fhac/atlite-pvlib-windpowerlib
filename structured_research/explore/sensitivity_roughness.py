from pathlib import Path
"""
sensitivity_roughness.py
========================
Sensitivity analysis of wind speed logarithmic extrapolation to roughness length (z0).
Simulates German onshore wind generation for 2023 across all four TSO control zones
varying the roughness length z0 from 0.01 m to 0.8 m.

Saves the results to structured_research/results/wind_z0_sensitivity.csv and generates
a summary plot.
"""

import sys
import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from windpowerlib import ModelChain, WindTurbine
import matplotlib.pyplot as plt

# Imports
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
logger = logging.getLogger("sensitivity")

ZONES = ['DE_50HZ', 'DE_AMPRION', 'DE_TENNET', 'DE_TRANSNET']
TURBINE_MODELS = {
    'class_low':     WindTurbine(hub_height=120.0, turbine_type="V112/3000"),
    'class_med_low': WindTurbine(hub_height=105.0, turbine_type="V90/2000"),
    'class_med':     WindTurbine(hub_height=100.0, turbine_type="E-82/2300"),
    'class_high':    WindTurbine(hub_height=80.0,  turbine_type="E-70/2000"),
}


def run_wind_sim_z0(z0, wind_groups, weather_dict, nuts3_all, nuts3_to_tso_dict, date_range):
    """Run windpowerlib simulation for a specific z0 roughness value."""
    sim_w_tso = {tso: pd.Series(0.0, index=date_range) for tso in ZONES}

    for nuts in nuts3_all:
        if nuts not in weather_dict or nuts not in nuts3_to_tso_dict:
            continue
        if nuts not in wind_groups.index:
            continue

        tso = nuts3_to_tso_dict[nuts]
        weather_df = weather_dict[nuts].reindex(date_range, method='nearest')
        
        row = wind_groups.loc[nuts]
        # logarithmic extrapolation inputs
        ww_data = np.asarray([
            z0 * np.ones(len(date_range)),         # roughness_length [m]
            weather_df["temp_air"].values,         # temperature [K]
            weather_df["wind_speed"].values,       # wind speed at 10 m [m/s]
        ]).T
        ww = pd.DataFrame(
            ww_data,
            index=date_range,
            columns=[["roughness_length", "temperature", "wind_speed"], [0, 2, 10]],
        )
        
        for cls_name, capacity_kw in row.items():
            if capacity_kw > 0:
                wt = TURBINE_MODELS[cls_name]
                mc = ModelChain(wt).run_model(ww)
                cls_power_mw = mc.power_output / wt.nominal_power * (capacity_kw * 1e3) / 1e6
                sim_w_tso[tso] += cls_power_mw

    return sim_w_tso


def main():
    ensure_results_dir()

    date_range = pd.date_range('2023-01-01 00:00:00', '2023-12-31 23:00:00', freq='h')
    engine = get_db_engine('timescale')

    logger.info("Loading wind registry and weather data...")
    wind_df  = query_mastr_wind(engine, '2023-01-01', '2023-12-31')
    plz_nuts = load_plz_nuts(engine)

    wind_df['nuts3']  = wind_df['plzCode'].map(plz_nuts['nuts3'])
    wind_df  = wind_df.dropna(subset=['nuts3'])
    wind_df['tso']  = wind_df.apply(map_row_to_tso, axis=1)
    wind_df  = wind_df[wind_df['tso'] != 'UNKNOWN']
    wind_df  = classify_wind_turbines(wind_df)

    wind_groups  = wind_df.groupby(['nuts3', 'class'])['maxPower'].sum().unstack(fill_value=0.0)
    
    nuts3_to_tso_dict = {}
    for nuts3, grp in wind_df.groupby('nuts3'):
        nuts3_to_tso_dict[nuts3] = grp['tso'].iloc[0]
        
    nuts3_all = list(wind_groups.index)

    logger.info("Loading weather data and actuals...")
    weather_raw  = query_ecmwf_weather_nuts3(
        engine, '2023-01-01', '2023-12-31', nuts_prefix='DE', date_range=date_range
    )
    weather_dict = {nuts: grp.set_index('time') for nuts, grp in weather_raw.groupby('nuts_id')}

    entsoe_wind_annual, _ = query_entsoe_generation(
        engine, '2023-01-01', '2023-12-31', ZONES, date_range
    )
    assert len(entsoe_wind_annual) == len(date_range)
    act_w_tso = {tso: entsoe_wind_annual[tso] for tso in ZONES}

    # Define range of roughness values to sweep
    # In logarithmic profiles: larger z0 implies steeper wind speed gradient
    # (higher shear, multiplying 10m wind speed by a larger factor to reach hub height).
    z0_values = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8]
    
    sensitivity_results = []

    logger.info("Starting roughness length sensitivity sweep...")
    for z0 in tqdm(z0_values, desc="z0 sweep"):
        sim_w_tso = run_wind_sim_z0(z0, wind_groups, weather_dict, nuts3_all, nuts3_to_tso_dict, date_range)
        
        # Calculate metrics for each TSO zone
        for tso in ZONES:
            metrics = calculate_metrics(sim_w_tso[tso], act_w_tso[tso].reindex(date_range, fill_value=0.0))
            sensitivity_results.append({
                'z0': z0,
                'TSO': tso,
                'Actual_GWh': round(metrics['act_sum'] / 1e3, 2),
                'Sim_GWh':    round(metrics['sim_sum'] / 1e3, 2),
                'Ratio_%':    round(metrics['ratio'], 2),
                'Correlation': round(metrics['corr'], 4),
                'MAE_MW':     round(metrics['mae'], 2),
                'RMSE_MW':    round(metrics['rmse'], 2),
            })

    # Save to CSV
    res_df = pd.DataFrame(sensitivity_results)
    csv_path = result_path("wind_z0_sensitivity.csv")
    res_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # Generate a beautiful plot
    plt.figure(figsize=(10, 6))
    colors = {'DE_50HZ': 'tab:purple', 'DE_AMPRION': 'tab:red', 'DE_TENNET': 'tab:green', 'DE_TRANSNET': 'tab:blue'}
    labels = {'DE_50HZ': '50Hertz', 'DE_AMPRION': 'Amprion', 'DE_TENNET': 'TenneT', 'DE_TRANSNET': 'TransnetBW'}

    for tso in ZONES:
        t_data = res_df[res_df['TSO'] == tso]
        plt.plot(t_data['z0'], t_data['Ratio_%'], marker='o', color=colors[tso], label=labels[tso], linewidth=2)

    plt.axhline(100.0, color='black', linestyle='--', alpha=0.7, label='100% (Perfect Match)')
    # Add target line indicating typical grid/wake loss target (e.g. 90%)
    plt.axhline(90.0, color='grey', linestyle=':', alpha=0.5, label='90% (Expected target after 10% losses)')

    plt.xlabel('Roughness Length $z_0$ [m] (Logarithmic scale)', fontsize=12)
    plt.ylabel('Simulated / Observed Wind Onshore Yield Ratio (%)', fontsize=12)
    plt.title('Wind Yield Sensitivity to Roughness Length $z_0$ (Full Year 2023)', fontsize=13, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10, loc='best')
    
    plot_path = result_path("wind_z0_sensitivity.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved sensitivity plot to {plot_path}")

    # Output optimal z0 for each TSO
    print("\n=== OPTIMAL ROUGHNESS LENGTH z0 PER TSO ZONE ===")
    print("Assuming a 10% operational/wake loss target (90% simulated yield ratio):")
    for tso in ZONES:
        t_data = res_df[res_df['TSO'] == tso]
        # Find z0 that gives ratio closest to 90%
        idx = (t_data['Ratio_%'] - 90.0).abs().idxmin()
        optimal_row = t_data.loc[idx]
        print(f"  {labels[tso]:12s}: Optimal z0 = {optimal_row['z0']:.3f} m (Yield Ratio: {optimal_row['Ratio_%']:.2f}%)")


if __name__ == "__main__":
    main()
