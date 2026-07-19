from pathlib import Path
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
import atlite
from windpowerlib import ModelChain, WindTurbine

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("kelmarsh_validation")

# Import modular utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    resolve_engine,
    offline_mode,
    query_ecmwf_weather,
    process_kelmarsh_scada,
    calculate_metrics,
    metrics_by_season,
    plot_duration_curves,
    plot_scatter_comparison,
    plot_timeseries_comparison,
    WORKSPACE_ROOT,
    ensure_results_dir,
    cutout_path,
    result_path,
)

def main():
    # 1. Configuration & Parameters
    logger.info("Initializing configuration...")
    
    zip_candidates = [
        WORKSPACE_ROOT / "Kelmarsh_SCADA_2023_5961.zip",
        WORKSPACE_ROOT.parent / "Kelmarsh_SCADA_2023_5961.zip",
    ]
    zip_path = next((p for p in zip_candidates if p.exists()), zip_candidates[0])
    csv_path = result_path('kelmarsh_actuals_hourly.csv')
    kelmarsh_cutout = cutout_path('kelmarsh_2023.nc')
    
    # Kelmarsh Turbine Metadata
    turbines_meta = {
        'KWF1': {'lat': 52.400604, 'lon': -0.947133, 'hub_height': 78.5, 'capacity_kw': 2050.0},
        'KWF2': {'lat': 52.402551, 'lon': -0.949527, 'hub_height': 78.5, 'capacity_kw': 2050.0},
        'KWF3': {'lat': 52.403834, 'lon': -0.94419,  'hub_height': 68.5, 'capacity_kw': 2050.0},
        'KWF4': {'lat': 52.398781, 'lon': -0.94115,  'hub_height': 78.5, 'capacity_kw': 2050.0},
        'KWF5': {'lat': 52.402308, 'lon': -0.940537, 'hub_height': 78.5, 'capacity_kw': 2050.0},
        'KWF6': {'lat': 52.400687, 'lon': -0.936093, 'hub_height': 68.5, 'capacity_kw': 2050.0}
    }
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31, 23)
    date_range = pd.date_range(start_date, end_date, freq='h')
    
    ensure_results_dir()
    kelmarsh_cutout.parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Load/Process Actual SCADA Generation Data
    if not csv_path.exists():
        if zip_path.exists():
            logger.info(f"Actuals CSV not found at {csv_path}. Processing zip archive at {zip_path}...")
            turbines_files = {
                'KWF1': 'Turbine_Data_Kelmarsh_1_2023-01-01_-_2024-01-01_228.csv',
                'KWF2': 'Turbine_Data_Kelmarsh_2_2023-01-01_-_2024-01-01_229.csv',
                'KWF3': 'Turbine_Data_Kelmarsh_3_2023-01-01_-_2024-01-01_230.csv',
                'KWF4': 'Turbine_Data_Kelmarsh_4_2023-01-01_-_2024-01-01_231.csv',
                'KWF5': 'Turbine_Data_Kelmarsh_5_2023-01-01_-_2024-01-01_232.csv',
                'KWF6': 'Turbine_Data_Kelmarsh_6_2023-01-01_-_2024-01-01_233.csv'
            }
            actuals_df = process_kelmarsh_scada(zip_path, csv_path, turbines_files, date_range)
        else:
            raise FileNotFoundError(f"SCADA zip archive not found at {zip_path} and CSV not found at {csv_path}!")
    else:
        logger.info(f"Loading actual SCADA generation data from {csv_path}...")
        actuals_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        actuals_df = actuals_df.reindex(date_range, fill_value=0.0)
    
    # Compute farm actual total
    actuals_df["Farm_actual"] = actuals_df[[f"{t}_actual" for t in turbines_meta.keys()]].sum(axis=1)
    
    # 3. Download/Load Atlite ERA5 Cutout (or replay prior farm timeseries)
    comparison_csv_path = result_path('kelmarsh_farm_comparison.csv')
    replay_only = False
    cutout = None
    if not kelmarsh_cutout.exists():
        if comparison_csv_path.exists():
            logger.warning(
                "Kelmarsh cutout missing at %s — replaying seasonal metrics from %s "
                "(set OFFLINE_MODE=0 and re-run prepare_cutouts to fully re-simulate).",
                kelmarsh_cutout, comparison_csv_path,
            )
            replay_only = True
        elif offline_mode():
            raise FileNotFoundError(
                f"OFFLINE_MODE: Kelmarsh cutout missing at {kelmarsh_cutout} and no "
                f"farm comparison cache at {comparison_csv_path}."
            )
        else:
            logger.info(f"Downloading and preparing Atlite ERA5 cutout at {kelmarsh_cutout}...")
            cutout = atlite.Cutout(
                path=kelmarsh_cutout,
                module="era5",
                x=slice(-1.5, -0.3),
                y=slice(52.0, 53.0),
                time="2023",
            )
            cutout.prepare(features=['wind', 'temperature'])
    else:
        logger.info(f"Loading existing Atlite cutout from {kelmarsh_cutout}...")
        cutout = atlite.Cutout(kelmarsh_cutout)

    if replay_only:
        comp_df = pd.read_csv(comparison_csv_path, index_col=0, parse_dates=True)
        comp_df = comp_df.reindex(date_range).fillna(0.0)
        metrics_wpl = calculate_metrics(comp_df["windpowerlib"], comp_df["actual"])
        metrics_atl = calculate_metrics(comp_df["atlite"], comp_df["actual"])
        print("\n=== KELMARSH (REPLAY FROM CACHED FARM TIMESERIES) ===")
        print(f"windpowerlib Ratio: {metrics_wpl['ratio']:.2f}% | corr {metrics_wpl['corr']:.4f}")
        print(f"Atlite Ratio:       {metrics_atl['ratio']:.2f}% | corr {metrics_atl['corr']:.4f}")
        seasonal_rows = []
        for model_name, series in [
            ("Windpowerlib (ECMWF)", comp_df["windpowerlib"]),
            ("Atlite (ERA5)", comp_df["atlite"]),
        ]:
            sdf = metrics_by_season(series, comp_df["actual"])
            sdf.insert(0, "Scale", "Single")
            sdf.insert(1, "Site", "Kelmarsh_Wind")
            sdf.insert(2, "Technology", "Wind")
            sdf.insert(3, "Model", model_name)
            seasonal_rows.append(sdf)
        pd.concat(seasonal_rows, ignore_index=True).to_csv(
            result_path("kelmarsh_seasonal_comparison.csv"), index=False
        )
        logger.info("Saved seasonal comparison from replay → kelmarsh_seasonal_comparison.csv")
        return
    
    # 4. Fetch ECMWF Weather Data for Windpowerlib (cache-first; DB only if needed)
    logger.info("Loading ECMWF weather for windpowerlib (cache-first)...")
    db_engine = resolve_engine('oeds')
    
    # Using nearest database grid point to Kelmarsh
    weather_df = query_ecmwf_weather(
        db_engine, '2023-01-01 00:00:00', '2023-12-31 23:00:00',
        nuts_id='UKF2', date_range=date_range,
    )
    weather_df = weather_df.reindex(date_range, method='nearest')
    
    # Prepare weather DataFrame for windpowerlib ModelChain (roughness = 0.2 m)
    weather_wpl = pd.DataFrame(
        np.asarray([
            0.2 * np.ones(len(weather_df.index)), # roughness
            weather_df["temp_air"],              # temperature
            weather_df["wind_speed"]             # wind speed
        ]).T,
        index=weather_df.index,
        columns=[["roughness_length", "temperature", "wind_speed"], [0, 2, 10]]
    )
    
    # 5. Extract Senvion MM92/2050 Power Curve from Windpowerlib
    logger.info("Extracting Senvion MM92/2050 power curve for simulations...")
    wt_ref = WindTurbine(hub_height=78.5, turbine_type="MM92/2050")
    
    # Extract curve variables for Atlite custom turbine dict
    wpl_v = wt_ref.power_curve["wind_speed"].tolist()
    # Convert power from W to MW
    wpl_pow_mw = (wt_ref.power_curve["value"] / 1e6).tolist()
    
    # 6. Run Turbine Simulations
    logger.info("Simulating turbines...")
    wpl_results = pd.DataFrame(index=date_range)
    atlite_results = pd.DataFrame(index=date_range)
    
    for t_name, meta in turbines_meta.items():
        logger.info(f"  Simulating turbine {t_name} (Hub Height: {meta['hub_height']}m)...")
        
        # A. Windpowerlib
        wt = WindTurbine(
            hub_height=meta['hub_height'],
            nominal_power=meta['capacity_kw'] * 1e3,
            power_curve=wt_ref.power_curve
        )
        mc = ModelChain(wt).run_model(weather_wpl)
        # Convert W to kW
        wpl_results[f"{t_name}_wpl"] = mc.power_output / 1e3
        
        # B. Atlite
        custom_turbine = {
            'POW': wpl_pow_mw,
            'V': wpl_v,
            'P': meta['capacity_kw'] / 1000.0, # in MW
            'hub_height': meta['hub_height']
        }
        
        # Atlite layout (single coordinate)
        cap_df = pd.DataFrame({
            'x': [meta['lon']],
            'y': [meta['lat']],
            'maxPower': [meta['capacity_kw'] / 1000.0] # in MW
        })
        layout = cutout.layout_from_capacity_list(cap_df, col="maxPower")
        atlite_ds = cutout.wind(turbine=custom_turbine, layout=layout)
        # Convert MW to kW
        atlite_results[f"{t_name}_atlite"] = atlite_ds.to_series().values * 1000.0
        
    # Aggregate farm totals
    wpl_results["Farm_wpl"] = wpl_results[[f"{t}_wpl" for t in turbines_meta.keys()]].sum(axis=1)
    atlite_results["Farm_atlite"] = atlite_results[[f"{t}_atlite" for t in turbines_meta.keys()]].sum(axis=1)
    
    # 7. Generate Comparison and Validation DataFrames
    logger.info("Compiling and analyzing results...")
    comp_df = pd.concat([
        actuals_df["Farm_actual"],
        wpl_results["Farm_wpl"],
        atlite_results["Farm_atlite"]
    ], axis=1)
    comp_df.columns = ["actual", "windpowerlib", "atlite"]
    
    comparison_csv_path = result_path('kelmarsh_farm_comparison.csv')
    comp_df.to_csv(comparison_csv_path)
    
    # Calculate stats
    metrics_wpl = calculate_metrics(comp_df["windpowerlib"], comp_df["actual"])
    metrics_atl = calculate_metrics(comp_df["atlite"], comp_df["actual"])
    
    print("\n=== KELMARSH WIND FARM 2023 VALIDATION (6 Turbines, 12.3 MW Total) ===")
    print(f"Actual Measured Total Yield: {metrics_wpl['act_sum'] / 1e6:.2f} GWh")
    print(f"windpowerlib Total Yield:    {metrics_wpl['sim_sum'] / 1e6:.2f} GWh | Ratio: {metrics_wpl['ratio']:.2f}%")
    print(f"Atlite Total Yield:          {metrics_atl['sim_sum'] / 1e6:.2f} GWh | Ratio: {metrics_atl['ratio']:.2f}%")
    print(f"windpowerlib -> Correlation: {metrics_wpl['corr']:.5f} | MAE: {metrics_wpl['mae']:.2f} kW | RMSE: {metrics_wpl['rmse']:.2f} kW")
    print(f"Atlite       -> Correlation: {metrics_atl['corr']:.5f} | MAE: {metrics_atl['mae']:.2f} kW | RMSE: {metrics_atl['rmse']:.2f} kW")
    
    # Write turbine-by-turbine validation table to csv
    logger.info("Generating turbine-by-turbine validation table...")
    t_stats = []
    for t in turbines_meta.keys():
        metrics_t_wpl = calculate_metrics(wpl_results[f"{t}_wpl"], actuals_df[f"{t}_actual"])
        metrics_t_atl = calculate_metrics(atlite_results[f"{t}_atlite"], actuals_df[f"{t}_actual"])
        
        t_stats.append({
            'Turbine': t,
            'Actual (GWh)': round(metrics_t_wpl['act_sum'] / 1e6, 3),
            'WPL (GWh)': round(metrics_t_wpl['sim_sum'] / 1e6, 3),
            'Atlite (GWh)': round(metrics_t_atl['sim_sum'] / 1e6, 3),
            'WPL/Act (%)': round(metrics_t_wpl['ratio'], 1),
            'Atlite/Act (%)': round(metrics_t_atl['ratio'], 1),
            'WPL_Corr': round(metrics_t_wpl['corr'], 4),
            'Atlite_Corr': round(metrics_t_atl['corr'], 4)
        })
    turbines_stats_path = result_path('kelmarsh_turbines_stats.csv')
    pd.DataFrame(t_stats).to_csv(turbines_stats_path, index=False)

    # Seasonal farm-level breakdown (Windpowerlib vs Atlite)
    seasonal_rows = []
    for model_name, series in [
        ("Windpowerlib (ECMWF)", comp_df["windpowerlib"]),
        ("Atlite (ERA5)", comp_df["atlite"]),
    ]:
        sdf = metrics_by_season(series, comp_df["actual"])
        sdf.insert(0, "Scale", "Single")
        sdf.insert(1, "Site", "Kelmarsh_Wind")
        sdf.insert(2, "Technology", "Wind")
        sdf.insert(3, "Model", model_name)
        seasonal_rows.append(sdf)
    seasonal_df = pd.concat(seasonal_rows, ignore_index=True)
    seasonal_df.to_csv(result_path("kelmarsh_seasonal_comparison.csv"), index=False)
    logger.info("Saved seasonal comparison → kelmarsh_seasonal_comparison.csv")
    
    # 8. Plot Results using modular functions
    logger.info("Generating plots...")
    
    # Plot A: Generation Duration Curve
    duration_plot_path = result_path('kelmarsh_duration_comparison.png')
    plot_duration_curves(
        {
            'Actual Measured (SCADA)': comp_df["actual"],
            'windpowerlib (ECMWF)': comp_df["windpowerlib"],
            'Atlite (ERA5)': comp_df["atlite"]
        },
        "Kelmarsh Wind Farm Generation Duration Curve (8760 Hours, 2023)",
        "Power Output (kW)",
        duration_plot_path,
        colors=['black', 'tab:red', 'tab:blue']
    )
    
    # Plot B: Scatter Plot
    scatter_plot_path = result_path('kelmarsh_scatter_comparison.png')
    plot_scatter_comparison(
        comp_df["actual"],
        {
            'windpowerlib (ECMWF)': comp_df["windpowerlib"],
            'Atlite (ERA5)': comp_df["atlite"]
        },
        "Scatter Plot: Simulated vs. Actual Wind Generation (2023) - Kelmarsh",
        "Simulated Power (kW)",
        scatter_plot_path,
        colors=['tab:red', 'tab:blue']
    )
    
    # Plot C: Sample Week in High Wind Season (Oct 15 - Oct 21, 2023)
    sample_plot_path = result_path('kelmarsh_sample_week.png')
    sample_range = slice("2023-10-15", "2023-10-21")
    plot_timeseries_comparison(
        comp_df,
        {
            "actual": "Actual Measured (SCADA)",
            "windpowerlib": "windpowerlib (ECMWF)",
            "atlite": "Atlite (ERA5)"
        },
        "Kelmarsh Wind Farm Generation - Sample Week (Oct 15-21, 2023)",
        "Power Output (kW)",
        sample_plot_path,
        sample_range=sample_range,
        colors=['black', 'tab:red', 'tab:blue'],
        linestyles=['-', '--', '-.']
    )
    
    logger.info("Kelmarsh wind validation study completed successfully.")

if __name__ == "__main__":
    main()
