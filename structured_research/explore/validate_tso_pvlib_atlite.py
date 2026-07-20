from pathlib import Path
import sys
import time
import logging
import numpy as np
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
import atlite
from tqdm import tqdm
from windpowerlib import ModelChain, WindTurbine
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.irradiance import erbs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tso_pvlib_atlite")

# Import modular utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    resolve_engine,
    offline_mode,
    load_plz_nuts,
    map_row_to_tso,
    classify_wind_turbines,
    parse_solar_orientation,
    query_mastr_wind,
    query_mastr_solar,
    query_entsoe_generation,
    query_ecmwf_weather_nuts3,
    calculate_metrics,
    plot_tso_timeseries_comparison,
    ensure_results_dir,
    cutout_path,
    result_path,
)

def main():
    logger.info("Initializing TSO-level PVLib/Windpowerlib vs. Atlite comparison...")
    
    # Connect to Timescale/regional DB
    print(f"Mode: {'OFFLINE (cache-only)' if offline_mode() else 'DB allowed (prefer cache)'}")
    engine = resolve_engine('timescale')
    
    start = pd.to_datetime('2023-01-01 00:00:00')
    end = pd.to_datetime('2023-01-31 23:00:00')
    date_range = pd.date_range(start, end, freq='h')
    
    # 1. Fetch ENTSO-E Actual Generation per TSO
    logger.info("Fetching ENTSO-E actual generation for TSO control areas...")
    zones = ['DE_50HZ', 'DE_AMPRION', 'DE_TENNET', 'DE_TRANSNET']
    entsoe_wind, entsoe_solar = query_entsoe_generation(engine, start, end, zones, date_range)

    # 2. Query MaStR Wind and Solar Installations (using modular functions)
    logger.info("Querying wind turbines from MaStR...")
    wind_df = query_mastr_wind(engine, start_date='2023-01-01', end_date='2023-01-31')
    
    logger.info("Querying solar PV panels from MaStR...")
    solar_df = query_mastr_solar(engine, start_date='2023-01-01', end_date='2023-01-31')
    
    # 3. Load NUTS3 mapping data
    logger.info("Mapping postcodes to NUTS3 counties...")
    plz_nuts = load_plz_nuts(engine)
    
    wind_df['nuts3'] = wind_df['plzCode'].map(plz_nuts['nuts3'])
    solar_df['nuts3'] = solar_df['plzCode'].map(plz_nuts['nuts3'])
    wind_df = wind_df.dropna(subset=['nuts3'])
    solar_df = solar_df.dropna(subset=['nuts3'])

    # Helper maps state/postcode to TSO
    wind_df['tso'] = wind_df.apply(map_row_to_tso, axis=1)
    solar_df['tso'] = solar_df.apply(map_row_to_tso, axis=1)
    
    # Drop UNKNOWNs
    wind_df = wind_df[wind_df['tso'] != 'UNKNOWN']
    solar_df = solar_df[solar_df['tso'] != 'UNKNOWN']

    # 4. Map NUTS3 Counties to TSO zones for OEDS aggregation
    logger.info("Building NUTS3 to TSO mapping dictionary...")
    nuts3_to_tso_dict = {}
    for nuts3, group in wind_df.groupby('nuts3'):
        nuts3_to_tso_dict[nuts3] = group['tso'].iloc[0]
    for nuts3, group in solar_df.groupby('nuts3'):
        if nuts3 not in nuts3_to_tso_dict:
            nuts3_to_tso_dict[nuts3] = group['tso'].iloc[0]

    # 5. OEDS Wind turbine classification & PV parsing
    logger.info("Preprocessing OEDS solar parameters and wind turbine classes...")
    wind_df = classify_wind_turbines(wind_df)
    solar_df = parse_solar_orientation(solar_df)
    
    # Group in memory
    wind_groups = wind_df.groupby(['nuts3', 'class'])['maxPower'].sum().unstack(fill_value=0.0) # in kW
    solar_groups = solar_df.groupby(['nuts3', 'azimuth', 'tilt'])['maxPower'].sum().reset_index() # in kW
    
    # 6. Retrieve ECMWF Weather and Run OEDS (pvlib/windpowerlib) Loop
    logger.info("Fetching ECMWF weather data for OEDS simulations...")
    weather_all = query_ecmwf_weather_nuts3(
        engine, start, end, nuts_prefix='DE', date_range=date_range
    )
    weather_dict = {nuts: group.set_index('time') for nuts, group in weather_all.groupby('nuts_id')}
    
    # Initialize OEDS TSO series
    oeds_wind = {tso: pd.Series(0.0, index=date_range) for tso in zones}
    oeds_solar = {tso: pd.Series(0.0, index=date_range) for tso in zones}
    
    nuts3_all = list(set(wind_groups.index).union(set(solar_groups['nuts3'])))
    turbine_models = {
        'class_low': WindTurbine(hub_height=120.0, turbine_type="V112/3000"),
        'class_med_low': WindTurbine(hub_height=105.0, turbine_type="V90/2000"),
        'class_med': WindTurbine(hub_height=100.0, turbine_type="E-82/2300"),
        'class_high': WindTurbine(hub_height=80.0, turbine_type="E-70/2000")
    }
    
    # Centroids of counties for localized solar calculations
    nuts3_coords = plz_nuts.groupby('nuts3')[['latitude', 'longitude']].mean()

    logger.info("Running county-by-county PVLib/Windpowerlib loops...")
    for nuts in tqdm(nuts3_all):
        if nuts not in weather_dict or nuts not in nuts3_to_tso_dict:
            continue
        
        tso = nuts3_to_tso_dict[nuts]
        weather_df = weather_dict[nuts]
        weather_df = weather_df.reindex(date_range, method='nearest')
        
        # Wind simulation
        if nuts in wind_groups.index:
            row = wind_groups.loc[nuts]
            data = [0.2 * np.ones(len(date_range)), weather_df["temp_air"], weather_df["wind_speed"]]
            columns = [["roughness_length", "temperature", "wind_speed"], [0, 2, 10]]
            ww = pd.DataFrame(np.asarray(data).T, index=date_range, columns=columns)
            
            for cls_name, capacity_kw in row.items():
                if capacity_kw > 0:
                    wt = turbine_models[cls_name]
                    mc = ModelChain(wt).run_model(ww)
                    cls_power = mc.power_output / wt.nominal_power * (capacity_kw * 1e3) # in W
                    oeds_wind[tso] += (cls_power / 1e6) # in MW
                    
        # Solar simulation
        nuts_solar = solar_groups[solar_groups['nuts3'] == nuts]
        if not nuts_solar.empty:
            if nuts in nuts3_coords.index:
                lat = nuts3_coords.loc[nuts, 'latitude']
                lon = nuts3_coords.loc[nuts, 'longitude']
            else:
                lat, lon = 50.0, 10.0
            location = Location(lat, lon, tz="Europe/Berlin")
            sun_pos = location.get_solarposition(date_range)
            ghi_wh = weather_df["ghi"] / 3600.0
            erbs_calc = erbs(ghi_wh, sun_pos["zenith"], date_range)
            
            for _, s_row in nuts_solar.iterrows():
                azimuth = s_row['azimuth']
                tilt = s_row['tilt']
                capacity_kw = s_row['maxPower']
                
                system = PVSystem(
                    surface_tilt=tilt,
                    surface_azimuth=azimuth,
                    module_parameters={"pdc0": capacity_kw},
                )
                irradiance = system.get_irradiance(
                    solar_zenith=sun_pos["zenith"],
                    solar_azimuth=sun_pos["azimuth"],
                    dni=erbs_calc["dni"],
                    ghi=ghi_wh,
                    dhi=erbs_calc["dhi"],
                )
                pv_power = (irradiance["poa_global"] * capacity_kw) / 1e6
                oeds_solar[tso] += pv_power

    # 7. Run Atlite TSO simulations (ERA5)
    logger.info("Running Atlite simulations per TSO zone...")
    cutout = atlite.Cutout(cutout_path("germany_2023_01.nc"))
    atlite_wind = {}
    atlite_solar = {}
    
    for tso in zones:
        # Wind
        t_wind_df = wind_df[wind_df['tso'] == tso].copy()
        t_wind_df['x'] = t_wind_df['lon']
        t_wind_df['y'] = t_wind_df['lat']
        t_wind_df['capacity_mw'] = t_wind_df['maxPower'] / 1e3
        layout_w = cutout.layout_from_capacity_list(t_wind_df, col="capacity_mw")
        w_ds = cutout.wind(turbine="Vestas_V112_3MW", layout=layout_w, add_cutout_windspeed=True)
        atlite_wind[tso] = pd.Series(w_ds.to_series().values, index=date_range)
        
        # Solar
        t_solar_df = solar_df[solar_df['tso'] == tso].copy()
        t_solar_df['x'] = t_solar_df['lon']
        t_solar_df['y'] = t_solar_df['lat']
        t_solar_df['capacity_mw'] = t_solar_df['maxPower'] / 1e3
        layout_s = cutout.layout_from_capacity_list(t_solar_df, col="capacity_mw")
        s_ds = cutout.pv(panel="CSi", orientation={"slope": 30.0, "azimuth": 180.0}, layout=layout_s)
        atlite_solar[tso] = pd.Series(s_ds.to_series().values, index=date_range)

    # 8. Compile and Calculate Validation Statistics
    logger.info("Analyzing and compiling metrics...")
    comparison_data = []
    
    sim_wind_dfs = {'OEDS (Windpowerlib)': pd.DataFrame(oeds_wind), 'Atlite (ERA5)': pd.DataFrame(atlite_wind)}
    sim_solar_dfs = {'OEDS (PVLib)': pd.DataFrame(oeds_solar), 'Atlite (ERA5)': pd.DataFrame(atlite_solar)}
    
    # Guard: ENTSO-E must match the simulation window (prevents full-year/Jan mixups)
    assert len(entsoe_wind) == len(date_range), (
        f"ENTSO-E wind length {len(entsoe_wind)} != sim window {len(date_range)}"
    )
    assert len(entsoe_solar) == len(date_range), (
        f"ENTSO-E solar length {len(entsoe_solar)} != sim window {len(date_range)}"
    )

    for tso in zones:
        act_w = entsoe_wind[tso]
        sim_w_o = oeds_wind[tso]
        sim_w_a = atlite_wind[tso]
        
        act_s = entsoe_solar[tso]
        sim_s_o = oeds_solar[tso]
        sim_s_a = atlite_solar[tso]
        
        metrics_w_o = calculate_metrics(sim_w_o, act_w)
        metrics_w_a = calculate_metrics(sim_w_a, act_w)
        metrics_s_o = calculate_metrics(sim_s_o, act_s)
        metrics_s_a = calculate_metrics(sim_s_a, act_s)
        assert metrics_w_o["n"] == len(date_range), metrics_w_o["n"]
        
        comparison_data.append({
            'TSO': tso,
            'Wind_Actual_GWh': round(metrics_w_o['act_sum'] / 1e3, 1),
            'Wind_OEDS_GWh': round(metrics_w_o['sim_sum'] / 1e3, 1),
            'Wind_Atlite_GWh': round(metrics_w_a['sim_sum'] / 1e3, 1),
            'Wind_OEDS_Ratio_%': round(metrics_w_o['ratio'], 1),
            'Wind_Atlite_Ratio_%': round(metrics_w_a['ratio'], 1),
            'Wind_OEDS_Corr': round(metrics_w_o['corr'], 4),
            'Wind_Atlite_Corr': round(metrics_w_a['corr'], 4),
            
            'Solar_Actual_GWh': round(metrics_s_o['act_sum'] / 1e3, 1),
            'Solar_OEDS_GWh': round(metrics_s_o['sim_sum'] / 1e3, 1),
            'Solar_Atlite_GWh': round(metrics_s_a['sim_sum'] / 1e3, 1),
            'Solar_OEDS_Ratio_%': round(metrics_s_o['ratio'], 1),
            'Solar_Atlite_Ratio_%': round(metrics_s_a['ratio'], 1),
            'Solar_OEDS_Corr': round(metrics_s_o['corr'], 4),
            'Solar_Atlite_Corr': round(metrics_s_a['corr'], 4)
        })
        
    comp_df = pd.DataFrame(comparison_data)
    
    ensure_results_dir()
    
    print("\n=== SIDE-BY-SIDE TSO VALIDATION: ATLITE (ERA5) VS. OEDS (PVLIB/WINDPOWERLIB + ECMWF) ===")
    print(comp_df.to_string(index=False))
    comp_df.to_csv(result_path("tso_validation_pvlib_atlite.csv"), index=False)
    
    # 9. Plotting (using modular helper)
    logger.info("Generating comparison plots...")
    
    plot_tso_timeseries_comparison(
        entsoe_wind,
        {'OEDS (Windpowerlib)': pd.DataFrame(oeds_wind), 'Atlite (ERA5)': pd.DataFrame(atlite_wind)},
        zones,
        "Wind Generation",
        "Power (MW)",
        result_path("tso_comparison_pvlib_atlite_wind.png"),
        figsize=(15, 12),
        colors_dict={'OEDS (Windpowerlib)': 'tab:red', 'Atlite (ERA5)': 'tab:blue'},
        linestyles_dict={'OEDS (Windpowerlib)': ':', 'Atlite (ERA5)': '--'}
    )
    
    plot_tso_timeseries_comparison(
        entsoe_solar,
        {'OEDS (PVLib)': pd.DataFrame(oeds_solar), 'Atlite (ERA5)': pd.DataFrame(atlite_solar)},
        zones,
        "Solar PV Generation",
        "Power (MW)",
        result_path("tso_comparison_pvlib_atlite_solar.png"),
        figsize=(15, 12),
        colors_dict={'OEDS (PVLib)': 'tab:red', 'Atlite (ERA5)': 'tab:blue'},
        linestyles_dict={'OEDS (PVLib)': ':', 'Atlite (ERA5)': '--'}
    )
    
    # 10. Write Markdown Report
    logger.info("Writing Markdown report...")
    report_content = f"""# Side-by-Side TSO Validation Report: Atlite (ERA5) vs. OEDS (PVLib/Windpowerlib + ECMWF)

To compare the physical simulation tools on a regional scale, we ran side-by-side simulations of the 4 German TSO control areas for January 2023:
1. **OEDS**: Simulated using **PVLib** and **Windpowerlib** county-by-county (NUTS3), running on ECMWF weather data.
2. **Atlite**: Simulated using a gridded **ERA5** reanalysis cutout.

---

## 1. Onshore Wind Validation (January 2023)

| TSO Control Area | Actual Yield (GWh) | OEDS Yield (GWh) | Atlite Yield (GWh) | OEDS Ratio (%) | Atlite Ratio (%) | OEDS Correlation | Atlite Correlation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DE_50HZ** | {comparison_data[0]['Wind_Actual_GWh']:.1f} | {comparison_data[0]['Wind_OEDS_GWh']:.1f} | {comparison_data[0]['Wind_Atlite_GWh']:.1f} | {comparison_data[0]['Wind_OEDS_Ratio_%']:.1f}% | {comparison_data[0]['Wind_Atlite_Ratio_%']:.1f}% | {comparison_data[0]['Wind_OEDS_Corr']:.4f} | {comparison_data[0]['Wind_Atlite_Corr']:.4f} |
| **DE_AMPRION** | {comparison_data[1]['Wind_Actual_GWh']:.1f} | {comparison_data[1]['Wind_OEDS_GWh']:.1f} | {comparison_data[1]['Wind_Atlite_GWh']:.1f} | {comparison_data[1]['Wind_OEDS_Ratio_%']:.1f}% | {comparison_data[1]['Wind_Atlite_Ratio_%']:.1f}% | {comparison_data[1]['Wind_OEDS_Corr']:.4f} | {comparison_data[1]['Wind_Atlite_Corr']:.4f} |
| **DE_TENNET** | {comparison_data[2]['Wind_Actual_GWh']:.1f} | {comparison_data[2]['Wind_OEDS_GWh']:.1f} | {comparison_data[2]['Wind_Atlite_GWh']:.1f} | {comparison_data[2]['Wind_OEDS_Ratio_%']:.1f}% | {comparison_data[2]['Wind_Atlite_Ratio_%']:.1f}% | {comparison_data[2]['Wind_OEDS_Corr']:.4f} | {comparison_data[2]['Wind_Atlite_Corr']:.4f} |
| **DE_TRANSNET** | {comparison_data[3]['Wind_Actual_GWh']:.1f} | {comparison_data[3]['Wind_OEDS_GWh']:.1f} | {comparison_data[3]['Wind_Atlite_GWh']:.1f} | {comparison_data[3]['Wind_OEDS_Ratio_%']:.1f}% | {comparison_data[3]['Wind_Atlite_Ratio_%']:.1f}% | {comparison_data[3]['Wind_OEDS_Corr']:.4f} | {comparison_data[3]['Wind_Atlite_Corr']:.4f} |

---

## 2. Solar PV Validation (January 2023)

| TSO Control Area | Actual Yield (GWh) | OEDS Yield (GWh) | Atlite Yield (GWh) | OEDS Ratio (%) | Atlite Ratio (%) | OEDS Correlation | Atlite Correlation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DE_50HZ** | {comparison_data[0]['Solar_Actual_GWh']:.1f} | {comparison_data[0]['Solar_OEDS_GWh']:.1f} | {comparison_data[0]['Solar_Atlite_GWh']:.1f} | {comparison_data[0]['Solar_OEDS_Ratio_%']:.1f}% | {comparison_data[0]['Solar_Atlite_Ratio_%']:.1f}% | {comparison_data[0]['Solar_OEDS_Corr']:.4f} | {comparison_data[0]['Solar_Atlite_Corr']:.4f} |
| **DE_AMPRION** | {comparison_data[1]['Solar_Actual_GWh']:.1f} | {comparison_data[1]['Solar_OEDS_GWh']:.1f} | {comparison_data[1]['Solar_Atlite_GWh']:.1f} | {comparison_data[1]['Solar_OEDS_Ratio_%']:.1f}% | {comparison_data[1]['Solar_Atlite_Ratio_%']:.1f}% | {comparison_data[1]['Solar_OEDS_Corr']:.4f} | {comparison_data[1]['Solar_Atlite_Corr']:.4f} |
| **DE_TENNET** | {comparison_data[2]['Solar_Actual_GWh']:.1f} | {comparison_data[2]['Solar_OEDS_GWh']:.1f} | {comparison_data[2]['Solar_Atlite_GWh']:.1f} | {comparison_data[2]['Solar_OEDS_Ratio_%']:.1f}% | {comparison_data[2]['Solar_Atlite_Ratio_%']:.1f}% | {comparison_data[2]['Solar_OEDS_Corr']:.4f} | {comparison_data[2]['Solar_Atlite_Corr']:.4f} |
| **DE_TRANSNET** | {comparison_data[3]['Solar_Actual_GWh']:.1f} | {comparison_data[3]['Solar_OEDS_GWh']:.1f} | {comparison_data[3]['Solar_Atlite_GWh']:.1f} | {comparison_data[3]['Solar_OEDS_Ratio_%']:.1f}% | {comparison_data[3]['Solar_Atlite_Ratio_%']:.1f}% | {comparison_data[3]['Solar_OEDS_Corr']:.4f} | {comparison_data[3]['Solar_Atlite_Corr']:.4f} |

---

## 3. Key Scientific Findings

(Derived from the table above — do not edit by hand.)

1. **Wind correlations**: OEDS {min(r['Wind_OEDS_Corr'] for r in comparison_data):.3f}–{max(r['Wind_OEDS_Corr'] for r in comparison_data):.3f}; Atlite {min(r['Wind_Atlite_Corr'] for r in comparison_data):.3f}–{max(r['Wind_Atlite_Corr'] for r in comparison_data):.3f}.
2. **Wind yield ratios (sim/actual)**: OEDS {min(r['Wind_OEDS_Ratio_%'] for r in comparison_data):.1f}–{max(r['Wind_OEDS_Ratio_%'] for r in comparison_data):.1f}%; Atlite {min(r['Wind_Atlite_Ratio_%'] for r in comparison_data):.1f}–{max(r['Wind_Atlite_Ratio_%'] for r in comparison_data):.1f}%.
3. **Solar correlations**: OEDS {min(r['Solar_OEDS_Corr'] for r in comparison_data):.3f}–{max(r['Solar_OEDS_Corr'] for r in comparison_data):.3f}; Atlite {min(r['Solar_Atlite_Corr'] for r in comparison_data):.3f}–{max(r['Solar_Atlite_Corr'] for r in comparison_data):.3f}.
4. **Solar yield ratios**: OEDS {min(r['Solar_OEDS_Ratio_%'] for r in comparison_data):.1f}–{max(r['Solar_OEDS_Ratio_%'] for r in comparison_data):.1f}%; Atlite {min(r['Solar_Atlite_Ratio_%'] for r in comparison_data):.1f}–{max(r['Solar_Atlite_Ratio_%'] for r in comparison_data):.1f}%.
5. **Caveats**: Weather sources differ (ECMWF IFS vs ERA5). Atlite uses uniform 30°/180° solar orientation; OEDS uses MaStR tilt/azimuth. OEDS wind uses fixed $z_0=0.2$ m logarithmic extrapolation from 10 m winds.

---

## 4. Visualizations
* [tso_comparison_pvlib_atlite_wind.png](tso_comparison_pvlib_atlite_wind.png): Wind timeseries comparison.
* [tso_comparison_pvlib_atlite_solar.png](tso_comparison_pvlib_atlite_solar.png): Solar timeseries comparison.
"""
    report_file = result_path("tso_validation_pvlib_atlite_report.md")
    with open(report_file, "w") as f:
        f.write(report_content)
    logger.info(f"Saved report to {report_file}")

if __name__ == "__main__":
    main()
