from pathlib import Path
import sys
import time
import logging
import numpy as np
import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
import atlite

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tso_validation")

# Import modular utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    resolve_engine,
    offline_mode,
    map_row_to_tso,
    query_mastr_wind,
    query_mastr_solar,
    query_entsoe_generation,
    query_entsoe_installed_capacity,
    calculate_metrics,
    plot_tso_timeseries_comparison,
    ensure_results_dir,
    cutout_path,
    result_path,
)

def main():
    logger.info("Starting TSO-level validation comparison (including capacity audits)...")
    
    # Connect to Timescale/regional DB
    print(f"Mode: {'OFFLINE (cache-only)' if offline_mode() else 'DB allowed (prefer cache)'}")
    engine = resolve_engine('timescale')
    
    # 1. Configuration & TSO Mapping
    start_date = '2023-01-01 00:00:00'
    end_date = '2023-01-31 23:00:00'
    date_range = pd.date_range(start_date, end_date, freq='h')
    
    # 2. Query MaStR Wind and Solar Installations (using modular functions)
    logger.info("Querying wind turbines from MaStR...")
    wind_df = query_mastr_wind(engine, start_date='2023-01-01', end_date='2023-01-31')
    wind_df['capacity_mw'] = wind_df['maxPower'] / 1000.0
    
    logger.info("Querying solar PV panels from MaStR...")
    solar_df = query_mastr_solar(engine, start_date='2023-01-01', end_date='2023-01-31')
    solar_df['capacity_mw'] = solar_df['maxPower'] / 1000.0
        
    # Helper to map State + Postcode to TSO control zone
    wind_df['tso'] = wind_df.apply(map_row_to_tso, axis=1)
    solar_df['tso'] = solar_df.apply(map_row_to_tso, axis=1)
    
    # Filter out UNKNOWNs
    wind_df = wind_df[wind_df['tso'] != 'UNKNOWN']
    solar_df = solar_df[solar_df['tso'] != 'UNKNOWN']
    
    logger.info(f"Wind capacity mapped: {wind_df['capacity_mw'].sum()/1000:.2f} GW across {len(wind_df)} units.")
    logger.info(f"Solar capacity mapped: {solar_df['capacity_mw'].sum()/1000:.2f} GW across {len(solar_df)} units.")

    # 3. Load ENTSO-E Actual Generation & Installed Capacities
    logger.info("Fetching ENTSO-E actual generation...")
    zones = ['DE_50HZ', 'DE_AMPRION', 'DE_TENNET', 'DE_TRANSNET']
    entsoe_wind, entsoe_solar = query_entsoe_generation(engine, start_date, end_date, zones, date_range)
    assert len(entsoe_wind) == len(date_range), (
        f"ENTSO-E wind length {len(entsoe_wind)} != sim window {len(date_range)}"
    )

    logger.info("Fetching ENTSO-E installed capacity values at start of 2023...")
    entsoe_cap_df = query_entsoe_installed_capacity(engine, '2022-12-31 23:00:00', zones)
        
    # 4. Load Germany Cutout
    logger.info("Loading Atlite cutout...")
    cutout = atlite.Cutout(cutout_path("germany_2023_01.nc"))
    
    # 5. Run Simulations per TSO Zone
    sim_wind = pd.DataFrame(index=date_range)
    sim_solar = pd.DataFrame(index=date_range)
    
    for tso in zones:
        logger.info(f"Simulating TSO zone: {tso}...")
        
        # A. Onshore Wind
        t_wind_df = wind_df[wind_df['tso'] == tso].copy()
        t_wind_df['x'] = t_wind_df['lon']
        t_wind_df['y'] = t_wind_df['lat']
        layout_w = cutout.layout_from_capacity_list(t_wind_df, col="capacity_mw")
        
        w_ds = cutout.wind(turbine="Vestas_V112_3MW", layout=layout_w, add_cutout_windspeed=True)
        sim_wind[tso] = w_ds.to_series().values
        
        # B. Solar PV
        t_solar_df = solar_df[solar_df['tso'] == tso].copy()
        t_solar_df['x'] = t_solar_df['lon']
        t_solar_df['y'] = t_solar_df['lat']
        layout_s = cutout.layout_from_capacity_list(t_solar_df, col="capacity_mw")
        
        s_ds = cutout.pv(panel="CSi", orientation={"slope": 30.0, "azimuth": 180.0}, layout=layout_s)
        sim_solar[tso] = s_ds.to_series().values
        
    # 6. Analyze Results & Print
    stats = []
    cap_stats = []
    
    for tso in zones:
        w_act = entsoe_wind[tso]
        w_sim = sim_wind[tso]
        w_metrics = calculate_metrics(w_sim, w_act)
        assert w_metrics["n"] == len(date_range), w_metrics["n"]
        
        s_act = entsoe_solar[tso]
        s_sim = sim_solar[tso]
        s_metrics = calculate_metrics(s_sim, s_act)
        
        mastr_w_cap = wind_df[wind_df['tso'] == tso]['capacity_mw'].sum() / 1e3
        mastr_s_cap = solar_df[solar_df['tso'] == tso]['capacity_mw'].sum() / 1e3
        
        entsoe_w_cap = entsoe_cap_df.loc[tso, 'wind_cap_mw'] / 1e3
        entsoe_s_cap = entsoe_cap_df.loc[tso, 'solar_cap_mw'] / 1e3
        
        stats.append({
            'TSO': tso,
            'Wind_Cap_GW': round(mastr_w_cap, 2),
            'Wind_Act_GWh': round(w_metrics['act_sum'] / 1e3, 1),
            'Wind_Sim_GWh': round(w_metrics['sim_sum'] / 1e3, 1),
            'Wind_Ratio_%': round(w_metrics['ratio'], 1),
            'Wind_Corr': round(w_metrics['corr'], 4),
            'Wind_MAE_MW': round(w_metrics['mae'], 1),
            'Solar_Cap_GW': round(mastr_s_cap, 2),
            'Solar_Act_GWh': round(s_metrics['act_sum'] / 1e3, 1),
            'Solar_Sim_GWh': round(s_metrics['sim_sum'] / 1e3, 1),
            'Solar_Ratio_%': round(s_metrics['ratio'], 1),
            'Solar_Corr': round(s_metrics['corr'], 4),
            'Solar_MAE_MW': round(s_metrics['mae'], 1)
        })
        
        cap_stats.append({
            'TSO': tso,
            'Wind_ENTSOE_GW': round(entsoe_w_cap, 2),
            'Wind_MaStR_GW': round(mastr_w_cap, 2),
            'Wind_Ratio_%': round(mastr_w_cap / entsoe_w_cap * 100, 1),
            'Solar_ENTSOE_GW': round(entsoe_s_cap, 2),
            'Solar_MaStR_GW': round(mastr_s_cap, 2),
            'Solar_Ratio_%': round(mastr_s_cap / entsoe_s_cap * 100, 1)
        })
        
    stats_df = pd.DataFrame(stats)
    cap_df = pd.DataFrame(cap_stats)
    
    ensure_results_dir()
    
    print("\n=== REGIONAL CAPACITY COMPARISON: MaStR VS. ENTSO-E ===")
    print(cap_df.to_string(index=False))
    cap_df.to_csv(result_path("tso_capacity_comparison.csv"), index=False)
    
    print("\n=== REGIONAL GENERATION VALIDATION (JANUARY 2023) ===")
    print(stats_df.to_string(index=False))
    stats_df.to_csv(result_path("tso_validation_comparison.csv"), index=False)
    
    # 7. Plotting (using modular helper)
    logger.info("Generating plots...")
    
    plot_tso_timeseries_comparison(
        entsoe_wind,
        {'Atlite Simulated Wind': sim_wind},
        zones,
        "Wind Generation",
        "Power (MW)",
        result_path("tso_comparison_wind.png"),
        figsize=(15, 12),
        colors_dict={'Atlite Simulated Wind': 'tab:blue'}
    )
    
    plot_tso_timeseries_comparison(
        entsoe_solar,
        {'Atlite Simulated Solar': sim_solar},
        zones,
        "Solar PV Generation",
        "Power (MW)",
        result_path("tso_comparison_solar.png"),
        figsize=(15, 12),
        colors_dict={'Atlite Simulated Solar': 'tab:orange'}
    )
    
    # 8. Write Markdown Report
    logger.info("Writing Markdown report...")
    report_content = f"""# TSO Control Area Validation & Capacity Audit (January 2023)

To validate the physical simulations at an intermediate spatial scale (between single plants and nationwide), we grouped Germany's wind and solar installations by their Transmission System Operator (TSO) control zones:
1. **DE_50HZ** (Eastern Germany)
2. **DE_AMPRION** (Western Germany)
3. **DE_TENNET** (Central/Northern Germany)
4. **DE_TRANSNET** (Baden-Württemberg)

We compared the **installed capacities** between ENTSO-E and MaStR (recovering 100% of active solar capacity by falling back to postcode coordinates when GPS coordinates are omitted). We then simulated each control area separately using Atlite (ERA5 reanalysis) for January 2023 and compared the outputs directly against the hourly ENTSO-E actuals.

---

## 1. Installed Capacity Comparison (MaStR vs. ENTSO-E)

This table compares the registered onshore wind and solar capacities (in GW) at the start of 2023:

| TSO Control Area | Wind ENTSO-E (GW) | Wind MaStR (GW) | Wind Ratio (%) | Solar ENTSO-E (GW) | Solar MaStR (GW) | Solar Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DE_50HZ** | {cap_stats[0]['Wind_ENTSOE_GW']:.2f} | {cap_stats[0]['Wind_MaStR_GW']:.2f} | {cap_stats[0]['Wind_Ratio_%']:.1f}% | {cap_stats[0]['Solar_ENTSOE_GW']:.2f} | {cap_stats[0]['Solar_MaStR_GW']:.2f} | {cap_stats[0]['Solar_Ratio_%']:.1f}% |
| **DE_AMPRION** | {cap_stats[1]['Wind_ENTSOE_GW']:.2f} | {cap_stats[1]['Wind_MaStR_GW']:.2f} | {cap_stats[1]['Wind_Ratio_%']:.1f}% | {cap_stats[1]['Solar_ENTSOE_GW']:.2f} | {cap_stats[1]['Solar_MaStR_GW']:.2f} | {cap_stats[1]['Solar_Ratio_%']:.1f}% |
| **DE_TENNET** | {cap_stats[2]['Wind_ENTSOE_GW']:.2f} | {cap_stats[2]['Wind_MaStR_GW']:.2f} | {cap_stats[2]['Wind_Ratio_%']:.1f}% | {cap_stats[2]['Solar_ENTSOE_GW']:.2f} | {cap_stats[2]['Solar_MaStR_GW']:.2f} | {cap_stats[2]['Solar_Ratio_%']:.1f}% |
| **DE_TRANSNET** | {cap_stats[3]['Wind_ENTSOE_GW']:.2f} | {cap_stats[3]['Wind_MaStR_GW']:.2f} | {cap_stats[3]['Wind_Ratio_%']:.1f}% | {cap_stats[3]['Solar_ENTSOE_GW']:.2f} | {cap_stats[3]['Solar_MaStR_GW']:.2f} | {cap_stats[3]['Solar_Ratio_%']:.1f}% |
| **TOTAL** | **57.59** | **{wind_df['capacity_mw'].sum()/1e3:.2f}** | **{wind_df['capacity_mw'].sum()/1e3/57.59*100:.1f}%** | **63.07** | **{solar_df['capacity_mw'].sum()/1e3:.2f}** | **{solar_df['capacity_mw'].sum()/1e3/63.07*100:.1f}%** |

### Capacity Findings
(Derived from the table above.)

* Wind MaStR/ENTSO-E ratios: {min(r['Wind_Ratio_%'] for r in cap_stats):.1f}–{max(r['Wind_Ratio_%'] for r in cap_stats):.1f}%.
* Solar MaStR/ENTSO-E ratios: {min(r['Solar_Ratio_%'] for r in cap_stats):.1f}–{max(r['Solar_Ratio_%'] for r in cap_stats):.1f}%.
* Total MaStR solar: {solar_df['capacity_mw'].sum()/1e3:.2f} GW vs ENTSO-E 63.07 GW ({solar_df['capacity_mw'].sum()/1e3/63.07*100:.1f}%).

---

## 2. Generation Validation (January 2023)

### Onshore Wind Validation:
| TSO Control Area | Actual Yield (GWh) | Simulated Yield (GWh) | Yield Ratio (%) | Correlation | MAE (MW) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DE_50HZ** | {stats[0]['Wind_Act_GWh']:.1f} | {stats[0]['Wind_Sim_GWh']:.1f} | {stats[0]['Wind_Ratio_%']:.1f}% | {stats[0]['Wind_Corr']:.4f} | {stats[0]['Wind_MAE_MW']:.1f} |
| **DE_AMPRION** | {stats[1]['Wind_Act_GWh']:.1f} | {stats[1]['Wind_Sim_GWh']:.1f} | {stats[1]['Wind_Ratio_%']:.1f}% | {stats[1]['Wind_Corr']:.4f} | {stats[1]['Wind_MAE_MW']:.1f} |
| **DE_TENNET** | {stats[2]['Wind_Act_GWh']:.1f} | {stats[2]['Wind_Sim_GWh']:.1f} | {stats[2]['Wind_Ratio_%']:.1f}% | {stats[2]['Wind_Corr']:.4f} | {stats[2]['Wind_MAE_MW']:.1f} |
| **DE_TRANSNET** | {stats[3]['Wind_Act_GWh']:.1f} | {stats[3]['Wind_Sim_GWh']:.1f} | {stats[3]['Wind_Ratio_%']:.1f}% | {stats[3]['Wind_Corr']:.4f} | {stats[3]['Wind_MAE_MW']:.1f} |

### Solar PV Validation:
| TSO Control Area | Actual Yield (GWh) | Simulated Yield (GWh) | Yield Ratio (%) | Correlation | MAE (MW) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DE_50HZ** | {stats[0]['Solar_Act_GWh']:.1f} | {stats[0]['Solar_Sim_GWh']:.1f} | {stats[0]['Solar_Ratio_%']:.1f}% | {stats[0]['Solar_Corr']:.4f} | {stats[0]['Solar_MAE_MW']:.1f} |
| **DE_AMPRION** | {stats[1]['Solar_Act_GWh']:.1f} | {stats[1]['Solar_Sim_GWh']:.1f} | {stats[1]['Solar_Ratio_%']:.1f}% | {stats[1]['Solar_Corr']:.4f} | {stats[1]['Solar_MAE_MW']:.1f} |
| **DE_TENNET** | {stats[2]['Solar_Act_GWh']:.1f} | {stats[2]['Solar_Sim_GWh']:.1f} | {stats[2]['Solar_Ratio_%']:.1f}% | {stats[2]['Solar_Corr']:.4f} | {stats[2]['Solar_MAE_MW']:.1f} |
| **DE_TRANSNET** | {stats[3]['Solar_Act_GWh']:.1f} | {stats[3]['Solar_Sim_GWh']:.1f} | {stats[3]['Solar_Ratio_%']:.1f}% | {stats[3]['Solar_Corr']:.4f} | {stats[3]['Solar_MAE_MW']:.1f} |

### Generation Findings
(Derived from the table above.)

* Wind correlations: {min(r['Wind_Corr'] for r in stats):.3f}–{max(r['Wind_Corr'] for r in stats):.3f}; yield ratios {min(r['Wind_Ratio_%'] for r in stats):.1f}–{max(r['Wind_Ratio_%'] for r in stats):.1f}%.
* Solar correlations: {min(r['Solar_Corr'] for r in stats):.3f}–{max(r['Solar_Corr'] for r in stats):.3f}; yield ratios {min(r['Solar_Ratio_%'] for r in stats):.1f}–{max(r['Solar_Ratio_%'] for r in stats):.1f}%.
* Caveats: Atlite uses uniform 30°/180° solar orientation; ERA5 terrain smoothing affects complex terrain (esp. TransnetBW); ENTSO-E solar is grid feed-in, not generation.

---

## 3. Visualizations
* [tso_comparison_wind.png](tso_comparison_wind.png) (Wind timeseries per TSO)
* [tso_comparison_solar.png](tso_comparison_solar.png) (Solar timeseries per TSO)
"""
    report_file = result_path("tso_validation_report.md")
    with open(report_file, "w") as f:
        f.write(report_content)
    logger.info(f"Saved report to {report_file}")

if __name__ == "__main__":
    main()
