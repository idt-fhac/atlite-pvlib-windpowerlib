from pathlib import Path
"""
validate_annual_seasonal.py
============================
Full-year 2023 comparison of ECMWF+pvlib/windpowerlib vs. atlite (ERA5)
across all four German TSO control zones and at national level.

Runs the same physical simulation pipeline as validate_tso_pvlib_atlite.py
and validate_germany_national.py, but:
  - Covers the full year 2023 (8760 h)
  - Computes metrics for each meteorological season:
      DJF = Winter  (Dec, Jan, Feb)
      MAM = Spring  (Mar, Apr, May)
      JJA = Summer  (Jun, Jul, Aug)
      SON = Autumn  (Sep, Oct, Nov)
  - Saves per-season CSV tables and a summary figure
  - Does NOT require InfrastructureInterface / ASSUME framework

Prerequisites:
  - Run extract_research_data.py first (full-year mode) to populate
    structured_research/data/ with 2023 and seasonal Parquet files.
  - Full-year ERA5 atlite cutout at structured_research/cutouts/germany_2023.nc
    (prepare with structured_research/prepare_cutouts.py if needed).

Output files (written to structured_research/results/):
  annual_seasonal_tso_wind.csv     — per-TSO per-season wind metrics
  annual_seasonal_tso_solar.csv    — per-TSO per-season solar metrics
  annual_seasonal_national.csv     — national-level per-season metrics
  annual_seasonal_timeseries.csv   — hourly merged timeseries (national)
  annual_seasonal_summary.png      — 2×4 panel summary figure
  annual_seasonal_tso_wind_*.png   — per-season 4-panel TSO wind timeseries
  annual_seasonal_tso_solar_*.png  — per-season 4-panel TSO solar timeseries
"""

import sys
import logging
import time
import numpy as np
import pandas as pd
import atlite
from tqdm import tqdm
from windpowerlib import ModelChain, WindTurbine
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.irradiance import erbs
from pvlib.temperature import sapm_cell
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    SEASONS,
    SEASON_LABELS,
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
    ensure_results_dir,
    cutout_path,
    result_path,
)

# Canonical model labels used in paper tables / synthesis report
MODEL_OEDS = "PVLib+Windpowerlib (ECMWF)"
MODEL_ATLITE = "Atlite (ERA5)"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("annual_seasonal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ZONES = ['DE_50HZ', 'DE_AMPRION', 'DE_TENNET', 'DE_TRANSNET']
ZONE_LABELS = {
    'DE_50HZ': '50Hertz',
    'DE_AMPRION': 'Amprion',
    'DE_TENNET': 'TenneT',
    'DE_TRANSNET': 'TransnetBW',
}

TURBINE_MODELS = {
    'class_low':     WindTurbine(hub_height=120.0, turbine_type="V112/3000"),
    'class_med_low': WindTurbine(hub_height=105.0, turbine_type="V90/2000"),
    'class_med':     WindTurbine(hub_height=100.0, turbine_type="E-82/2300"),
    'class_high':    WindTurbine(hub_height=80.0,  turbine_type="E-70/2000"),
}

# Jülich PV temperature model constants (Sandia SAPM open-rack glass-glass)
SAPM_A, SAPM_B, SAPM_DT = -3.47, -0.0594, 3
TEMP_COEFF = -0.004  # silicon module: -0.4%/°C


# ---------------------------------------------------------------------------
# Core simulation functions
# ---------------------------------------------------------------------------

def run_oeds_simulation(wind_groups, solar_groups, nuts3_coords, weather_dict,
                        nuts3_all, nuts3_to_tso_dict, date_range):
    """
    Runs pvlib (solar) + windpowerlib (wind, Option-2 SP classes) for every NUTS3
    county, accumulating results per TSO zone and nationally.

    Returns:
        oeds_wind_tso  : dict[tso -> pd.Series]  MW
        oeds_solar_tso : dict[tso -> pd.Series]  MW
        oeds_wind_nat  : pd.Series  MW
        oeds_solar_nat : pd.Series  MW
    """
    oeds_wind_tso  = {tso: pd.Series(0.0, index=date_range) for tso in ZONES}
    oeds_solar_tso = {tso: pd.Series(0.0, index=date_range) for tso in ZONES}
    oeds_wind_nat  = pd.Series(0.0, index=date_range)
    oeds_solar_nat = pd.Series(0.0, index=date_range)

    for nuts in tqdm(nuts3_all, desc="OEDS county loop"):
        if nuts not in weather_dict or nuts not in nuts3_to_tso_dict:
            continue

        tso = nuts3_to_tso_dict[nuts]
        weather_df = weather_dict[nuts].reindex(date_range, method='nearest')
        temp_c     = weather_df["temp_air"] - 273.15
        wind_speed = weather_df["wind_speed"]
        ghi_wh     = weather_df["ghi"] / 3600.0

        # --- Wind ---
        if nuts in wind_groups.index:
            row = wind_groups.loc[nuts]
            ww_data = np.asarray([
                0.2 * np.ones(len(date_range)),   # roughness_length [m]
                weather_df["temp_air"].values,     # temperature [K] — wpl converts
                wind_speed.values,                 # wind speed at 10 m [m/s]
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
                    oeds_wind_tso[tso]  += cls_power_mw
                    oeds_wind_nat       += cls_power_mw

        # --- Solar ---
        nuts_solar = solar_groups[solar_groups['nuts3'] == nuts]
        if not nuts_solar.empty:
            lat = nuts3_coords.loc[nuts, 'latitude']  if nuts in nuts3_coords.index else 50.0
            lon = nuts3_coords.loc[nuts, 'longitude'] if nuts in nuts3_coords.index else 10.0

            location = Location(lat, lon, tz="Europe/Berlin")
            sun_pos  = location.get_solarposition(date_range)
            erbs_out = erbs(ghi_wh, sun_pos["zenith"], date_range)

            for _, s_row in nuts_solar.iterrows():
                azimuth     = s_row['azimuth']
                tilt        = s_row['tilt']
                capacity_kw = s_row['maxPower']

                system = PVSystem(
                    surface_tilt=tilt,
                    surface_azimuth=azimuth,
                    module_parameters={"pdc0": capacity_kw},
                )
                irr = system.get_irradiance(
                    solar_zenith=sun_pos["zenith"],
                    solar_azimuth=sun_pos["azimuth"],
                    dni=erbs_out["dni"],
                    ghi=ghi_wh,
                    dhi=erbs_out["dhi"],
                )
                poa = irr["poa_global"]
                # Temperature correction (Sandia SAPM)
                cell_temp = sapm_cell(poa, temp_c, wind_speed,
                                      SAPM_A, SAPM_B, SAPM_DT)
                temp_factor = 1 + TEMP_COEFF * (cell_temp - 25.0)
                pv_power_mw = (poa * temp_factor * capacity_kw) / 1e6

                oeds_solar_tso[tso]  += pv_power_mw
                oeds_solar_nat       += pv_power_mw

    return oeds_wind_tso, oeds_solar_tso, oeds_wind_nat, oeds_solar_nat


def run_atlite_simulation(cutout, wind_df, solar_df, date_range):
    """
    Runs Atlite for all four TSO zones and nationally.

    Returns:
        atlite_wind_tso  : dict[tso -> pd.Series]  MW
        atlite_solar_tso : dict[tso -> pd.Series]  MW
        atlite_wind_nat  : pd.Series  MW
        atlite_solar_nat : pd.Series  MW
    """
    atlite_wind_tso  = {}
    atlite_solar_tso = {}

    for tso in ZONES:
        logger.info(f"  Atlite wind  {tso}...")
        t_wind = wind_df[wind_df['tso'] == tso].copy()
        t_wind['x'] = t_wind['lon']
        t_wind['y'] = t_wind['lat']
        t_wind['capacity_mw'] = t_wind['maxPower'] / 1e3
        layout_w = cutout.layout_from_capacity_list(t_wind, col="capacity_mw")
        w_ds = cutout.wind(turbine="Vestas_V112_3MW", layout=layout_w,
                           add_cutout_windspeed=True)
        atlite_wind_tso[tso] = pd.Series(w_ds.to_series().values, index=date_range)

        logger.info(f"  Atlite solar {tso}...")
        t_solar = solar_df[solar_df['tso'] == tso].copy()
        t_solar['x'] = t_solar['lon']
        t_solar['y'] = t_solar['lat']
        t_solar['capacity_mw'] = t_solar['maxPower'] / 1e3
        layout_s = cutout.layout_from_capacity_list(t_solar, col="capacity_mw")
        s_ds = cutout.pv(panel="CSi",
                         orientation={"slope": 30.0, "azimuth": 180.0},
                         layout=layout_s)
        atlite_solar_tso[tso] = pd.Series(s_ds.to_series().values, index=date_range)

    atlite_wind_nat  = sum(atlite_wind_tso.values())
    atlite_solar_nat = sum(atlite_solar_tso.values())
    return atlite_wind_tso, atlite_solar_tso, atlite_wind_nat, atlite_solar_nat


def compute_season_metrics(sim_w_tso, sim_s_tso, sim_w_nat, sim_s_nat,
                           act_w_tso, act_s_tso, act_w_nat, act_s_nat,
                           sim_label, season_code):
    """
    Compute metrics for one simulation (OEDS or Atlite) and one season slice.

    Returns two lists (tso_rows, nat_row).
    """
    months = SEASONS[season_code]
    tso_rows = []
    for tso in ZONES:
        # Subset to season months
        w_sim = sim_w_tso[tso][sim_w_tso[tso].index.month.isin(months)]
        s_sim = sim_s_tso[tso][sim_s_tso[tso].index.month.isin(months)]
        w_act = act_w_tso[tso][act_w_tso[tso].index.month.isin(months)]
        s_act = act_s_tso[tso][act_s_tso[tso].index.month.isin(months)]

        mw = calculate_metrics(w_sim, w_act.reindex(w_sim.index, fill_value=0.0))
        ms = calculate_metrics(s_sim, s_act.reindex(s_sim.index, fill_value=0.0))

        tso_rows.append({
            'Season': SEASON_LABELS[season_code],
            'Model': sim_label,
            'TSO': tso,
            'Wind_Actual_GWh':  round(mw['act_sum'] / 1e3, 1),
            'Wind_Sim_GWh':     round(mw['sim_sum'] / 1e3, 1),
            'Wind_Ratio_%':     round(mw['ratio'],   1),
            'Wind_Corr':        round(mw['corr'],    4),
            'Wind_MAE_MW':      round(mw['mae'],     1),
            'Wind_RMSE_MW':     round(mw['rmse'],    1),
            'Solar_Actual_GWh': round(ms['act_sum'] / 1e3, 1),
            'Solar_Sim_GWh':    round(ms['sim_sum'] / 1e3, 1),
            'Solar_Ratio_%':    round(ms['ratio'],   1),
            'Solar_Corr':       round(ms['corr'],    4),
            'Solar_MAE_MW':     round(ms['mae'],     1),
            'Solar_RMSE_MW':    round(ms['rmse'],    1),
        })

    # National
    w_sim = sim_w_nat[sim_w_nat.index.month.isin(months)]
    s_sim = sim_s_nat[sim_s_nat.index.month.isin(months)]
    w_act = act_w_nat[act_w_nat.index.month.isin(months)]
    s_act = act_s_nat[act_s_nat.index.month.isin(months)]

    mw = calculate_metrics(w_sim, w_act.reindex(w_sim.index, fill_value=0.0))
    ms = calculate_metrics(s_sim, s_act.reindex(s_sim.index, fill_value=0.0))

    nat_row = {
        'Season': SEASON_LABELS[season_code],
        'Model': sim_label,
        'Wind_Actual_GWh':  round(mw['act_sum'] / 1e3, 1),
        'Wind_Sim_GWh':     round(mw['sim_sum'] / 1e3, 1),
        'Wind_Ratio_%':     round(mw['ratio'],   1),
        'Wind_Corr':        round(mw['corr'],    4),
        'Wind_MAE_MW':      round(mw['mae'],     1),
        'Wind_RMSE_MW':     round(mw['rmse'],    1),
        'Solar_Actual_GWh': round(ms['act_sum'] / 1e3, 1),
        'Solar_Sim_GWh':    round(ms['sim_sum'] / 1e3, 1),
        'Solar_Ratio_%':    round(ms['ratio'],   1),
        'Solar_Corr':       round(ms['corr'],    4),
        'Solar_MAE_MW':     round(ms['mae'],     1),
        'Solar_RMSE_MW':    round(ms['rmse'],    1),
    }

    return tso_rows, nat_row


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SEASON_COLORS = {
    'DJF': '#4878CF',  # blue
    'MAM': '#6ACC65',  # green
    'JJA': '#D65F5F',  # red
    'SON': '#B47CC7',  # purple
}
MODEL_COLORS = {
    MODEL_OEDS: 'tab:red',
    MODEL_ATLITE: 'tab:blue',
}
MODEL_LINESTYLES = {
    MODEL_OEDS: ':',
    MODEL_ATLITE: '--',
}


def plot_seasonal_summary(tso_wind_rows, tso_solar_rows, nat_rows, results_dir):
    """
    2×4 panel plot:
      Row 0: Wind Correlation per season (national, OEDS vs Atlite)
      Row 1: Wind Yield Ratio per season
      Row 2: Solar Correlation per season
      Row 3: Solar Yield Ratio per season
    """
    season_list = ['DJF', 'MAM', 'JJA', 'SON']
    season_names = [SEASON_LABELS[s] for s in season_list]
    nat_df = pd.DataFrame(nat_rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Full-Year 2023: Seasonal Comparison — pvlib+ECMWF vs. Atlite (ERA5)\n"
                 "Germany Nationwide vs. ENTSO-E Actuals",
                 fontsize=13, fontweight='bold', y=1.01)

    metrics = [
        ('Wind_Corr',    'Wind Correlation (r)',          axes[0, 0], (0.6, 1.0)),
        ('Wind_Ratio_%', 'Wind Yield Ratio (%)',           axes[0, 1], (50, 150)),
        ('Solar_Corr',   'Solar Correlation (r)',          axes[1, 0], (0.5, 1.0)),
        ('Solar_Ratio_%','Solar Yield Ratio (%)',          axes[1, 1], (50, 400)),
    ]

    models = [MODEL_OEDS, MODEL_ATLITE]
    x = np.arange(len(season_list))
    width = 0.35

    for metric, ylabel, ax, ylim in metrics:
        for i, model in enumerate(models):
            m_df = nat_df[nat_df['Model'] == model].set_index('Season')
            if m_df.empty:
                logger.warning("No national rows for model %s — skipping bars", model)
                continue
            vals = []
            for s in season_list:
                label = SEASON_LABELS[s]
                vals.append(float(m_df.loc[label, metric]) if label in m_df.index else float("nan"))
            offset = (i - 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=model,
                   color=MODEL_COLORS.get(model, f"C{i}"),
                   alpha=0.85)

        if 'Ratio' in metric:
            ax.axhline(100.0, color='black', linestyle='--', linewidth=1.2, alpha=0.7,
                       label='Perfect match (100%)')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(season_names, fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = (results_dir / "annual_seasonal_summary.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved summary plot → {path}")


def plot_monthly_timeseries(sim_w_nat, sim_s_nat, atlite_w_nat, atlite_s_nat,
                             act_w_nat, act_s_nat, results_dir):
    """Monthly mean power plot (national, wind + solar)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    months = pd.date_range('2023-01', '2024-01', freq='MS')
    month_labels = [m.strftime('%b') for m in months[:-1]]
    x = np.arange(12)

    def monthly_means(series):
        return [series[series.index.month == m].mean() for m in range(1, 13)]

    for ax, (oeds_s, atl_s, act_s, tech, unit) in [
        (ax1, (sim_w_nat, atlite_w_nat, act_w_nat, 'Wind Onshore', 'MW')),
        (ax2, (sim_s_nat, atlite_s_nat, act_s_nat, 'Solar PV', 'MW')),
    ]:
        width = 0.28
        ax.bar(x - width, monthly_means(act_s),  width, label='ENTSO-E Actual', color='black', alpha=0.7)
        ax.bar(x,         monthly_means(oeds_s), width, label='OEDS (pvlib+ECMWF)', color='tab:red', alpha=0.8)
        ax.bar(x + width, monthly_means(atl_s),  width, label='Atlite (ERA5)', color='tab:blue', alpha=0.8)
        ax.set_ylabel(f'Mean {tech} Power ({unit})', fontsize=11)
        ax.set_title(f'Monthly Mean {tech} Generation — Germany 2023', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(month_labels)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = (results_dir / "annual_monthly_timeseries.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved monthly timeseries plot → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("validate_annual_seasonal.py — Full-year 2023 + 4 Seasons")
    logger.info("=" * 70)

    results_dir = ensure_results_dir()
    logger.info(f"Mode: {'OFFLINE (cache-only)' if offline_mode() else 'DB allowed (prefer cache)'}")

    # -----------------------------------------------------------------------
    # Date range — full year 2023
    # -----------------------------------------------------------------------
    full_start = pd.to_datetime('2023-01-01 00:00:00')
    full_end   = pd.to_datetime('2023-12-31 23:00:00')
    date_range = pd.date_range(full_start, full_end, freq='h')
    logger.info(f"Date range: {full_start.date()} → {full_end.date()}  ({len(date_range)} hours)")

    # -----------------------------------------------------------------------
    # 1. Load plant data and mappings
    # -----------------------------------------------------------------------
    logger.info("Loading MaStR wind and solar data...")
    engine = resolve_engine('timescale')

    wind_df  = query_mastr_wind(engine, '2023-01-01', '2023-12-31')
    solar_df = query_mastr_solar(engine, '2023-01-01', '2023-12-31')
    plz_nuts = load_plz_nuts(engine)

    # Map to NUTS3
    wind_df['nuts3']  = wind_df['plzCode'].map(plz_nuts['nuts3'])
    solar_df['nuts3'] = solar_df['plzCode'].map(plz_nuts['nuts3'])
    wind_df  = wind_df.dropna(subset=['nuts3'])
    solar_df = solar_df.dropna(subset=['nuts3'])

    # Map to TSO
    wind_df['tso']  = wind_df.apply(map_row_to_tso, axis=1)
    solar_df['tso'] = solar_df.apply(map_row_to_tso, axis=1)
    wind_df  = wind_df[wind_df['tso'] != 'UNKNOWN']
    solar_df = solar_df[solar_df['tso'] != 'UNKNOWN']

    logger.info(f"  Wind:  {len(wind_df):,} turbines  {wind_df['maxPower'].sum()/1e6:.2f} GW")
    logger.info(f"  Solar: {len(solar_df):,} systems  {solar_df['maxPower'].sum()/1e6:.2f} GW")

    # TSO→NUTS3 mapping dict
    nuts3_to_tso_dict = {}
    for nuts3, grp in wind_df.groupby('nuts3'):
        nuts3_to_tso_dict[nuts3] = grp['tso'].iloc[0]
    for nuts3, grp in solar_df.groupby('nuts3'):
        if nuts3 not in nuts3_to_tso_dict:
            nuts3_to_tso_dict[nuts3] = grp['tso'].iloc[0]

    # Classify and group
    wind_df  = classify_wind_turbines(wind_df)
    solar_df = parse_solar_orientation(solar_df)

    wind_groups  = wind_df.groupby(['nuts3', 'class'])['maxPower'].sum().unstack(fill_value=0.0)
    solar_groups = solar_df.groupby(['nuts3', 'azimuth', 'tilt'])['maxPower'].sum().reset_index()
    nuts3_coords = plz_nuts.groupby('nuts3')[['latitude', 'longitude']].mean()
    nuts3_all    = list(set(wind_groups.index) | set(solar_groups['nuts3']))

    # -----------------------------------------------------------------------
    # 2. Load full-year ECMWF NUTS3 weather
    # -----------------------------------------------------------------------
    logger.info("Loading full-year ECMWF NUTS3 weather (may take a moment)...")
    weather_raw  = query_ecmwf_weather_nuts3(
        engine, '2023-01-01', '2023-12-31', nuts_prefix='DE', date_range=date_range
    )
    weather_dict = {nuts: grp.set_index('time') for nuts, grp in weather_raw.groupby('nuts_id')}
    logger.info(f"  Loaded weather for {len(weather_dict)} NUTS3 regions.")

    # -----------------------------------------------------------------------
    # 3. Load full-year ENTSO-E actuals
    # -----------------------------------------------------------------------
    logger.info("Loading full-year ENTSO-E actual generation...")
    entsoe_wind_annual, entsoe_solar_annual = query_entsoe_generation(
        engine, '2023-01-01', '2023-12-31', ZONES, date_range
    )
    act_w_tso = {tso: entsoe_wind_annual[tso]  for tso in ZONES}
    act_s_tso = {tso: entsoe_solar_annual[tso] for tso in ZONES}
    act_w_nat = entsoe_wind_annual.sum(axis=1)
    act_s_nat = entsoe_solar_annual.sum(axis=1)

    # -----------------------------------------------------------------------
    # 4. Run OEDS simulation (full year — county loop)
    # -----------------------------------------------------------------------
    logger.info("Running OEDS (pvlib + windpowerlib) full-year simulation...")
    t0 = time.time()
    oeds_wind_tso, oeds_solar_tso, oeds_wind_nat, oeds_solar_nat = run_oeds_simulation(
        wind_groups, solar_groups, nuts3_coords, weather_dict,
        nuts3_all, nuts3_to_tso_dict, date_range
    )
    logger.info(f"  OEDS simulation finished in {time.time() - t0:.1f} s")

    # -----------------------------------------------------------------------
    # 5. Run Atlite simulation (full year — needs germany_2023.nc)
    # -----------------------------------------------------------------------
    germany_cutout = cutout_path("germany_2023.nc")
    if not germany_cutout.exists():
        logger.error(
            f"Full-year Germany atlite cutout not found at: {germany_cutout}\n"
            "Please run:  python structured_research/prepare_cutouts.py\n"
            "Skipping Atlite simulation — OEDS results will still be saved."
        )
        atlite_wind_tso  = {tso: pd.Series(np.nan, index=date_range) for tso in ZONES}
        atlite_solar_tso = {tso: pd.Series(np.nan, index=date_range) for tso in ZONES}
        atlite_wind_nat  = pd.Series(np.nan, index=date_range)
        atlite_solar_nat = pd.Series(np.nan, index=date_range)
        atlite_available = False
    else:
        logger.info(f"Loading Atlite cutout from {germany_cutout} ...")
        t0 = time.time()
        cutout = atlite.Cutout(germany_cutout)
        atlite_wind_tso, atlite_solar_tso, atlite_wind_nat, atlite_solar_nat = \
            run_atlite_simulation(cutout, wind_df, solar_df, date_range)
        logger.info(f"  Atlite simulation finished in {time.time() - t0:.1f} s")
        atlite_available = True

    # -----------------------------------------------------------------------
    # 6. Compute seasonal metrics and save
    # -----------------------------------------------------------------------
    logger.info("Computing seasonal metrics...")
    tso_wind_rows = []
    tso_solar_rows = []
    nat_rows = []

    model_pairs = [
        (MODEL_OEDS,
         oeds_wind_tso, oeds_solar_tso, oeds_wind_nat, oeds_solar_nat),
    ]
    if atlite_available:
        model_pairs.append(
            (MODEL_ATLITE,
             atlite_wind_tso, atlite_solar_tso, atlite_wind_nat, atlite_solar_nat)
        )

    for season_code in SEASONS.keys():
        for model_label, sim_w_tso, sim_s_tso, sim_w_nat, sim_s_nat in model_pairs:
            t_rows, n_row = compute_season_metrics(
                sim_w_tso, sim_s_tso, sim_w_nat, sim_s_nat,
                act_w_tso, act_s_tso, act_w_nat, act_s_nat,
                model_label, season_code
            )
            tso_wind_rows.extend(t_rows)
            nat_rows.append(n_row)

    # Also compute full-year metrics
    for model_label, sim_w_tso, sim_s_tso, sim_w_nat, sim_s_nat in model_pairs:
        for tso in ZONES:
            mw = calculate_metrics(sim_w_tso[tso], act_w_tso[tso].reindex(date_range, fill_value=0.0))
            ms = calculate_metrics(sim_s_tso[tso], act_s_tso[tso].reindex(date_range, fill_value=0.0))
            tso_wind_rows.append({
                'Season': 'Full Year', 'Model': model_label, 'TSO': tso,
                'Wind_Actual_GWh':  round(mw['act_sum'] / 1e3, 1),
                'Wind_Sim_GWh':     round(mw['sim_sum'] / 1e3, 1),
                'Wind_Ratio_%':     round(mw['ratio'], 1),
                'Wind_Corr':        round(mw['corr'], 4),
                'Wind_MAE_MW':      round(mw['mae'], 1),
                'Wind_RMSE_MW':     round(mw['rmse'], 1),
                'Solar_Actual_GWh': round(ms['act_sum'] / 1e3, 1),
                'Solar_Sim_GWh':    round(ms['sim_sum'] / 1e3, 1),
                'Solar_Ratio_%':    round(ms['ratio'], 1),
                'Solar_Corr':       round(ms['corr'], 4),
                'Solar_MAE_MW':     round(ms['mae'], 1),
                'Solar_RMSE_MW':    round(ms['rmse'], 1),
            })
        mw = calculate_metrics(sim_w_nat, act_w_nat.reindex(date_range, fill_value=0.0))
        ms = calculate_metrics(sim_s_nat, act_s_nat.reindex(date_range, fill_value=0.0))
        nat_rows.append({
            'Season': 'Full Year', 'Model': model_label,
            'Wind_Actual_GWh':  round(mw['act_sum'] / 1e3, 1),
            'Wind_Sim_GWh':     round(mw['sim_sum'] / 1e3, 1),
            'Wind_Ratio_%':     round(mw['ratio'], 1),
            'Wind_Corr':        round(mw['corr'], 4),
            'Wind_MAE_MW':      round(mw['mae'], 1),
            'Wind_RMSE_MW':     round(mw['rmse'], 1),
            'Solar_Actual_GWh': round(ms['act_sum'] / 1e3, 1),
            'Solar_Sim_GWh':    round(ms['sim_sum'] / 1e3, 1),
            'Solar_Ratio_%':    round(ms['ratio'], 1),
            'Solar_Corr':       round(ms['corr'], 4),
            'Solar_MAE_MW':     round(ms['mae'], 1),
            'Solar_RMSE_MW':    round(ms['rmse'], 1),
        })

    # Save CSVs
    pd.DataFrame(tso_wind_rows).to_csv(
        (results_dir / "annual_seasonal_tso.csv"), index=False
    )
    pd.DataFrame(nat_rows).to_csv(
        (results_dir / "annual_seasonal_national.csv"), index=False
    )
    logger.info("Saved seasonal metrics CSVs.")

    # Save full-year hourly timeseries (national)
    ts_df = pd.DataFrame({
        'entsoe_wind':        act_w_nat,
        'entsoe_solar':       act_s_nat,
        'oeds_wind':          oeds_wind_nat,
        'oeds_solar':         oeds_solar_nat,
        'atlite_wind':        atlite_wind_nat,
        'atlite_solar':       atlite_solar_nat,
    })
    ts_df.to_csv((results_dir / "annual_seasonal_timeseries.csv"))
    logger.info("Saved full-year hourly national timeseries.")

    # Save full-year hourly TSO timeseries (for week plots / day-night diagnostics)
    tso_ts = pd.DataFrame(index=date_range)
    for tso in ZONES:
        tso_ts[f'entsoe_wind_{tso}'] = act_w_tso[tso].reindex(date_range, fill_value=0.0)
        tso_ts[f'entsoe_solar_{tso}'] = act_s_tso[tso].reindex(date_range, fill_value=0.0)
        tso_ts[f'oeds_wind_{tso}'] = oeds_wind_tso[tso].reindex(date_range)
        tso_ts[f'oeds_solar_{tso}'] = oeds_solar_tso[tso].reindex(date_range)
        tso_ts[f'atlite_wind_{tso}'] = atlite_wind_tso[tso].reindex(date_range)
        tso_ts[f'atlite_solar_{tso}'] = atlite_solar_tso[tso].reindex(date_range)
    tso_ts_path = (results_dir / "annual_seasonal_tso_timeseries.parquet")
    tso_ts.to_parquet(tso_ts_path)
    logger.info(f"Saved full-year hourly TSO timeseries → {tso_ts_path}")

    # -----------------------------------------------------------------------
    # 7. Print summary tables
    # -----------------------------------------------------------------------
    nat_df = pd.DataFrame(nat_rows)
    print("\n=== NATIONAL GERMANY — SEASONAL COMPARISON 2023 ===")
    print(nat_df.to_string(index=False))

    tso_df = pd.DataFrame(tso_wind_rows)
    print("\n=== TSO-LEVEL — SEASONAL COMPARISON 2023 ===")
    print(tso_df[tso_df['Season'] == 'Full Year'].to_string(index=False))

    # -----------------------------------------------------------------------
    # 8. Plots
    # -----------------------------------------------------------------------
    if atlite_available:
        logger.info("Generating plots...")
        plot_seasonal_summary(tso_wind_rows, tso_solar_rows, nat_rows, results_dir)
        plot_monthly_timeseries(
            oeds_wind_nat, oeds_solar_nat,
            atlite_wind_nat, atlite_solar_nat,
            act_w_nat, act_s_nat,
            results_dir
        )
    else:
        logger.warning("Atlite cutout not available — skipping comparative plots.")

    elapsed = time.time() - t_start
    logger.info(f"validate_annual_seasonal.py completed in {elapsed:.0f} s ({elapsed/60:.1f} min).")


if __name__ == "__main__":
    main()
