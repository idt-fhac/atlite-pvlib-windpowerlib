from pathlib import Path
"""
validate_germany_national.py
============================
January 2023 Germany-wide comparison: PVLib+Windpowerlib (ECMWF) vs Atlite (ERA5)
against ENTSO-E national feed-in. Cache-first via utils (OFFLINE_MODE supported).
"""

import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
import atlite
from tqdm import tqdm
from windpowerlib import ModelChain, WindTurbine
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.irradiance import erbs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    resolve_engine,
    offline_mode,
    load_plz_nuts,
    classify_wind_turbines,
    parse_solar_orientation,
    query_mastr_wind,
    query_mastr_solar,
    query_entsoe_generation,
    query_ecmwf_weather_nuts3,
    calculate_metrics,
    plot_timeseries_comparison,
    ensure_results_dir,
    cutout_path,
    result_path,
)

ZONES = ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET"]

TURBINE_MODELS = {
    "class_low": WindTurbine(hub_height=120.0, turbine_type="V112/3000"),
    "class_med_low": WindTurbine(hub_height=105.0, turbine_type="V90/2000"),
    "class_med": WindTurbine(hub_height=100.0, turbine_type="E-82/2300"),
    "class_high": WindTurbine(hub_height=80.0, turbine_type="E-70/2000"),
}


def main():
    print(f"Mode: {'OFFLINE (cache-only)' if offline_mode() else 'DB allowed (prefer cache)'}")
    engine = resolve_engine("timescale")

    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 31, 23)
    date_range = pd.date_range(start, end, freq="h")

    print("Loading MaStR wind/solar + NUTS3 map...")
    wind_df = query_mastr_wind(engine, "2023-01-01", "2023-01-31")
    solar_df = query_mastr_solar(engine, "2023-01-01", "2023-01-31")
    plz_nuts = load_plz_nuts(engine)

    wind_df["nuts3"] = wind_df["plzCode"].map(plz_nuts["nuts3"])
    solar_df["nuts3"] = solar_df["plzCode"].map(plz_nuts["nuts3"])
    wind_df = wind_df.dropna(subset=["nuts3"])
    solar_df = solar_df.dropna(subset=["nuts3"])
    wind_df = classify_wind_turbines(wind_df)
    solar_df = parse_solar_orientation(solar_df)

    print(
        f"Loaded {len(wind_df)} wind turbines ({wind_df['maxPower'].sum()/1e6:.2f} GW), "
        f"{len(solar_df)} solar systems ({solar_df['maxPower'].sum()/1e6:.2f} GW)"
    )

    wind_groups = wind_df.groupby(["nuts3", "class"])["maxPower"].sum().unstack(fill_value=0.0)
    solar_groups = solar_df.groupby(["nuts3", "azimuth", "tilt"])["maxPower"].sum().reset_index()

    print("Loading ECMWF NUTS3 weather (January)...")
    weather_all = query_ecmwf_weather_nuts3(
        engine, "2023-01-01", "2023-01-31", nuts_prefix="DE", date_range=date_range
    )
    weather_dict = {nuts: grp.set_index("time") for nuts, grp in weather_all.groupby("nuts_id")}

    print("Loading ENTSO-E national actuals...")
    entsoe_wind, entsoe_solar = query_entsoe_generation(
        engine, "2023-01-01", "2023-01-31", ZONES, date_range
    )
    act_wind = entsoe_wind.sum(axis=1)
    act_solar = entsoe_solar.sum(axis=1)

    print("Running PVLib + Windpowerlib county loop...")
    oeds_wind = pd.Series(0.0, index=date_range)
    oeds_solar = pd.Series(0.0, index=date_range)
    nuts3_coords = plz_nuts.groupby("nuts3")[["latitude", "longitude"]].mean()
    nuts3_all = list(set(wind_groups.index) | set(solar_groups["nuts3"]))

    for nuts in tqdm(nuts3_all, desc="OEDS NUTS3"):
        if nuts not in weather_dict:
            continue
        weather_df = weather_dict[nuts].reindex(date_range, method="nearest")

        if nuts in wind_groups.index:
            row = wind_groups.loc[nuts]
            ww = pd.DataFrame(
                np.asarray([
                    0.2 * np.ones(len(date_range)),
                    weather_df["temp_air"].values,
                    weather_df["wind_speed"].values,
                ]).T,
                index=date_range,
                columns=[["roughness_length", "temperature", "wind_speed"], [0, 2, 10]],
            )
            for cls_name, capacity in row.items():
                if capacity > 0:
                    wt = TURBINE_MODELS[cls_name]
                    mc = ModelChain(wt).run_model(ww)
                    oeds_wind += mc.power_output / wt.nominal_power * (capacity * 1e3) / 1e6

        nuts_solar = solar_groups[solar_groups["nuts3"] == nuts]
        if nuts_solar.empty:
            continue
        lat = float(nuts3_coords.loc[nuts, "latitude"]) if nuts in nuts3_coords.index else 50.0
        lon = float(nuts3_coords.loc[nuts, "longitude"]) if nuts in nuts3_coords.index else 10.0
        location = Location(lat, lon, tz="Europe/Berlin")
        sun_pos = location.get_solarposition(date_range)
        ghi_wh = weather_df["ghi"] / 3600.0
        erbs_calc = erbs(ghi_wh, sun_pos["zenith"], date_range)
        for _, s_row in nuts_solar.iterrows():
            system = PVSystem(
                surface_tilt=int(s_row["tilt"]),
                surface_azimuth=int(s_row["azimuth"]),
                module_parameters={"pdc0": float(s_row["maxPower"])},
            )
            irradiance = system.get_irradiance(
                solar_zenith=sun_pos["zenith"],
                solar_azimuth=sun_pos["azimuth"],
                dni=erbs_calc["dni"],
                ghi=ghi_wh,
                dhi=erbs_calc["dhi"],
            )
            oeds_solar += (irradiance["poa_global"] * float(s_row["maxPower"])) / 1e6

    print("Running Atlite (germany_2023_01.nc)...")
    cutout = atlite.Cutout(cutout_path("germany_2023_01.nc"))
    wind_cap = wind_df.copy()
    wind_cap["x"] = wind_cap["lon"]
    wind_cap["y"] = wind_cap["lat"]
    wind_cap["maxPower_mw"] = wind_cap["maxPower"] / 1000.0
    t0 = time.time()
    atlite_wind = cutout.wind(
        turbine="Vestas_V112_3MW",
        layout=cutout.layout_from_capacity_list(wind_cap, col="maxPower_mw"),
        add_cutout_windspeed=True,
    ).to_series()
    atlite_wind.index = date_range
    print(f"  Atlite wind done in {time.time()-t0:.1f}s")

    solar_cap = solar_df.copy()
    solar_cap["x"] = solar_cap["lon"]
    solar_cap["y"] = solar_cap["lat"]
    solar_cap["maxPower_mw"] = solar_cap["maxPower"] / 1000.0
    t0 = time.time()
    atlite_solar = cutout.pv(
        panel="CSi",
        orientation={"slope": 30.0, "azimuth": 180.0},
        layout=cutout.layout_from_capacity_list(solar_cap, col="maxPower_mw"),
    ).to_series()
    atlite_solar.index = date_range
    print(f"  Atlite solar done in {time.time()-t0:.1f}s")

    comp_df = pd.DataFrame({
        "entsoe_wind": act_wind,
        "oeds_wind": oeds_wind,
        "atlite_wind": atlite_wind,
        "entsoe_solar": act_solar,
        "oeds_solar": oeds_solar,
        "atlite_solar": atlite_solar,
    })
    ensure_results_dir()
    comp_df.to_csv(result_path("germany_entsoe_comparison.csv"))

    w_o = calculate_metrics(comp_df["oeds_wind"], comp_df["entsoe_wind"])
    w_a = calculate_metrics(comp_df["atlite_wind"], comp_df["entsoe_wind"])
    s_o = calculate_metrics(comp_df["oeds_solar"], comp_df["entsoe_solar"])
    s_a = calculate_metrics(comp_df["atlite_solar"], comp_df["entsoe_solar"])

    print("\n=== GERMANY NATIONWIDE (JANUARY 2023) ===")
    print(f"Wind  OEDS {w_o['ratio']:.1f}% r={w_o['corr']:.3f} | Atlite {w_a['ratio']:.1f}% r={w_a['corr']:.3f}")
    print(f"Solar OEDS {s_o['ratio']:.1f}% r={s_o['corr']:.3f} | Atlite {s_a['ratio']:.1f}% r={s_a['corr']:.3f}")

    pd.DataFrame({
        "Metric": [
            "Wind ENTSO-E Total (MWh)", "Wind OEDS Total (MWh)", "Wind Atlite Total (MWh)",
            "Wind OEDS Ratio (%)", "Wind Atlite Ratio (%)",
            "Wind OEDS Correlation", "Wind Atlite Correlation",
            "Wind OEDS MAE (MW)", "Wind Atlite MAE (MW)",
            "Solar ENTSO-E Total (MWh)", "Solar OEDS Total (MWh)", "Solar Atlite Total (MWh)",
            "Solar OEDS Ratio (%)", "Solar Atlite Ratio (%)",
            "Solar OEDS Correlation", "Solar Atlite Correlation",
            "Solar OEDS MAE (MW)", "Solar Atlite MAE (MW)",
        ],
        "Value": [
            w_o["act_sum"], w_o["sim_sum"], w_a["sim_sum"],
            w_o["ratio"], w_a["ratio"], w_o["corr"], w_a["corr"], w_o["mae"], w_a["mae"],
            s_o["act_sum"], s_o["sim_sum"], s_a["sim_sum"],
            s_o["ratio"], s_a["ratio"], s_o["corr"], s_a["corr"], s_o["mae"], s_a["mae"],
        ],
    }).to_csv(result_path("germany_entsoe_comparison_stats.csv"), index=False)

    plot_timeseries_comparison(
        comp_df,
        {"entsoe_wind": "ENTSO-E Actual Onshore Wind", "oeds_wind": "OEDS Simulated Wind", "atlite_wind": "Atlite Simulated Wind"},
        "Germany Onshore Wind Generation Comparison - January 2023",
        "Generation (MW)",
        result_path("germany_wind_comparison.png"),
        colors=["tab:blue", "tab:red", "tab:green"],
        linestyles=["-", "--", ":"],
    )
    plot_timeseries_comparison(
        comp_df,
        {"entsoe_solar": "ENTSO-E Actual Solar", "oeds_solar": "OEDS Simulated Solar", "atlite_solar": "Atlite Simulated Solar"},
        "Germany Solar Generation Comparison - January 2023",
        "Generation (MW)",
        result_path("germany_solar_comparison.png"),
        colors=["tab:orange", "tab:red", "tab:green"],
        linestyles=["-", "--", ":"],
    )
    print(f"Saved outputs to {ensure_results_dir()}")


if __name__ == "__main__":
    main()
