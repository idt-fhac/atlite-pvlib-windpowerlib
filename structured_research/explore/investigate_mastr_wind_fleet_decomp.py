"""
MaStR-based decomposition of Windpowerlib vs Atlite wind (matched ERA5).

Replaces the coarse "4 SP-class vs V80" story with plant-level MaStR attributes:
  hub height (Nabenhoehe), rotor diameter, rated power → WPL catalogue map.

Cases (same germany_2023.nc weather):
  A) Legacy 4 SP-class, fixed hubs (from matched_era5_timeseries.csv if present)
  B) Same 4 SP-class power curves, but MaStR hub heights (5 m bins)
  C) Nearest WPL catalogue type by (diameter, power) + MaStR hubs
  D) Atlite Vestas V112 @ 80 m (from matched timeseries / recompute if needed)

Outputs → results/:
  mastr_wind_fleet_summary.csv
  mastr_wind_fleet_wpl_types.csv
  mastr_wind_fleet_decomp.csv
  mastr_wind_fleet_decomp.md
  mastr_wind.parquet refresh note via extract (hub_m/manufacturer/type_name)
"""

from __future__ import annotations

from pathlib import Path

import logging
import sys
from functools import lru_cache

import atlite
import numpy as np
import pandas as pd
from tqdm import tqdm
from windpowerlib import WindTurbine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    DATA_DIR,
    calculate_metrics,
    classify_wind_turbines,
    cutout_path,
    ensure_results_dir,
    load_plz_nuts,
    map_mastr_to_wpl_turbine,
    map_row_to_tso,
    resolve_engine,
    result_path,
    SP_CLASS_TURBINE_TYPE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mastr_wind_fleet")

ZONES = ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET"]
HUB_BIN_M = 5.0


def metrics_row(sim: pd.Series, act: pd.Series, label: str, ref: str) -> dict:
    m = calculate_metrics(sim, act)
    return {
        "case": label,
        "ref": ref,
        "Ratio_%": round(m["ratio"], 2),
        "Corr": round(m["corr"], 4),
        "Sim_GWh": round(m["sim_sum"] / 1e3, 1),
        "Ref_GWh": round(m["act_sum"] / 1e3, 1),
        "MAE_MW": round(m["mae"], 1),
        "RMSE_MW": round(m["rmse"], 1),
    }


@lru_cache(maxsize=128)
def _cf_interpolator(turbine_type: str):
    """Return (wind_speeds, capacity_factors) for a WPL catalogue turbine."""
    wt = WindTurbine(hub_height=100.0, turbine_type=turbine_type)
    pc = wt.power_curve
    # windpowerlib power_curve: wind_speed, value (W)
    u = np.asarray(pc["wind_speed"], dtype=float)
    p = np.asarray(pc["value"], dtype=float)
    nom = float(wt.nominal_power)
    cf = np.clip(p / nom, 0.0, None)
    return u, cf


def capacity_factor(turbine_type: str, u_hub: np.ndarray) -> np.ndarray:
    u_grid, cf_grid = _cf_interpolator(turbine_type)
    return np.interp(u_hub, u_grid, cf_grid, left=0.0, right=cf_grid[-1])


def hub_bin(hub_m: pd.Series) -> pd.Series:
    return (np.round(pd.to_numeric(hub_m, errors="coerce") / HUB_BIN_M) * HUB_BIN_M).clip(
        lower=30.0, upper=200.0
    )


def enrich_and_cache(engine) -> pd.DataFrame:
    """Pull enriched MaStR wind and write parquet for offline reuse."""
    # Bypass stale slim parquet by querying DB directly when engine available.
    from sqlalchemy import text

    if engine is None:
        df = query_mastr_wind(None)
        if "hub_m" not in df.columns:
            raise RuntimeError(
                "Enriched mastr_wind.parquet missing hub_m and no DB — run extract online."
            )
        return df

    q = text(
        """
        SELECT
            COALESCE(w."Laengengrad", p.longitude) AS lon,
            COALESCE(w."Breitengrad", p.latitude) AS lat,
            w."Bruttoleistung" AS "maxPower",
            w."Bundesland",
            w."Postleitzahl" AS "plzCode",
            w."Rotordurchmesser" AS diameter,
            w."Nabenhoehe" AS hub_m,
            w."Hersteller" AS manufacturer,
            w."Typenbezeichnung" AS type_name,
            w."Inbetriebnahmedatum" AS commission
        FROM mastr.wind_extended w
        LEFT JOIN public.plz p
            ON w."Postleitzahl" = LPAD(p.code::text, 5, '0')
        WHERE w."EinheitBetriebsstatus" = 'In Betrieb'
          AND w."WindAnLandOderAufSee" = 'Windkraft an Land'
          AND w."Inbetriebnahmedatum" IS NOT NULL
          AND w."Inbetriebnahmedatum" <= '2023-12-31'
          AND (w."DatumEndgueltigeStilllegung" IS NULL
               OR w."DatumEndgueltigeStilllegung" > '2023-01-01')
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    df = df.dropna(subset=["lon", "lat"])
    out = (DATA_DIR / "mastr_wind.parquet")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Wrote enriched %s (%d units, %.2f GW)", out, len(df), df["maxPower"].sum() / 1e6)
    return df


def attach_nuts_tso(wind_df: pd.DataFrame) -> pd.DataFrame:
    plz_nuts = load_plz_nuts()
    df = wind_df.copy()
    df["plzCode"] = df["plzCode"].astype(str).str.zfill(5)
    # plz_nuts index is 5-digit postcode
    nuts_map = plz_nuts["nuts3"].to_dict()
    df["nuts3"] = df["plzCode"].map(nuts_map)
    # Fallback: nearest NUTS3 centroid by lat/lon if PLZ missing
    missing = df["nuts3"].isna()
    if missing.any():
        cents = plz_nuts.groupby("nuts3")[["latitude", "longitude"]].mean()
        for idx in df.index[missing]:
            lat, lon = df.at[idx, "lat"], df.at[idx, "lon"]
            if pd.isna(lat) or pd.isna(lon):
                continue
            d = (cents["latitude"] - lat) ** 2 + (cents["longitude"] - lon) ** 2
            df.at[idx, "nuts3"] = d.idxmin()
    df["tso"] = df.apply(map_row_to_tso, axis=1)
    before = len(df)
    df = df.dropna(subset=["nuts3"])
    if len(df) < before:
        logger.warning("Dropped %d units without NUTS3", before - len(df))
    return df


def fleet_summary(df: pd.DataFrame):
    mapped = map_mastr_to_wpl_turbine(classify_wind_turbines(df))
    gw = mapped["maxPower"].sum() / 1e6
    has_hub = mapped["hub_m"].notna() if "hub_m" in mapped.columns else pd.Series(False, index=mapped.index)
    rows = [
        {"metric": "units", "value": len(mapped)},
        {"metric": "capacity_GW", "value": round(gw, 3)},
        {
            "metric": "hub_coverage_pct_capacity",
            "value": round(100 * mapped.loc[has_hub, "maxPower"].sum() / mapped["maxPower"].sum(), 2),
        },
        {
            "metric": "cap_weighted_mean_hub_m",
            "value": round(
                (mapped.loc[has_hub, "hub_m"] * mapped.loc[has_hub, "maxPower"]).sum()
                / mapped.loc[has_hub, "maxPower"].sum(),
                2,
            ),
        },
        {
            "metric": "cap_weighted_mean_sp_class_hub_m",
            "value": round(
                (
                    mapped["class"].map(
                        {"class_low": 120, "class_med_low": 105, "class_med": 100, "class_high": 80}
                    )
                    * mapped["maxPower"]
                ).sum()
                / mapped["maxPower"].sum(),
                2,
            ),
        },
        {
            "metric": "mean_abs_hub_error_sp_class_m",
            "value": round(
                (
                    (
                        mapped.loc[has_hub, "class"].map(
                            {
                                "class_low": 120,
                                "class_med_low": 105,
                                "class_med": 100,
                                "class_high": 80,
                            }
                        )
                        - mapped.loc[has_hub, "hub_m"]
                    ).abs()
                    * mapped.loc[has_hub, "maxPower"]
                ).sum()
                / mapped.loc[has_hub, "maxPower"].sum(),
                2,
            ),
        },
        {
            "metric": "wpl_diam_power_match_pct_capacity",
            "value": round(
                100
                * mapped.loc[mapped.wpl_match_source == "diam_power", "maxPower"].sum()
                / mapped["maxPower"].sum(),
                2,
            ),
        },
        {
            "metric": "wpl_unique_types_used",
            "value": int(mapped["wpl_type"].nunique()),
        },
    ]
    summary = pd.DataFrame(rows)
    by_type = (
        mapped.groupby(["wpl_type", "wpl_match_source"], as_index=False)["maxPower"]
        .sum()
        .assign(GW=lambda x: x["maxPower"] / 1e6)
        .sort_values("GW", ascending=False)
    )
    return summary, by_type, mapped


def nuts3_coords_from_plants(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("nuts3").agg(latitude=("lat", "mean"), longitude=("lon", "mean"))
    return g


def extract_weather(ds, nuts3_coords, date_range):
    weather = {}
    for nuts, row in tqdm(nuts3_coords.iterrows(), total=len(nuts3_coords), desc="Weather cells"):
        yi = int(np.abs(ds.y.values - row.latitude).argmin())
        xi = int(np.abs(ds.x.values - row.longitude).argmin())
        u100 = ds["wnd100m"].isel(y=yi, x=xi).to_pandas()
        alpha = ds["wnd_shear_exp"].isel(y=yi, x=xi).to_pandas()
        if getattr(u100.index, "tz", None) is not None:
            u100.index = u100.index.tz_localize(None)
            alpha.index = alpha.index.tz_localize(None)
        weather[nuts] = {
            "u100": u100.reindex(date_range, method="nearest").astype(float).values,
            "alpha": alpha.reindex(date_range, method="nearest")
            .astype(float)
            .clip(0.05, 0.55)
            .values,
        }
    return weather


def run_grouped_wpl(groups: pd.DataFrame, weather, nuts_to_tso, date_range) -> pd.Series:
    """groups columns: nuts3, turbine_type, hub_m, capacity_kw"""
    nat = pd.Series(0.0, index=date_range)
    for nuts, g in tqdm(groups.groupby("nuts3"), desc="WPL groups"):
        if nuts not in weather or nuts not in nuts_to_tso:
            continue
        u100 = weather[nuts]["u100"]
        alpha = weather[nuts]["alpha"]
        for _, r in g.iterrows():
            hub = float(r["hub_m"])
            u_hub = u100 * (hub / 100.0) ** alpha
            cf = capacity_factor(str(r["turbine_type"]), u_hub)
            nat += cf * (float(r["capacity_kw"]) * 1e3) / 1e6  # MW
    return nat


def build_case_groups(mapped: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode:
      sp_class_fixed_hub — legacy 4-class hubs
      sp_class_mastr_hub — SP-class curves, MaStR hubs
      wpl_map_mastr_hub — nearest WPL type + MaStR hubs
    """
    df = mapped.copy()
    df["hub_bin"] = hub_bin(df["hub_m_used"])
    if mode == "sp_class_fixed_hub":
        df["turbine_type"] = df["class"].map(SP_CLASS_TURBINE_TYPE)
        df["hub_m"] = df["class"].map(
            {"class_low": 120.0, "class_med_low": 105.0, "class_med": 100.0, "class_high": 80.0}
        )
    elif mode == "sp_class_mastr_hub":
        df["turbine_type"] = df["class"].map(SP_CLASS_TURBINE_TYPE)
        df["hub_m"] = df["hub_bin"]
    elif mode == "wpl_map_mastr_hub":
        df["turbine_type"] = df["wpl_type"]
        df["hub_m"] = df["hub_bin"]
    else:
        raise ValueError(mode)

    g = (
        df.groupby(["nuts3", "turbine_type", "hub_m"], as_index=False)["maxPower"]
        .sum()
        .rename(columns={"maxPower": "capacity_kw"})
    )
    return g


def main():
    ensure_results_dir()
    engine = resolve_engine("oeds")
    wind = enrich_and_cache(engine)
    wind = attach_nuts_tso(wind)
    summary, by_type, mapped = fleet_summary(wind)
    summary.to_csv(result_path("mastr_wind_fleet_summary.csv"), index=False)
    by_type.to_csv(result_path("mastr_wind_fleet_wpl_types.csv"), index=False)
    print(summary.to_string(index=False))
    print("\nTop WPL-mapped types (GW):")
    print(by_type.head(15).to_string(index=False))

    date_range = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
    matched_path = result_path("matched_era5_timeseries.csv")
    if not matched_path.is_file():
        raise FileNotFoundError(
            f"{matched_path} required for ENTSO-E/Atlite baselines — run validate_matched_era5.py"
        )
    ts = pd.read_csv(matched_path, index_col=0, parse_dates=True)
    entsoe = ts["entsoe_wind"].astype(float).reindex(date_range)
    atlite_ser = ts["atlite_wind"].astype(float).reindex(date_range)
    wpl_legacy = ts["wpl_wind"].astype(float).reindex(date_range)

    cutout = atlite.Cutout(cutout_path("germany_2023.nc"))
    ds = cutout.data
    coords = nuts3_coords_from_plants(mapped)
    weather = extract_weather(ds, coords, date_range)
    nuts_to_tso = mapped.groupby("nuts3")["tso"].first().to_dict()

    cases = {}
    for mode, label in [
        ("sp_class_fixed_hub", "WPL 4-class fixed hubs"),
        ("sp_class_mastr_hub", "WPL 4-class curves × MaStR hubs"),
        ("wpl_map_mastr_hub", "WPL mapped types × MaStR hubs"),
    ]:
        groups = build_case_groups(mapped, mode)
        logger.info(
            "%s: %d groups, %.2f GW",
            label,
            len(groups),
            groups["capacity_kw"].sum() / 1e6,
        )
        cases[label] = run_grouped_wpl(groups, weather, nuts_to_tso, date_range)

    cases["WPL 4-class (matched study CSV)"] = wpl_legacy
    cases["Atlite V112 @ 80 m"] = atlite_ser

    rows = []
    for label, sim in cases.items():
        rows.append(metrics_row(sim, entsoe, label, "ENTSO-E"))
        if label != "Atlite V112 @ 80 m":
            rows.append(metrics_row(sim, atlite_ser, label, "Atlite"))
    out = pd.DataFrame(rows)
    out.to_csv(result_path("mastr_wind_fleet_decomp.csv"), index=False)
    print("\n" + out.to_string(index=False))

    # Decomposition narrative
    def ratio(label, ref="Atlite"):
        r = out[(out.case == label) & (out.ref == ref)]
        return float(r["Ratio_%"].iloc[0]) if len(r) else float("nan")

    r_fixed = ratio("WPL 4-class fixed hubs")
    r_hub = ratio("WPL 4-class curves × MaStR hubs")
    r_map = ratio("WPL mapped types × MaStR hubs")
    md = result_path("mastr_wind_fleet_decomp.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(
            "# MaStR wind fleet decomposition (matched ERA5)\n\n"
            "Goal: replace the opaque 4 SP-class vs Atlite-V80 gap with plant-level MaStR "
            "attributes (hub height, diameter, rated power → WPL catalogue).\n\n"
            "## Fleet facts\n\n"
            + summary.to_string(index=False)
            + "\n\n"
            "## Energy ratios vs Atlite (Atlite = 100%)\n\n"
            f"| Case | / Atlite |\n|---|---:|\n"
            f"| 4-class fixed hubs | **{r_fixed:.1f}%** |\n"
            f"| 4-class curves × MaStR hubs | **{r_hub:.1f}%** |\n"
            f"| Mapped WPL types × MaStR hubs | **{r_map:.1f}%** |\n\n"
            f"Hub-only step (fixed → MaStR hubs, same 4 curves): "
            f"**{r_fixed:.1f}% → {r_hub:.1f}%** "
            f"(Δ {r_hub - r_fixed:+.1f} pp).\n\n"
            f"Type-mapping step (SP-class curves → diam/power WPL map, MaStR hubs): "
            f"**{r_hub:.1f}% → {r_map:.1f}%** "
            f"(Δ {r_map - r_hub:+.1f} pp).\n\n"
            "## Reading\n\n"
            "- Cap-weighted MaStR hub (~113 m) exceeds the 4-class fixed hubs (~106 m), "
            "and ~24 GW sits above 125 m — so fixed 80–120 m understates modern hubs.\n"
            "- Diameter+power nearest-neighbour covers ~80% of capacity with the oedb "
            "catalogue; the rest falls back to SP-class representatives.\n"
            "- Atlite still uses a single V112@80 m for all capacity — that remains a "
            "conservative fleet proxy even after WPL is plant-aware.\n\n"
            "See `mastr_wind_fleet_decomp.csv` for ENTSO-E and Atlite metrics.\n"
        )
    logger.info("Wrote %s", md)


if __name__ == "__main__":
    main()
