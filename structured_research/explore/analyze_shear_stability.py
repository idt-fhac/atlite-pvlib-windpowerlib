"""
Inspect 10→100 m shear: neutral log vs temperature/stability proxies vs ERA5.

Facts about current OEDS stack:
  - windpowerlib logarithmic_profile uses only u10, z0, heights (neutral).
  - ModelChain density_correction defaults to False → temp_air does not
    affect power; it is unused for shear and (by default) unused for density.

This script:
  1) Samples capacity-weighted NUTS3 cells
  2) Compares ECMWF u10 → log/Hellman 100 m vs ERA5 wnd100m + wnd_shear_exp
  3) Relates implied α to T, dT/dt, GHI, hour
  4) Tests improved extrapolators and optional national power impact

Outputs → structured_research/results/:
  shear_stability_pairs.csv
  shear_stability_bins.csv
  shear_stability_models.csv
  shear_stability_summary.md
  shear_stability_alpha_vs_hour.png
  shear_stability_era5_vs_log.png
"""

from __future__ import annotations

from pathlib import Path

import sys
import logging

import atlite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from windpowerlib import ModelChain, WindTurbine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    resolve_engine,
    offline_mode,
    load_plz_nuts,
    map_row_to_tso,
    classify_wind_turbines,
    query_mastr_wind,
    query_entsoe_generation,
    query_ecmwf_weather_nuts3,
    calculate_metrics,
    ensure_results_dir,
    result_path,
    cutout_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("shear_stability")

ZONES = ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET"]
HUB = 100.0
Z0 = 0.2
DAY_HOURS = list(range(6, 18))

TURBINE_MODELS = {
    "class_low": WindTurbine(hub_height=120.0, turbine_type="V112/3000"),
    "class_med_low": WindTurbine(hub_height=105.0, turbine_type="V90/2000"),
    "class_med": WindTurbine(hub_height=100.0, turbine_type="E-82/2300"),
    "class_high": WindTurbine(hub_height=80.0, turbine_type="E-70/2000"),
}


def log_u(u10, z0=Z0, hub=HUB):
    return u10 * np.log(hub / z0) / np.log(10.0 / z0)


def hellman_u(u10, alpha, hub=HUB):
    return u10 * (hub / 10.0) ** alpha


def implied_alpha(u10, u100):
    """α such that u100 = u10 * (10)^α."""
    ratio = np.asarray(u100, dtype=float) / np.asarray(u10, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log(ratio) / np.log(HUB / 10.0)
    a = np.where((u10 > 0.5) & (u100 > 0.5) & np.isfinite(a), a, np.nan)
    return a


def solar_radiation_alpha(ghi_wm2: np.ndarray, u10: np.ndarray, is_night: np.ndarray) -> np.ndarray:
    """
    Simple Pasquill–Gifford / solar-radiation style α lookup.

    Day: stronger insolation + light winds → unstable → smaller α.
    Night: clear (low residual GHI) + light winds → stable → larger α.
    Rough onshore defaults around 0.14–0.40.
    """
    alpha = np.full(len(u10), 0.20, dtype=float)
    # Night
    n = is_night
    light = u10 < 2.0
    mod = (u10 >= 2.0) & (u10 < 5.0)
    strong = u10 >= 5.0
    alpha[n & light] = 0.40
    alpha[n & mod] = 0.30
    alpha[n & strong] = 0.20
    # Day by GHI (W/m2); our ghi is J/m2 per hour → /3600
    d = ~is_night
    alpha[d & (ghi_wm2 >= 500) & (u10 < 3)] = 0.10  # unstable
    alpha[d & (ghi_wm2 >= 500) & (u10 >= 3)] = 0.14
    alpha[d & (ghi_wm2 >= 200) & (ghi_wm2 < 500)] = 0.16
    alpha[d & (ghi_wm2 < 200) & (u10 < 3)] = 0.22
    alpha[d & (ghi_wm2 < 200) & (u10 >= 3)] = 0.18
    return alpha


def temp_tendency_alpha(temp_k: np.ndarray, u10: np.ndarray, is_night: np.ndarray) -> np.ndarray:
    """Increase α when air is cooling (stable) and winds are light."""
    dT = np.empty_like(temp_k, dtype=float)
    dT[0] = 0.0
    dT[1:] = temp_k[1:] - temp_k[:-1]
    alpha = np.full(len(temp_k), 0.20, dtype=float)
    # cooling → more stable
    cool = dT < -0.3
    warm = dT > 0.3
    alpha[is_night & cool & (u10 < 4)] = 0.38
    alpha[is_night & cool & (u10 >= 4)] = 0.28
    alpha[is_night & ~cool & (u10 < 4)] = 0.30
    alpha[is_night & ~cool & (u10 >= 4)] = 0.22
    alpha[~is_night & warm] = 0.14
    alpha[~is_night & ~warm & (u10 < 3)] = 0.18
    alpha[~is_night & ~warm & (u10 >= 3)] = 0.16
    return alpha


def collect_pairs(wind_groups, weather_dict, nuts3_coords, date_range, top_frac=0.50, max_nuts=40):
    cap = wind_groups.sum(axis=1).sort_values(ascending=False)
    cum = cap.cumsum() / cap.sum()
    keep = list(cap.index[cum <= top_frac])
    if len(keep) < 15:
        keep = list(cap.head(25).index)
    keep = keep[:max_nuts]

    cutout = atlite.Cutout(cutout_path("germany_2023.nc"))
    ds = cutout.data

    frames = []
    for nuts in tqdm(keep, desc="Pair ECMWF/ERA5"):
        if nuts not in weather_dict or nuts not in nuts3_coords.index:
            continue
        w = weather_dict[nuts].reindex(date_range, method="nearest")
        lat = float(nuts3_coords.loc[nuts, "latitude"])
        lon = float(nuts3_coords.loc[nuts, "longitude"])
        yi = int(np.abs(ds.y.values - lat).argmin())
        xi = int(np.abs(ds.x.values - lon).argmin())
        era_u100 = ds["wnd100m"].isel(y=yi, x=xi).to_pandas().reindex(date_range, method="nearest")
        era_alpha = ds["wnd_shear_exp"].isel(y=yi, x=xi).to_pandas().reindex(date_range, method="nearest")
        era_t = ds["temperature"].isel(y=yi, x=xi).to_pandas().reindex(date_range, method="nearest")

        u10 = w["wind_speed"].astype(float)
        temp = w["temp_air"].astype(float)
        ghi = (w["ghi"].astype(float) / 3600.0).clip(lower=0.0)  # W/m2 approx
        u100_log = log_u(u10.values)
        alpha_imp = implied_alpha(u10.values, era_u100.values)
        is_night = ~np.isin(date_range.hour, DAY_HOURS)

        df = pd.DataFrame({
            "nuts3": nuts,
            "capacity_kW": float(cap.loc[nuts]),
            "u10_ecmwf": u10.values,
            "temp_ecmwf": temp.values,
            "ghi_wm2": ghi.values,
            "u100_log_z0.2": u100_log,
            "u100_era5": era_u100.values,
            "alpha_era5": era_alpha.values,
            "temp_era5": era_t.values,
            "alpha_implied": alpha_imp,
            "is_night": is_night,
            "hour": date_range.hour,
        }, index=date_range)
        df["u100_bias_log"] = df["u100_log_z0.2"] - df["u100_era5"]
        df["u100_ratio_log"] = df["u100_log_z0.2"] / df["u100_era5"]
        frames.append(df)

    return pd.concat(frames)


def bin_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    p = pairs.dropna(subset=["alpha_implied", "u10_ecmwf", "u100_era5"])
    p = p[(p.u10_ecmwf > 0.5) & (p.u100_era5 > 0.5)]

    def add(label, sub):
        if len(sub) < 50:
            return
        rows.append({
            "bin": label,
            "n": len(sub),
            "alpha_implied_mean": sub.alpha_implied.mean(),
            "alpha_era5_mean": sub.alpha_era5.mean(),
            "u100_log_mean": sub["u100_log_z0.2"].mean(),
            "u100_era5_mean": sub.u100_era5.mean(),
            "log_bias_ms": (sub["u100_log_z0.2"] - sub.u100_era5).mean(),
            "log_ratio": sub["u100_log_z0.2"].mean() / sub.u100_era5.mean(),
            "u10_mean": sub.u10_ecmwf.mean(),
            "temp_mean": sub.temp_ecmwf.mean(),
        })

    add("all", p)
    add("day", p[~p.is_night])
    add("night", p[p.is_night])
    for h in range(0, 24, 3):
        add(f"hour_{h:02d}-{h+2:02d}", p[p.hour.isin(range(h, h + 3))])

    # temperature terciles at night
    night = p[p.is_night]
    if len(night):
        q = night.temp_ecmwf.quantile([0.33, 0.67])
        add("night_cold", night[night.temp_ecmwf <= q.iloc[0]])
        add("night_mild", night[(night.temp_ecmwf > q.iloc[0]) & (night.temp_ecmwf <= q.iloc[1])])
        add("night_warm", night[night.temp_ecmwf > q.iloc[1]])

    # dT/dt at night
    night = night.copy()
    night["dT"] = night.groupby("nuts3")["temp_ecmwf"].diff()
    add("night_cooling_dT<-0.3", night[night.dT < -0.3])
    add("night_neutral_|dT|<0.3", night[night.dT.abs() <= 0.3])
    add("night_warming_dT>0.3", night[night.dT > 0.3])

    # light vs strong wind at night
    add("night_u10<3", night[night.u10_ecmwf < 3])
    add("night_u10>=5", night[night.u10_ecmwf >= 5])

    # day GHI
    day = p[~p.is_night]
    add("day_ghi>=400", day[day.ghi_wm2 >= 400])
    add("day_ghi<150", day[day.ghi_wm2 < 150])

    return pd.DataFrame(rows)


def evaluate_extrapolators(pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.dropna(subset=["u10_ecmwf", "u100_era5"]).copy()
    p = p[(p.u10_ecmwf > 0.5) & (p.u100_era5 > 0.5)]
    is_night = p.is_night.values
    u10 = p.u10_ecmwf.values
    era = p.u100_era5.values
    ghi = p.ghi_wm2.values
    temp = p.temp_ecmwf.values

    # learned dual alpha from implied means
    a_day = np.nanmean(p.loc[~p.is_night, "alpha_implied"])
    a_night = np.nanmean(p.loc[p.is_night, "alpha_implied"])

    models = {
        "neutral_log_z0.2": log_u(u10),
        "hellman_alpha_1/7": hellman_u(u10, 1 / 7),
        "hellman_alpha_era5_field": hellman_u(u10, np.clip(p.alpha_era5.values, 0.05, 0.6)),
        "hellman_dual_learned": hellman_u(u10, np.where(is_night, a_night, a_day)),
        "hellman_solar_rad_class": hellman_u(u10, solar_radiation_alpha(ghi, u10, is_night)),
        "hellman_temp_tendency": hellman_u(u10, temp_tendency_alpha(temp, u10, is_night)),
    }

    rows = []
    for name, uhat in models.items():
        bias = uhat - era
        for which, mask in [("all", np.ones(len(era), dtype=bool)),
                            ("day", ~is_night),
                            ("night", is_night)]:
            m = mask & np.isfinite(uhat) & np.isfinite(era)
            if m.sum() < 50:
                continue
            rows.append({
                "model": name,
                "subset": which,
                "rmse": float(np.sqrt(np.mean((uhat[m] - era[m]) ** 2))),
                "mae": float(np.mean(np.abs(uhat[m] - era[m]))),
                "bias": float(np.mean(uhat[m] - era[m])),
                "ratio_mean": float(np.mean(uhat[m]) / np.mean(era[m])),
                "corr": float(np.corrcoef(uhat[m], era[m])[0, 1]),
                "alpha_day_used": a_day,
                "alpha_night_used": a_night,
            })
    return pd.DataFrame(rows)


def run_power_with_alpha_series(wind_groups, weather_dict, nuts3_to_tso, date_range, alpha_fn, act_w_tso):
    """National day/night power ratios for a given alpha_fn(u10,temp,ghi,is_night)->alpha array."""
    sim = {t: pd.Series(0.0, index=date_range) for t in ZONES}
    is_night = ~np.isin(date_range.hour, DAY_HOURS)
    for nuts in wind_groups.index:
        if nuts not in weather_dict or nuts not in nuts3_to_tso:
            continue
        tso = nuts3_to_tso[nuts]
        w = weather_dict[nuts].reindex(date_range, method="nearest")
        u10 = w["wind_speed"].values.astype(float)
        temp = w["temp_air"].values.astype(float)
        ghi = (w["ghi"].values.astype(float) / 3600.0).clip(min=0)
        alpha = alpha_fn(u10, temp, ghi, is_night)
        row = wind_groups.loc[nuts]
        for cls_name, capacity_kw in row.items():
            if capacity_kw <= 0:
                continue
            wt = TURBINE_MODELS[cls_name]
            u_hub = hellman_u(u10, alpha, wt.hub_height)
            ww = pd.DataFrame(
                np.asarray([temp, u_hub]).T,
                index=date_range,
                columns=[["temperature", "wind_speed"], [2, wt.hub_height]],
            )
            mc = ModelChain(wt, wind_speed_model="interpolation_extrapolation").run_model(ww)
            sim[tso] += mc.power_output / wt.nominal_power * (capacity_kw * 1e3) / 1e6

    sim_nat = sum((sim[t] for t in ZONES), start=pd.Series(0.0, index=date_range))
    act_nat = sum((act_w_tso[t].reindex(date_range, fill_value=0.0) for t in ZONES),
                  start=pd.Series(0.0, index=date_range))
    out = {}
    for which, mask in [("Day", ~is_night), ("Night", is_night), ("Full", np.ones(len(date_range), bool))]:
        m = calculate_metrics(sim_nat[mask], act_nat[mask])
        out[f"{which}_ratio"] = m["ratio"]
        out[f"{which}_corr"] = m["corr"]
    return out


def plots(pairs, models, out_dir):
    p = pairs.dropna(subset=["alpha_implied"])
    p = p[(p.u10_ecmwf > 0.5) & (p.u100_era5 > 0.5)]

    # alpha vs hour
    fig, ax = plt.subplots(figsize=(8, 4))
    hod = p.groupby("hour").agg(
        a_imp=("alpha_implied", "mean"),
        a_era=("alpha_era5", "mean"),
    )
    ax.plot(hod.index, hod.a_imp, "o-", label="Implied α from ERA5 100m / ECMWF 10m")
    ax.plot(hod.index, hod.a_era, "s-", label="ERA5 wnd_shear_exp field")
    ax.axhline(1 / np.log(HUB / Z0), color="gray", ls="--", label=f"Neutral-log equiv. α (z0={Z0})")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Shear exponent α")
    ax.set_title("Diurnal shear: implied α vs ERA5 field vs neutral log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig((out_dir / "shear_stability_alpha_vs_hour.png"), dpi=300)
    plt.close(fig)

    # era5 vs log scatter by day/night (subsample)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, night, title in zip(axes, [False, True], ["Day", "Night"]):
        sub = p[p.is_night == night].sample(min(8000, (p.is_night == night).sum()), random_state=0)
        ax.scatter(sub.u100_era5, sub["u100_log_z0.2"], s=2, alpha=0.15, c="C0")
        lim = (0, max(sub.u100_era5.quantile(0.99), sub["u100_log_z0.2"].quantile(0.99)))
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{title}: neutral log vs ERA5 100 m")
        ax.set_xlabel("ERA5 wnd100m (m/s)")
        ax.set_ylabel("ECMWF 10m → log z0=0.2 (m/s)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    fig.tight_layout()
    fig.savefig((out_dir / "shear_stability_era5_vs_log.png"), dpi=300)
    plt.close(fig)


def write_summary(bins, models, power_rows, path, a_day, a_night):
    m = models
    lines = [
        "# Shear / stability inspection (10 m → 100 m)",
        "",
        "## Does temperature enter the current OEDS wind stack?",
        "- **Shear: no.** `windpowerlib.logarithmic_profile` is neutral: only `u`, heights, `z0`.",
        "- **Density: not in our runs.** `ModelChain(..., density_correction=False)` (default) "
        "with `power_curve` → temperature is carried in the weather frame but **does not change power**.",
        "- So the night collapse cannot be blamed on missing temperature *in the power curve*; "
        "it is missing **stability-dependent shear** (and/or hub-height weather).",
        "",
        f"## Implied α (ERA5 100 m / ECMWF 10 m)",
        f"- Day mean implied α ≈ **{a_day:.3f}**",
        f"- Night mean implied α ≈ **{a_night:.3f}**",
        f"- Neutral-log equivalent α for z0=0.2 at 100 m: "
        f"**{1/np.log(HUB/Z0):.3f}** (constant — no day/night)",
        "",
        "## Bin diagnostics (log bias vs ERA5)",
        bins.to_string(index=False),
        "",
        "## Extrapolator skill vs ERA5 100 m (same ECMWF u10)",
        m.to_string(index=False),
        "",
        "## National power ratios when feeding Hellman hub winds into Windpowerlib",
        pd.DataFrame(power_rows).to_string(index=False),
        "",
        "## Practical improvement path",
        "1. Best physics with current fields: use a **diurnal / stability-dependent α** "
        "(solar-radiation class or dual day/night α fitted to ERA5).",
        "2. Better: ingest **hub-height wind** (ERA5 100 m or ECMWF 100/120 m if available in DB).",
        "3. Temperature alone (density) will not fix night yield; use T / GHI as **stability proxies for α**.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", path)


def main():
    ensure_results_dir()
    out = ensure_results_dir()
    date_range = pd.date_range("2023-01-01", "2023-12-31 23:00:00", freq="h")
    engine = resolve_engine()
    logger.info("Mode: %s", "OFFLINE" if offline_mode() or engine is None else "DB-allowed")

    wind_df = query_mastr_wind(engine, "2023-01-01", "2023-12-31")
    plz = load_plz_nuts(engine)
    wind_df["nuts3"] = wind_df["plzCode"].map(plz["nuts3"])
    wind_df = wind_df.dropna(subset=["nuts3"])
    wind_df["tso"] = wind_df.apply(map_row_to_tso, axis=1)
    wind_df = wind_df[wind_df["tso"] != "UNKNOWN"]
    wind_df = classify_wind_turbines(wind_df)
    wind_groups = wind_df.groupby(["nuts3", "class"])["maxPower"].sum().unstack(fill_value=0.0)
    nuts3_to_tso = {n: g["tso"].iloc[0] for n, g in wind_df.groupby("nuts3")}
    nuts3_coords = plz.groupby("nuts3")[["latitude", "longitude"]].mean()

    weather_raw = query_ecmwf_weather_nuts3(
        engine, "2023-01-01", "2023-12-31", nuts_prefix="DE", date_range=date_range
    )
    weather_dict = {n: g.set_index("time") for n, g in weather_raw.groupby("nuts_id")}
    entsoe_w, _ = query_entsoe_generation(engine, "2023-01-01", "2023-12-31", ZONES, date_range)
    act_w_tso = {z: entsoe_w[z] for z in ZONES}

    logger.info("Collecting ECMWF/ERA5 pairs...")
    pairs = collect_pairs(wind_groups, weather_dict, nuts3_coords, date_range)
    pairs.to_csv(result_path("shear_stability_pairs.csv"))

    bins = bin_table(pairs)
    bins.to_csv(result_path("shear_stability_bins.csv"), index=False)
    print(bins.to_string(index=False))

    models = evaluate_extrapolators(pairs)
    models.to_csv(result_path("shear_stability_models.csv"), index=False)
    print(models.to_string(index=False))

    a_day = float(np.nanmean(pairs.loc[~pairs.is_night, "alpha_implied"]))
    a_night = float(np.nanmean(pairs.loc[pairs.is_night, "alpha_implied"]))
    logger.info("Learned dual α: day=%.3f night=%.3f", a_day, a_night)

    plots(pairs, models, out)

    logger.info("Running national power tests for improved α models...")
    power_rows = []
    # baseline neutral ~ hellman 0.20 was already studied; include dual + solar + temp
    tests = [
        ("hellman_dual_learned", lambda u, t, g, n: np.where(n, a_night, a_day)),
        ("hellman_solar_rad_class", lambda u, t, g, n: solar_radiation_alpha(g, u, n)),
        ("hellman_temp_tendency", lambda u, t, g, n: temp_tendency_alpha(t, u, n)),
        ("hellman_fixed_0.20", lambda u, t, g, n: np.full(len(u), 0.20)),
    ]
    for name, fn in tests:
        logger.info("Power test: %s", name)
        stats = run_power_with_alpha_series(
            wind_groups, weather_dict, nuts3_to_tso, date_range, fn, act_w_tso
        )
        power_rows.append({"model": name, **stats})
        logger.info("  %s", stats)

    pd.DataFrame(power_rows).to_csv(result_path("shear_stability_power_impact.csv"), index=False)
    write_summary(bins, models, power_rows, result_path("shear_stability_summary.md"), a_day, a_night)
    logger.info("Done.")


if __name__ == "__main__":
    main()
