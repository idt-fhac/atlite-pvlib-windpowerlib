"""
Build day/night wind metrics and six sample-week comparison plots.

Outputs (structured_research/results/):
  diurnal_wind_daynight.csv
  week_single_wind.png      — Kelmarsh: actual / Windpowerlib / Atlite
  week_single_solar.png     — Jülich: measured / PVLib+SAPM / Atlite
  week_tso_wind.png         — TenneT: ENTSO-E / OEDS / Atlite
  week_tso_solar.png        — TenneT: ENTSO-E / OEDS / Atlite
  week_national_wind.png    — DE: ENTSO-E / OEDS / Atlite
  week_national_solar.png   — DE: ENTSO-E / OEDS / Atlite

Also copies PNGs to latex/data/ for optional manuscript inclusion later.
"""

from __future__ import annotations

from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import calculate_metrics, ensure_results_dir, result_path
from lib.figures import save_vector_figure
from lib.manuscript import copy_to_manuscript

DAY_HOURS = range(6, 18)
NIGHT_HOURS = list(range(18, 24)) + list(range(0, 6))

# Representative weeks with clear signal
WEEK_WIND = ("2023-01-15", "2023-01-21")   # synoptic winter wind week
WEEK_SOLAR = ("2023-07-10", "2023-07-16")  # summer PV week
TSO_FOCUS = "DE_TENNET"
TSO_LABEL = "TenneT"
TSO_ORDER = ("DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET")
TSO_LABELS = {
    "DE_50HZ": "50Hertz",
    "DE_AMPRION": "Amprion",
    "DE_TENNET": "TenneT",
    "DE_TRANSNET": "TransnetBW",
}

COLORS = {
    "actual": "#222222",
    "oeds": "#1f77b4",
    "atlite": "#d62728",
}


def _day_night_mask(index: pd.DatetimeIndex, which: str) -> np.ndarray:
    hours = index.hour
    if which == "Day":
        return np.isin(hours, list(DAY_HOURS))
    if which == "Night":
        return np.isin(hours, NIGHT_HOURS)
    raise ValueError(which)


def _subset_metrics(sim: pd.Series, act: pd.Series, which: str) -> dict:
    mask = _day_night_mask(act.index, which)
    return calculate_metrics(sim[mask], act[mask])


def build_diurnal_table() -> pd.DataFrame:
    """Day/night wind metrics for the paper path (matched ERA5 + Kelmarsh SCADA)."""
    rows = []

    # Kelmarsh (point wind SCADA check)
    k = pd.read_csv(result_path("kelmarsh_farm_comparison.csv"), index_col=0, parse_dates=True)
    for subset in ("Day", "Night"):
        mw = _subset_metrics(k["windpowerlib"], k["actual"], subset)
        ma = _subset_metrics(k["atlite"], k["actual"], subset)
        rows.append({
            "Scale": "Kelmarsh (farm)",
            "Subset": subset,
            "Observed_GWh": round(mw["act_sum"] / 1e6, 2),
            "WPL_Ratio_%": round(mw["ratio"], 1),
            "Atlite_Ratio_%": round(ma["ratio"], 1),
            "WPL_Corr": round(mw["corr"], 4),
            "Atlite_Corr": round(ma["corr"], 4),
            "Stack": "SCADA / ERA5 100m Hellman",
        })

    # Prefer matched ERA5 national timeseries (paper path)
    matched = Path(result_path("matched_era5_timeseries.csv"))
    if matched.exists():
        n = pd.read_csv(matched, index_col=0, parse_dates=True)
        wpl_col, atl_col = "wpl_wind", "atlite_wind"
        stack = "matched ERA5 cutout"
    else:
        n = pd.read_csv(result_path("annual_seasonal_timeseries.csv"), index_col=0, parse_dates=True)
        wpl_col, atl_col = "oeds_wind", "atlite_wind"
        stack = "legacy OEDS/ECMWF (fallback)"

    for subset in ("Day", "Night"):
        mw = _subset_metrics(n[wpl_col], n["entsoe_wind"], subset)
        ma = _subset_metrics(n[atl_col], n["entsoe_wind"], subset)
        rows.append({
            "Scale": "Germany national",
            "Subset": subset,
            "Observed_GWh": round(mw["act_sum"] / 1e3, 1),
            "WPL_Ratio_%": round(mw["ratio"], 1),
            "Atlite_Ratio_%": round(ma["ratio"], 1),
            "WPL_Corr": round(mw["corr"], 4),
            "Atlite_Corr": round(ma["corr"], 4),
            "Stack": stack,
        })

    return pd.DataFrame(rows)


def _style_week_ax(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.tick_params(axis="x", labelrotation=25)


def plot_week(series_map: dict, start: str, end: str, title: str, ylabel: str, save_path: str):
    """series_map: label -> Series with DatetimeIndex."""
    fig, ax = plt.subplots(figsize=(12, 4.2))
    style = {
        list(series_map.keys())[0]: (COLORS["actual"], "-", 1.8),
        list(series_map.keys())[1]: (COLORS["oeds"], "--", 1.4),
        list(series_map.keys())[2]: (COLORS["atlite"], "-.", 1.4),
    }
    for label, ser in series_map.items():
        ser = ser.loc[start:end]
        color, ls, lw = style[label]
        ax.plot(ser.index, ser.values, label=label, color=color, linestyle=ls, linewidth=lw, alpha=0.9)
    _style_week_ax(ax, title, ylabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def build_week_plots():
    out = ensure_results_dir()

    # 1) Single wind — Kelmarsh
    k = pd.read_csv(result_path("kelmarsh_farm_comparison.csv"), index_col=0, parse_dates=True)
    plot_week(
        {
            "SCADA (actual)": k["actual"] / 1e3,  # kW -> MW
            "Windpowerlib (ERA5 100m)": k["windpowerlib"] / 1e3,
            "Atlite (ERA5)": k["atlite"] / 1e3,
        },
        *WEEK_WIND,
        title=f"Kelmarsh wind farm — sample week {WEEK_WIND[0]} to {WEEK_WIND[1]}",
        ylabel="Power (MW)",
        save_path=(out / "week_single_wind.png"),
    )

    # 2) Single solar — Jülich (PVLib temp-corrected on matched ERA5)
    j = pd.read_csv(result_path("juelich_eview_comparison_complete.csv"), index_col=0, parse_dates=True)
    plot_week(
        {
            "Measured AC": j["actual_generation"],
            "PVLib+SAPM (ERA5)": j["oeds_temp_corrected"],
            "Atlite (ERA5)": j["atlite_sim"],
        },
        *WEEK_SOLAR,
        title=f"Campus Jülich PV — sample week {WEEK_SOLAR[0]} to {WEEK_SOLAR[1]}",
        ylabel="Power (kW)",
        save_path=(out / "week_single_solar.png"),
    )

    # 3–4) National (matched ERA5 paper path)
    matched = Path(result_path("matched_era5_timeseries.csv"))
    if not matched.exists():
        raise FileNotFoundError(
            f"Missing {matched} — run validate_matched_era5 before plot_weeks."
        )
    n = pd.read_csv(matched, index_col=0, parse_dates=True)
    plot_week(
        {
            "ENTSO-E feed-in": n["entsoe_wind"],
            "Windpowerlib MaStR (ERA5)": n["wpl_wind"],
            "Atlite (ERA5)": n["atlite_wind"],
        },
        *WEEK_WIND,
        title=f"Germany national wind — sample week {WEEK_WIND[0]} to {WEEK_WIND[1]}",
        ylabel="Power (MW)",
        save_path=(out / "week_national_wind.png"),
    )
    plot_week(
        {
            "ENTSO-E feed-in": n["entsoe_solar"],
            "PVLib MaStR (ERA5)": n["pvlib_mastr_solar"],
            "Atlite (ERA5)": n["atlite_solar"],
        },
        *WEEK_SOLAR,
        title=f"Germany national solar — sample week {WEEK_SOLAR[0]} to {WEEK_SOLAR[1]}",
        ylabel="Power (MW)",
        save_path=(out / "week_national_solar.png"),
    )

    # 5–6) TSO (TenneT) on matched ERA5 + 4-panel all-TSO weeks for the paper
    tso_path = Path(result_path("matched_era5_tso_timeseries.parquet"))
    sample_path = Path(result_path("matched_era5_tso_sample_weeks.csv"))
    tso = None
    if tso_path.exists():
        tso = pd.read_parquet(tso_path)
        if not isinstance(tso.index, pd.DatetimeIndex):
            tso.index = pd.to_datetime(tso.index)
        # Cache the two paper sample weeks so export can rebuild without the full parquet
        sample = pd.concat(
            [tso.loc[WEEK_WIND[0] : WEEK_WIND[1]], tso.loc[WEEK_SOLAR[0] : WEEK_SOLAR[1]]]
        )
        sample = sample[~sample.index.duplicated(keep="first")].sort_index()
        sample.to_csv(sample_path)
        copy_to_manuscript(sample_path)
        print(f"Saved {sample_path}")
    elif sample_path.exists():
        tso = pd.read_csv(sample_path, index_col=0, parse_dates=True)
        print(f"Using cached TSO sample weeks → {sample_path}")
    else:
        print(
            f"WARNING: skipping TSO week plots — missing {tso_path} "
            f"and {sample_path}"
        )
        return

    z = TSO_FOCUS
    plot_week(
        {
            "ENTSO-E feed-in": tso[f"entsoe_wind_{z}"],
            "Windpowerlib MaStR (ERA5)": tso[f"wpl_wind_{z}"],
            "Atlite (ERA5)": tso[f"atlite_wind_{z}"],
        },
        *WEEK_WIND,
        title=f"{TSO_LABEL} wind — sample week {WEEK_WIND[0]} to {WEEK_WIND[1]}",
        ylabel="Power (MW)",
        save_path=(out / "week_tso_wind.png"),
    )
    plot_week(
        {
            "ENTSO-E feed-in": tso[f"entsoe_solar_{z}"],
            "PVLib MaStR (ERA5)": tso[f"pvlib_mastr_solar_{z}"],
            "Atlite (ERA5)": tso[f"atlite_solar_{z}"],
        },
        *WEEK_SOLAR,
        title=f"{TSO_LABEL} solar — sample week {WEEK_SOLAR[0]} to {WEEK_SOLAR[1]}",
        ylabel="Power (MW)",
        save_path=(out / "week_tso_solar.png"),
    )
    plot_tso_four_panel(
        tso,
        tech="wind",
        start=WEEK_WIND[0],
        end=WEEK_WIND[1],
        stem="tso_matched_week_wind",
        ylabel="Power (MW)",
        title=(
            f"German TSO onshore wind — matched ERA5 sample week "
            f"{WEEK_WIND[0]} to {WEEK_WIND[1]}"
        ),
    )
    plot_tso_four_panel(
        tso,
        tech="solar",
        start=WEEK_SOLAR[0],
        end=WEEK_SOLAR[1],
        stem="tso_matched_week_solar",
        ylabel="Power (MW)",
        title=(
            f"German TSO solar — matched ERA5 sample week "
            f"{WEEK_SOLAR[0]} to {WEEK_SOLAR[1]}"
        ),
    )


def plot_tso_four_panel(
    tso: pd.DataFrame,
    *,
    tech: str,
    start: str,
    end: str,
    stem: str,
    ylabel: str,
    title: str,
) -> None:
    """Four-zone chronology for wind or solar on the matched ERA5 cutout."""
    if tech == "wind":
        ent_key, sim_a, sim_b, lab_a, lab_b = (
            "entsoe_wind",
            "wpl_wind",
            "atlite_wind",
            "Windpowerlib MaStR (ERA5)",
            "Atlite (ERA5)",
        )
    elif tech == "solar":
        ent_key, sim_a, sim_b, lab_a, lab_b = (
            "entsoe_solar",
            "pvlib_mastr_solar",
            "atlite_solar",
            "PVLib MaStR (ERA5)",
            "Atlite (ERA5)",
        )
    else:
        raise ValueError(tech)

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 10.5), sharex=True)
    for ax, zone in zip(axes, TSO_ORDER):
        ent = tso[f"{ent_key}_{zone}"].loc[start:end]
        a = tso[f"{sim_a}_{zone}"].loc[start:end]
        b = tso[f"{sim_b}_{zone}"].loc[start:end]
        ax.plot(ent.index, ent.values, color=COLORS["actual"], lw=1.6, label="ENTSO-E feed-in")
        ax.plot(a.index, a.values, color=COLORS["oeds"], ls="--", lw=1.2, label=lab_a)
        ax.plot(b.index, b.values, color=COLORS["atlite"], ls="-.", lw=1.2, label=lab_b)
        ax.set_ylabel(ylabel)
        ax.set_title(TSO_LABELS[zone], loc="left", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("Time")
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout()
    save_vector_figure(fig, stem)
    plt.close(fig)
    print(f"Saved vector figure {stem}")


def sync_to_manuscript_data():
    names = [
        "week_single_wind.png",
        "week_single_solar.png",
        "week_tso_wind.png",
        "week_tso_solar.png",
        "week_national_wind.png",
        "week_national_solar.png",
        "diurnal_wind_daynight.csv",
        "matched_era5_tso_sample_weeks.csv",
        "tso_matched_week_wind.pdf",
        "tso_matched_week_solar.pdf",
        "tso_matched_week_wind.svg",
        "tso_matched_week_solar.svg",
    ]
    for name in names:
        src = Path(result_path(name))
        if src.exists():
            copy_to_manuscript(src)


def main():
    ensure_results_dir()
    diurnal = build_diurnal_table()
    out_csv = result_path("diurnal_wind_daynight.csv")
    diurnal.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    print(diurnal.to_string(index=False))
    build_week_plots()
    sync_to_manuscript_data()


if __name__ == "__main__":
    main()
