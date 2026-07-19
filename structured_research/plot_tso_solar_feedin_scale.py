"""
TSO solar: physics derates + illustrative feed-in scale vs ENTSO-E.

Prefers matched_era5_tso_timeseries.parquet (paper path). Falls back to
annual_seasonal_tso_timeseries.parquet only if the matched file is missing.

Outputs → results/ (+ copies PNGs to text/data/):
  tso_solar_feedin_scale_metrics.csv
  tso_solar_two_weeks.png
  tso_solar_hourly_duration.png
  tso_solar_daily_max_duration.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import FEED_IN_SCALE, FLEET_AGING_FALLBACK, INVERTER_EFFICIENCY, ZONES, ZONE_LABEL, copy_to_manuscript
from utils import calculate_metrics, ensure_results_dir, result_path


INV = INVERTER_EFFICIENCY
ETA = FLEET_AGING_FALLBACK
FEED = FEED_IN_SCALE
# Mid-June window (clear summer generation)
WEEK_START, WEEK_END = "2023-06-12", "2023-06-25"



def load_tso_series() -> dict[str, pd.DataFrame]:
    """Return per-TSO DataFrame with entsoe, pvlib_phys, atlite_phys, *_feed."""
    matched_path = result_path("matched_era5_tso_timeseries.parquet")
    legacy_path = result_path("annual_seasonal_tso_timeseries.parquet")
    if matched_path.exists():
        tso_ts = pd.read_parquet(matched_path)
        use_matched_hourly = True
    elif legacy_path.exists():
        tso_ts = pd.read_parquet(legacy_path)
        use_matched_hourly = False
        print(f"WARN: using legacy {legacy_path}; re-run validate_matched_era5.py")
    else:
        raise FileNotFoundError(
            "Need matched_era5_tso_timeseries.parquet (run validate_matched_era5.py)"
        )

    out = {}
    for z in ZONES:
        ent = tso_ts[f"entsoe_solar_{z}"].astype(float)
        atl_raw = tso_ts[f"atlite_solar_{z}"].astype(float)
        if use_matched_hourly and f"pvlib_mastr_solar_{z}" in tso_ts.columns:
            pv_raw = tso_ts[f"pvlib_mastr_solar_{z}"].astype(float)
        else:
            # Legacy bridge: scale Atlite shape to matched annual PVLib/Atlite ratio
            matched = pd.read_csv(result_path("matched_era5_tso.csv"))
            sol = matched[matched.Technology == "Solar"]
            pv_ann = float(
                sol.loc[(sol.Scale == z) & sol.Model.str.contains("MaStR"), "Sim_GWh"].iloc[0]
            )
            atl_ann = float(
                sol.loc[(sol.Scale == z) & sol.Model.str.startswith("Atlite"), "Sim_GWh"].iloc[0]
            )
            pv_raw = atl_raw * (pv_ann * 1e3 / atl_raw.sum())
            atl_raw = atl_raw * (atl_ann * 1e3 / atl_raw.sum())

        pv_phys = pv_raw * INV * ETA
        atl_phys = atl_raw * ETA
        out[z] = pd.DataFrame(
            {
                "entsoe": ent,
                "pvlib_phys": pv_phys,
                "atlite_phys": atl_phys,
                "pvlib_feed": pv_phys * FEED,
                "atlite_feed": atl_phys * FEED,
            }
        )
    return out


def metrics_table(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    matched = pd.read_csv(result_path("matched_era5_tso.csv"))
    sol = matched[matched.Technology == "Solar"]
    # True PVLib MaStR correlations (scale-invariant); hourly PVLib TSO is
    # reconstructed from Atlite shape × matched annual energy.
    pv_corr = {
        r.Scale: float(r.Correlation)
        for _, r in sol[sol.Model.str.contains("MaStR")].iterrows()
    }
    atl_corr = {
        r.Scale: float(r.Correlation)
        for _, r in sol[sol.Model.str.startswith("Atlite")].iterrows()
    }

    rows = []
    for z, df in series.items():
        for label, col in [
            ("PVLib ×η_inv×age", "pvlib_phys"),
            (f"PVLib ×η_inv×age×{FEED}", "pvlib_feed"),
            ("Atlite ×age", "atlite_phys"),
            (f"Atlite ×age×{FEED}", "atlite_feed"),
        ]:
            m = calculate_metrics(df[col], df["entsoe"])
            corr = pv_corr[z] if "PVLib" in label else atl_corr[z]
            rows.append(
                {
                    "TSO": z,
                    "TSO_label": ZONE_LABEL[z],
                    "Configuration": label,
                    "Ratio_%": round(m["ratio"], 1),
                    "Corr": round(corr, 4),
                    "MAE_MW": round(m["mae"], 1),
                    "RMSE_MW": round(m["rmse"], 1),
                    "Sim_GWh": round(m["sim_sum"] / 1e3, 1),
                    "Feedin_GWh": round(m["act_sum"] / 1e3, 1),
                }
            )
    # National = sum of TSOs
    nat = series[ZONES[0]].copy()
    for z in ZONES[1:]:
        nat = nat.add(series[z], fill_value=0.0)
    nat_matched = pd.read_csv(result_path("matched_era5_timeseries.csv"), index_col=0, parse_dates=True)
    nat_pv_corr = float(
        calculate_metrics(
            nat_matched["pvlib_mastr_solar"] * INV * ETA, nat_matched["entsoe_solar"]
        )["corr"]
    )
    nat_atl_corr = float(
        calculate_metrics(
            nat_matched["atlite_solar"] * ETA, nat_matched["entsoe_solar"]
        )["corr"]
    )
    for label, col in [
        ("PVLib ×η_inv×age", "pvlib_phys"),
        (f"PVLib ×η_inv×age×{FEED}", "pvlib_feed"),
        ("Atlite ×age", "atlite_phys"),
        (f"Atlite ×age×{FEED}", "atlite_feed"),
    ]:
        m = calculate_metrics(nat[col], nat["entsoe"])
        corr = nat_pv_corr if "PVLib" in label else nat_atl_corr
        rows.append(
            {
                "TSO": "DE national",
                "TSO_label": "Germany",
                "Configuration": label,
                "Ratio_%": round(m["ratio"], 1),
                "Corr": round(corr, 4),
                "MAE_MW": round(m["mae"], 1),
                "RMSE_MW": round(m["rmse"], 1),
                "Sim_GWh": round(m["sim_sum"] / 1e3, 1),
                "Feedin_GWh": round(m["act_sum"] / 1e3, 1),
            }
        )
    return pd.DataFrame(rows)


def plot_two_weeks(series: dict[str, pd.DataFrame], outfile: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, z in zip(axes.ravel(), ZONES):
        df = series[z].loc[WEEK_START:WEEK_END]
        ax.plot(df.index, df["entsoe"], color="black", lw=1.8, label="ENTSO-E feed-in")
        ax.plot(
            df.index,
            df["pvlib_phys"],
            color="C0",
            lw=1.2,
            ls=":",
            alpha=0.85,
            label="PVLib gen. (η_inv×age)",
        )
        ax.plot(
            df.index,
            df["pvlib_feed"],
            color="C0",
            lw=1.5,
            ls="--",
            label=f"PVLib ×{FEED} (illustrative)",
        )
        ax.plot(
            df.index,
            df["atlite_feed"],
            color="C2",
            lw=1.5,
            ls="-.",
            label=f"Atlite ×age×{FEED} (illustrative)",
        )
        ax.set_title(ZONE_LABEL[z])
        ax.set_ylabel("Power (MW)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle(
        f"TSO solar 2023 — two weeks ({WEEK_START} to {WEEK_END})\n"
        f"Generation physics vs ENTSO-E feed-in; dashed = illustrative ×{FEED} energy match",
        fontsize=11,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_tso_hourly_duration(series: dict[str, pd.DataFrame], outfile: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, z in zip(axes.ravel(), ZONES):
        df = series[z]
        n = len(df)
        x = np.arange(1, n + 1) / n * 100.0
        ax.plot(x, np.sort(df["entsoe"].to_numpy())[::-1], color="black", lw=2.0, label="ENTSO-E")
        ax.plot(
            x,
            np.sort(df["pvlib_phys"].to_numpy())[::-1],
            color="C0",
            lw=1.2,
            ls=":",
            alpha=0.8,
            label="PVLib gen.",
        )
        ax.plot(
            x,
            np.sort(df["pvlib_feed"].to_numpy())[::-1],
            color="C0",
            lw=1.6,
            ls="--",
            label=f"PVLib ×{FEED}",
        )
        ax.plot(
            x,
            np.sort(df["atlite_feed"].to_numpy())[::-1],
            color="C2",
            lw=1.6,
            ls="-.",
            label=f"Atlite ×{FEED}",
        )
        ax.set_title(ZONE_LABEL[z])
        ax.set_xlabel("Percentage of time (%)")
        ax.set_ylabel("Power (MW)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle(
        f"TSO solar 2023 — hourly duration curves (physics gen. vs illustrative ×{FEED})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def plot_tso_daily_max_duration(series: dict[str, pd.DataFrame], outfile: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, z in zip(axes.ravel(), ZONES):
        df = series[z]
        daily = df.groupby(df.index.normalize()).max()
        n = len(daily)
        x = np.arange(1, n + 1) / n * 100.0
        ax.plot(
            x,
            np.sort(daily["entsoe"].to_numpy())[::-1],
            color="black",
            lw=2.0,
            label="ENTSO-E",
        )
        ax.plot(
            x,
            np.sort(daily["pvlib_phys"].to_numpy())[::-1],
            color="C0",
            lw=1.2,
            ls=":",
            alpha=0.8,
            label="PVLib gen.",
        )
        ax.plot(
            x,
            np.sort(daily["pvlib_feed"].to_numpy())[::-1],
            color="C0",
            lw=1.6,
            ls="--",
            label=f"PVLib ×{FEED}",
        )
        ax.plot(
            x,
            np.sort(daily["atlite_feed"].to_numpy())[::-1],
            color="C2",
            lw=1.6,
            ls="-.",
            label=f"Atlite ×{FEED}",
        )
        ax.set_title(ZONE_LABEL[z])
        ax.set_xlabel("Percentage of days (%)")
        ax.set_ylabel("Daily max power (MW)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle(
        f"TSO solar 2023 — daily-max duration curves (physics gen. vs illustrative ×{FEED})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


def update_national_duration_plots(series: dict[str, pd.DataFrame]) -> None:
    nat = series[ZONES[0]].copy()
    for z in ZONES[1:]:
        nat = nat.add(series[z], fill_value=0.0)
    # Hourly
    n = len(nat)
    x = np.arange(1, n + 1) / n * 100.0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, np.sort(nat["entsoe"].to_numpy())[::-1], color="black", lw=2.0, label="ENTSO-E feed-in")
    ax.plot(
        x,
        np.sort(nat["pvlib_phys"].to_numpy())[::-1],
        color="C0",
        lw=1.3,
        ls=":",
        label="PVLib MaStR ×η_inv×aging (gen.)",
    )
    ax.plot(
        x,
        np.sort(nat["atlite_phys"].to_numpy())[::-1],
        color="C2",
        lw=1.3,
        ls=":",
        alpha=0.85,
        label="Atlite CSi ×aging (gen.)",
    )
    ax.plot(
        x,
        np.sort(nat["pvlib_feed"].to_numpy())[::-1],
        color="C0",
        lw=1.8,
        ls="--",
        label=f"PVLib ×η_inv×age×{FEED} (illustrative)",
    )
    ax.plot(
        x,
        np.sort(nat["atlite_feed"].to_numpy())[::-1],
        color="C2",
        lw=1.8,
        ls="-.",
        label=f"Atlite ×age×{FEED} (illustrative)",
    )
    ax.set_xlabel("Percentage of time (%)")
    ax.set_ylabel("Power (MW)")
    ax.set_title(
        "Germany national solar 2023 — hourly duration curve\n"
        f"Dotted = generation after physics; dashed = illustrative ×{FEED} energy match"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    out = result_path("national_hourly_duration.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    copy_to_manuscript(out)

    # Daily max
    daily = nat.groupby(nat.index.normalize()).max()
    n = len(daily)
    x = np.arange(1, n + 1) / n * 100.0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, np.sort(daily["entsoe"].to_numpy())[::-1], color="black", lw=2.0, label="ENTSO-E feed-in")
    ax.plot(
        x,
        np.sort(daily["pvlib_phys"].to_numpy())[::-1],
        color="C0",
        lw=1.3,
        ls=":",
        label="PVLib gen. (η_inv×age)",
    )
    ax.plot(
        x,
        np.sort(daily["atlite_phys"].to_numpy())[::-1],
        color="C2",
        lw=1.3,
        ls=":",
        alpha=0.85,
        label="Atlite gen. (×age)",
    )
    ax.plot(
        x,
        np.sort(daily["pvlib_feed"].to_numpy())[::-1],
        color="C0",
        lw=1.8,
        ls="--",
        label=f"PVLib ×{FEED} (illustrative)",
    )
    ax.plot(
        x,
        np.sort(daily["atlite_feed"].to_numpy())[::-1],
        color="C2",
        lw=1.8,
        ls="-.",
        label=f"Atlite ×{FEED} (illustrative)",
    )
    ax.set_xlabel("Percentage of days (%)")
    ax.set_ylabel("Daily maximum power (MW)")
    ax.set_title(
        "Germany national solar 2023 — daily-max duration curve\n"
        f"Dotted = generation after physics; dashed = illustrative ×{FEED} "
        "(does not recover feed-in shape)"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    out = result_path("national_daily_max_duration.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")
    copy_to_manuscript(out)


def main():
    ensure_results_dir()
    series = load_tso_series()
    metrics = metrics_table(series)
    metrics_path = result_path("tso_solar_feedin_scale_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(metrics.to_string(index=False))

    two_w = result_path("tso_solar_two_weeks.png")
    hourly = result_path("tso_solar_hourly_duration.png")
    daily = result_path("tso_solar_daily_max_duration.png")
    plot_two_weeks(series, two_w)
    plot_tso_hourly_duration(series, hourly)
    plot_tso_daily_max_duration(series, daily)
    # National duration plots owned by investigate_national_solar_derates.py

    for src in (two_w, hourly, daily):
        copy_to_manuscript(src)


if __name__ == "__main__":
    main()
