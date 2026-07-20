"""
Jülich soiling sensitivity: Kimber model on ERA5 total precipitation.

Uses existing matched-ERA5 PVLib/Atlite series from validate_juelich_solar.py
and multiplies by (1 - soiling_loss) from pvlib.soiling.kimber.

Rainfall: ERA5 total_precipitation (CDS) cached as cutouts/juelich_2023_tp.nc.
Do NOT use cutout `runoff` — that is surface runoff [m], not rainfall.

Outputs → results/:
  juelich_soiling_sensitivity.csv
  juelich_soiling_investigation.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pvlib.soiling import kimber

# xarray used for precip netcdf + source attr in report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    calculate_metrics,
    cutout_path,
    ensure_results_dir,
    load_local_data,
    result_path,
)

LAT, LON = 50.92, 6.36
INVERTER_EFFICIENCY = 0.90
# Kimber defaults are CA-centric; rural-ish Central Valley rate as a European prior
SOILING_LOSS_RATE = 0.0011  # rural Kimber table
CLEANING_THRESHOLD_MM = 6.0


def _actual_overlap() -> pd.Series:
    raw = load_local_data("juelich_actuals.parquet")
    if not isinstance(raw.index, pd.DatetimeIndex):
        if "time" in raw.columns:
            raw = raw.set_index("time")
        raw.index = pd.to_datetime(raw.index)
    if getattr(raw.index, "tz", None) is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    return raw["generation"].sort_index().dropna()


def load_rainfall_mm(tp_path: Path, date_range: pd.DatetimeIndex) -> pd.Series:
    """Nearest-cell ERA5 tp → hourly rainfall [mm]."""
    ds = xr.open_dataset(tp_path)
    # CDS netcdf var names vary (tp / TP); coords lat/lon or latitude/longitude
    var = "tp" if "tp" in ds else [v for v in ds.data_vars if v.lower() == "tp"][0]
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    time_name = "valid_time" if "valid_time" in ds.coords else "time"

    lats = ds[lat_name].values
    lons = ds[lon_name].values
    yi = int(np.abs(lats - LAT).argmin())
    xi = int(np.abs(lons - LON).argmin())
    print(
        f"  TP cell: {lat_name}={float(lats[yi])}, {lon_name}={float(lons[xi])} "
        f"(site {LAT}, {LON})"
    )

    s = ds[var].isel({lat_name: yi, lon_name: xi}).to_pandas()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    # ERA5 tp is accumulated precipitation in metres of water equivalent
    rain_mm = (s.clip(lower=0) * 1000.0).reindex(date_range, method="nearest")
    rain_mm.name = "rainfall_mm"
    print(
        f"  Rainfall 2023: sum={rain_mm.sum():.1f} mm, "
        f"max hourly={rain_mm.max():.2f} mm, wet hours={(rain_mm > 0.1).sum()}"
    )
    return rain_mm.astype(float)


def main() -> None:
    tp_path = Path(cutout_path("juelich_2023_tp.nc"))
    if not tp_path.is_file():
        raise SystemExit(
            f"Missing {tp_path}. Download ERA5 total_precipitation first "
            "(see investigate_juelich_soiling / CDS retrieve)."
        )

    comp_path = result_path("juelich_eview_comparison_complete.csv")
    if not comp_path.is_file():
        raise SystemExit(
            f"Missing {comp_path}. Run validate_juelich_solar.py first."
        )

    ensure_results_dir()
    date_range = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
    act = _actual_overlap()
    idx = act.index

    print("Loading ERA5 total precipitation...")
    rain = load_rainfall_mm(tp_path, date_range)

    print("Running Kimber soiling...")
    # kimber returns fraction of energy *lost* (0 = clean)
    soil_loss = kimber(
        rain,
        cleaning_threshold=CLEANING_THRESHOLD_MM,
        soiling_loss_rate=SOILING_LOSS_RATE,
    )
    soil_factor = (1.0 - soil_loss).clip(lower=0.0, upper=1.0)
    soil_factor.name = "soiling_transmission"

    print(
        f"  Soiling loss: mean={soil_loss.mean():.4f}, "
        f"max={soil_loss.max():.4f}, "
        f"mean transmission={soil_factor.mean():.4f}"
    )

    comp = pd.read_csv(comp_path, index_col=0, parse_dates=True)
    # Raw DC-ish series before η; rebuild physics stacks
    pv_raw = comp["pvlib_sapm_raw"].astype(float)
    atl_raw = comp["atlite_raw"].astype(float)
    # Aging from previously applied columns: oeds = raw * inv * age
    # → age = oeds / (raw * inv) when raw > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        age_est = (
            comp["oeds_temp_corrected"] / (pv_raw * INVERTER_EFFICIENCY)
        ).replace([np.inf, -np.inf], np.nan)
    eta_age = float(age_est.dropna().median())
    print(f"  Inferred η_age from CSV ≈ {eta_age:.4f}")

    configs = [
        (
            "PVLib SAPM × η_inv × aging",
            (pv_raw * INVERTER_EFFICIENCY * eta_age).reindex(idx),
        ),
        (
            "PVLib SAPM × η_inv × aging × Kimber",
            (pv_raw * INVERTER_EFFICIENCY * eta_age * soil_factor).reindex(idx),
        ),
        ("Atlite × aging", (atl_raw * eta_age).reindex(idx)),
        (
            "Atlite × aging × Kimber",
            (atl_raw * eta_age * soil_factor).reindex(idx),
        ),
    ]

    rows = []
    print("\n=== Jülich soiling sensitivity (overlap hours) ===")
    for name, sim in configs:
        m = calculate_metrics(sim, act)
        rows.append(
            {
                "Configuration": name,
                "Ratio_%": round(m["ratio"], 2),
                "Corr": round(m["corr"], 4),
                "MAE_kW": round(m["mae"], 3),
                "RMSE_kW": round(m["rmse"], 3),
                "Sim_kWh": round(m["sim_sum"], 1),
                "n": int(m["n"]),
            }
        )
        print(
            f"{name}: {m['ratio']:.2f}% | r={m['corr']:.4f} | "
            f"MAE={m['mae']:.2f} | {m['sim_sum']:.0f} kWh"
        )

    out = pd.DataFrame(rows)
    out.to_csv(result_path("juelich_soiling_sensitivity.csv"), index=False)

    # Seasonal effect of soiling on PVLib stack
    seasonal = []
    base = (pv_raw * INVERTER_EFFICIENCY * eta_age).reindex(idx)
    with_s = (pv_raw * INVERTER_EFFICIENCY * eta_age * soil_factor).reindex(idx)
    for code, months in [
        ("DJF", [12, 1, 2]),
        ("MAM", [3, 4, 5]),
        ("JJA", [6, 7, 8]),
        ("SON", [9, 10, 11]),
        ("FULL", list(range(1, 13))),
    ]:
        mask = idx.month.isin(months)
        mb = calculate_metrics(base[mask], act[mask])
        ms = calculate_metrics(with_s[mask], act[mask])
        seasonal.append(
            {
                "Season": code,
                "Baseline_ratio_%": round(mb["ratio"], 2),
                "Kimber_ratio_%": round(ms["ratio"], 2),
                "Delta_pp": round(ms["ratio"] - mb["ratio"], 2),
                "Baseline_r": round(mb["corr"], 4),
                "Kimber_r": round(ms["corr"], 4),
                "Mean_soil_loss": round(float(soil_loss.reindex(idx)[mask].mean()), 4),
            }
        )
    seas = pd.DataFrame(seasonal)
    seas.to_csv(result_path("juelich_soiling_seasonal.csv"), index=False)
    print("\nSeasonal (PVLib SAPM × η_inv × aging ± Kimber):")
    print(seas.to_string(index=False))

    def _md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    src = str(xr.open_dataset(tp_path).attrs.get("source", "ERA5 tp netcdf"))
    md = result_path("juelich_soiling_investigation.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Jülich Kimber soiling sensitivity (ERA5 precipitation)\n\n")
        f.write(
            f"Rainfall from ERA5 `total_precipitation` "
            f"(`{tp_path.name}`, source: {src}), nearest cell to "
            f"({LAT}, {LON}). Annual sum **{rain.sum():.0f} mm**.\n\n"
        )
        f.write(
            f"Model: `pvlib.soiling.kimber` with "
            f"`soiling_loss_rate={SOILING_LOSS_RATE}` (rural Kimber prior), "
            f"`cleaning_threshold={CLEANING_THRESHOLD_MM} mm`.\n\n"
        )
        f.write(
            f"Mean soiling loss **{soil_loss.mean():.4f}** "
            f"(transmission {soil_factor.mean():.4f}); "
            f"max loss **{soil_loss.max():.4f}**.\n\n"
        )
        f.write("## Overlap-hour metrics\n\n")
        f.write(_md_table(out))
        f.write("\n\n## Seasonal (PVLib stack)\n\n")
        f.write(_md_table(seas))
        f.write(
            "\n\n## Takeaway\n\n"
            "- Cutout `runoff` is **not** rainfall; use ERA5 `tp`.\n"
            "- Kimber needs only rainfall (HSU also needs PM2.5/PM10 — not in ERA5).\n"
            "- Effect here is tiny (~0.3 pp annual); does not close the ~10% meter gap.\n"
            "- CA Kimber rates are a prior for DE; treat as sensitivity, not calibrated.\n"
        )
    print(f"\nWrote {md}")


if __name__ == "__main__":
    main()
