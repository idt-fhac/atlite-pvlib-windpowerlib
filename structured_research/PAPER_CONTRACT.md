# Paper ↔ script ↔ artifact contract

Canonical sources for manuscript claims. Do **not** cite `explore/` scripts for paper numbers.

Weather for all paper claims: shared ERA5 cutout (`cutouts/germany_2023.nc` / Jülich cutout).

## Assumptions (state explicitly)

| Item | Value |
|---|---|
| Atlite onshore wind | `Vestas_V112_3MW` @ 80 m (PyPSA-Eur default) |
| Windpowerlib national | MaStR → oedb type by diameter/power + MaStR hub (5 m bins for agg.) |
| PVLib national | MaStR tilt/azimuth where available; else 30°/180°; SAPM |
| Atlite PV | CSi, 30°/180°, panel `inverter_efficiency=0.90` |
| Physics derates (solar) | η_inv = 0.90 (PVLib); η_age = 0.5 %/y (fleet- or plant-weighted) |
| ENTSO-E solar | public-grid **feed-in**, not generation |
| Illustrative feed-in scalar | ×0.62 — **not** a validated model; appendix-only |
| Jülich soiling (Kimber) | Qualitative site note only (~0.3 pp); `explore/investigate_juelich_soiling.py` — **not** a paper-path number source |

## Claim map

| Paper claim | Script | Primary artifact(s) | → `text/data/` |
|---|---|---|---|
| Jülich PV vs meter (tables, duration, scatter, 2-week) | `validate_juelich_solar.py` | `juelich_*` | yes |
| Kelmarsh wind SCADA check | `validate_kelmarsh_wind.py` | `kelmarsh_*` | optional |
| National / TSO wind & solar vs ENTSO-E (matched ERA5) | `validate_matched_era5.py` | `matched_era5_*.csv`, `matched_era5_timeseries.csv`, `matched_era5_tso_timeseries.parquet`, `matched_era5_library_delta.png` | summary plots as needed |
| National solar physics derates + scatter / duration | `investigate_national_solar_derates.py` | `national_solar_derates_*.csv`, `national_*_duration.png`, `national_solar_scatter*.png` | yes |
| TSO solar duration / two-week (physics + illustrative ×0.62) | `plot_tso_solar_feedin_scale.py` | `tso_solar_*.png/csv` only — **does not** own national duration | yes |
| Sample weeks + day/night wind table | `plot_week_and_diurnal.py` | `week_*.png`, `diurnal_wind_daynight.csv` | yes |
| Seasonal comparison matrix (markdown) | `build_seasonal_comparison_report.py` | `main_seasonal_comparison_*` from **matched** + plant CSVs | no |
| Manuscript vector figures (SVG→PDF) + scatter PNG | `export_paper_figures.py` | `*.svg`/`*.pdf` for fortnights, duration, seasonal bars; scatters stay `*.png` | yes (PDF/PNG) |

## Orchestration

```bash
python run_all.py --offline          # cache / results already present
python run_all.py --step validate_matched_era5
```

Paper stages (in order): alignment → (cutouts) → (extract) → Jülich → Kelmarsh → matched ERA5 → national solar derates → TSO solar plots → week/diurnal → report → export figures.
