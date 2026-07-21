# Structured research pipeline

Canonical home for data processing, validation studies, and paper artifacts.

**Start here:** [`PAPER_CONTRACT.md`](PAPER_CONTRACT.md) — which script owns which paper claim.

## Layout

```text
structured_research/
  PAPER_CONTRACT.md   # paper ↔ script ↔ artifact map
  lib/                # shared constants, CF interp, manuscript copy
  explore/            # legacy / one-offs (NOT in run_all)
  data/               # parquet input cache
  cutouts/            # atlite ERA5 cutouts
  results/            # CSV / PNG outputs
  validate_*.py       # paper-path studies
  investigate_national_solar_derates.py
  plot_*.py
  utils.py            # DB loaders, metrics, fleet mapping
```

## Run (paper path only)

```bash
cd paper
python run_all.py --skip-cutout --skip-extract   # use existing cache/cutouts
python run_all.py --offline                      # refuse DB
python run_all.py --step validate_matched_era5
```

Legacy stacks (ECMWF/OEDS, 4-class WPL, shear/soiling one-offs):

```bash
OFFLINE_MODE=1 .venv/bin/python structured_research/explore/<script>.py
```

## Paper stages

| Stage | Script |
|---|---|
| 0 | `test_date_alignment.py` |
| 1 | `prepare_cutouts.py` |
| 2 | `extract_research_data.py` |
| 3 | `validate_juelich_solar.py` |
| 4 | `validate_kelmarsh_wind.py` |
| 5 | `validate_matched_era5.py` — MaStR WPL + PVLib vs Atlite V112@80 m |
| 6 | `investigate_national_solar_derates.py` — η_inv / fleet aging + national figures |
| 7 | `plot_tso_solar_feedin_scale.py` — TSO solar figures only |
| 8 | `plot_week_and_diurnal.py` |
| 9 | `build_seasonal_comparison_report.py` |
| 10 | `export_paper_figures.py` — SVG→PDF for vector figs (incl. TSO/seasonal/budget bars); PNG for scatters → `latex/data/` |

## Design notes

- Weather fixed: ERA5 cutout. Do not mix OEDS/ECMWF into paper claims.
- Wind: Atlite = single `Vestas_V112_3MW`@80 m; WPL = MaStR types + hubs.
- Solar ENTSO-E = feed-in ≠ generation; ×0.62 is illustrative only.
- Shared constants / CF / manuscript copy: `lib/`.
