# atlite-pvlib-windpowerlib

Companion code and aggregated tables for the manuscript comparing **Atlite** versus **PVLib** / **Windpowerlib** for Germany (calendar year 2023) on a shared ERA5 cutout.

Repository: <https://github.com/idt-fhac/atlite-pvlib-windpowerlib>

## Layout

```text
run_all.py                 # paper-path orchestrator
latex/                     # Elsevier elsarticle manuscript for Renewable Energy + figure data/
structured_research/
  PAPER_CONTRACT.md        # which script owns which paper claim
  lib/                     # shared constants, CF interp, figure helpers
  validate_*.py            # plant / national studies
  investigate_*.py
  plot_*.py
  explore/                 # non-paper diagnostics (not cited in manuscript)
  prepare_cutouts.py       # ERA5 cutouts via CDS (online)
  extract_research_data.py # MaStR / ENTSO-E / weather caches (online)
  data/                    # parquet caches (gitignored; regenerate)
  cutouts/                 # *.nc cutouts (gitignored; regenerate)
  results/                 # summary CSV/MD tables shipped with the paper
requirements.txt
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Reproduce tables/figures from shipped results + local caches when present:
python run_all.py --offline

# Full rebuild (needs DB access + CDS credentials for cutouts):
python run_all.py
```

Single stage:

```bash
python run_all.py --step validate_matched_era5
python run_all.py --step export_paper_figures --offline
```

## Data notes

| Input | How to obtain |
|---|---|
| ERA5 cutouts (`structured_research/cutouts/*.nc`) | `python run_all.py --step prepare_cutouts` (CDS API: `~/.cdsapirc`) |
| Parquet caches (`structured_research/data/*.parquet`) | `python run_all.py --step extract_data` (open-energy-data-server / Timescale) |
| Kelmarsh SCADA zip | Place `Kelmarsh_SCADA_2023_5961.zip` in the repo root (or parent directory) |
| Campus Jülich measured AC | Public eview portal; also cached via extract when DB is available |

Large cutouts and parquet caches are **not** in git. Summary comparison tables under `structured_research/results/` are included so paper numbers can be inspected without re-running the full stack.

Offline mode (`OFFLINE_MODE=1` / `--offline`) refuses database connections and expects local caches/results.

## Paper stages

See [`structured_research/PAPER_CONTRACT.md`](structured_research/PAPER_CONTRACT.md) for the claim ↔ script ↔ artifact map.

1. `test_date_alignment.py`
2. `prepare_cutouts.py`
3. `extract_research_data.py`
4. `validate_juelich_solar.py`
5. `validate_kelmarsh_wind.py`
6. `validate_matched_era5.py`
7. `investigate_national_solar_derates.py`
8. `plot_tso_solar_feedin_scale.py`
9. `plot_week_and_diurnal.py`
10. `build_seasonal_comparison_report.py`
11. `export_paper_figures.py`

## License

MIT — see [LICENSE](LICENSE).
