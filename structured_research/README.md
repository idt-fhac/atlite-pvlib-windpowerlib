# Structured research pipeline

Canonical home for validation studies and paper artifacts.

**Start here:** [`PAPER_CONTRACT.md`](PAPER_CONTRACT.md) — which script owns which paper claim.

## Run

From the repository root:

```bash
python run_all.py --offline                      # use shipped results / local caches
python run_all.py --skip-cutout --skip-extract   # DB allowed, reuse cutouts/data
python run_all.py --step validate_matched_era5
```

Weather for all paper claims: shared ERA5 cutout (`cutouts/germany_2023.nc` / Jülich cutout).
