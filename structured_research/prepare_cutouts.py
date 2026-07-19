from pathlib import Path
"""
prepare_cutouts.py
==================
Prepares Atlite ERA5 cutouts needed for the full-year 2023 paper analysis.

Requirements:
  - `cdsapi` configured with valid Copernicus CDS credentials
    (~/.cdsapirc or CDSAPI_KEY / CDSAPI_URL env vars)
  - `atlite` installed

Cutouts written to structured_research/cutouts/:
  germany_2023.nc   — Full Germany, full year 2023
  juelich_2023.nc   — Campus Jülich area, full year 2023
  kelmarsh_2023.nc  — Kelmarsh (UK) area, full year 2023

Run from anywhere:
  python structured_research/prepare_cutouts.py
"""
import sys
import logging
import atlite

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CUTOUTS_DIR, cutout_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("prepare_cutouts")


def prepare_germany_annual(path: str):
    """Download full-year 2023 ERA5 cutout covering Germany."""
    if path.exists():
        logger.info(f"Germany annual cutout already exists at {path}. Skipping.")
        try:
            c = atlite.Cutout(path)
            logger.info(f"  → time span: {c.coords['time'].values[0]} → {c.coords['time'].values[-1]}")
            logger.info(f"  → shape: {dict(c.data.dims)}")
        except Exception as e:
            logger.warning(f"  → Could not open existing file: {e}")
        return

    logger.info(f"Preparing full-year Germany ERA5 cutout at {path} ...")
    logger.info("  This will download ~500 MB–1.5 GB from Copernicus CDS. This may take 10–60 minutes.")

    cutout = atlite.Cutout(
        path=path,
        module="era5",
        # Bounding box for Germany (slightly padded)
        x=slice(5.5, 15.5),
        y=slice(47.0, 55.5),
        time="2023",
    )
    cutout.prepare(features=["wind", "influx", "temperature"])
    logger.info(f"Germany annual cutout prepared successfully at {path}.")


def prepare_juelich_annual(path: str):
    """Download full-year 2023 ERA5 cutout for the Jülich area."""
    if path.exists():
        logger.info(f"Jülich annual cutout already exists at {path}. Skipping.")
        return

    logger.info(f"Preparing Jülich annual ERA5 cutout at {path} ...")
    cutout = atlite.Cutout(
        path=path,
        module="era5",
        # Small bounding box around Campus Jülich (50.92 N, 6.36 E)
        x=slice(6.0, 7.0),
        y=slice(50.5, 51.3),
        time="2023",
    )
    cutout.prepare(features=["wind", "influx", "temperature"])
    logger.info(f"Jülich annual cutout prepared successfully at {path}.")


def prepare_kelmarsh_annual(path: str):
    """Download full-year 2023 ERA5 cutout for the Kelmarsh area (UK)."""
    if path.exists():
        logger.info(f"Kelmarsh annual cutout already exists at {path}. Skipping.")
        return

    logger.info(f"Preparing Kelmarsh annual ERA5 cutout at {path} ...")
    cutout = atlite.Cutout(
        path=path,
        module="era5",
        # Bounding box around Kelmarsh Wind Farm (52.40 N, -0.94 E)
        x=slice(-1.5, -0.3),
        y=slice(52.0, 53.0),
        time="2023",
    )
    cutout.prepare(features=["wind", "temperature"])
    logger.info(f"Kelmarsh annual cutout prepared successfully at {path}.")


def main():
    CUTOUTS_DIR.mkdir(parents=True, exist_ok=True)

    germany_path = cutout_path("germany_2023.nc")
    juelich_path = cutout_path("juelich_2023.nc")
    kelmarsh_path = cutout_path("kelmarsh_2023.nc")

    prepare_germany_annual(germany_path)
    prepare_juelich_annual(juelich_path)
    prepare_kelmarsh_annual(kelmarsh_path)

    logger.info("All cutouts ready.")
    logger.info(f"  Germany:  {germany_path}")
    logger.info(f"  Jülich:   {juelich_path}")
    logger.info(f"  Kelmarsh: {kelmarsh_path}")


if __name__ == "__main__":
    main()
