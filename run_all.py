"""
run_all.py
==========
Paper-path orchestration. See structured_research/PAPER_CONTRACT.md.

    python run_all.py [--step STEP] [--skip-extract] [--skip-cutout] [--offline]
    python run_all.py --timed-both
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_all")

PAPER_ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = PAPER_ROOT / "structured_research"
RESULTS_DIR = SCRIPT_DIR / "results"

VENV_PYTHON = PAPER_ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

STEPS = {
    "check_alignment": SCRIPT_DIR / "test_date_alignment.py",
    "prepare_cutouts": SCRIPT_DIR / "prepare_cutouts.py",
    "extract_data": SCRIPT_DIR / "extract_research_data.py",
    "validate_juelich": SCRIPT_DIR / "validate_juelich_solar.py",
    "validate_kelmarsh": SCRIPT_DIR / "validate_kelmarsh_wind.py",
    "validate_matched_era5": SCRIPT_DIR / "validate_matched_era5.py",
    "national_solar_derates": SCRIPT_DIR / "investigate_national_solar_derates.py",
    "tso_solar_feedin_plots": SCRIPT_DIR / "plot_tso_solar_feedin_scale.py",
    "plot_weeks": SCRIPT_DIR / "plot_week_and_diurnal.py",
    "build_report": SCRIPT_DIR / "build_seasonal_comparison_report.py",
    "export_paper_figures": SCRIPT_DIR / "export_paper_figures.py",
}

PAPER_PIPELINE = [
    "check_alignment",
    "prepare_cutouts",
    "extract_data",
    "validate_juelich",
    "validate_kelmarsh",
    "validate_matched_era5",
    "national_solar_derates",
    "tso_solar_feedin_plots",
    "plot_weeks",
    "build_report",
    "export_paper_figures",
]

OFFLINE_PIPELINE = [
    "check_alignment",
    "validate_juelich",
    "validate_kelmarsh",
    "validate_matched_era5",
    "national_solar_derates",
    "tso_solar_feedin_plots",
    "plot_weeks",
    "build_report",
    "export_paper_figures",
]


def run_step(name: str, script_path: Path, env: dict | None = None) -> float:
    """Run a Python script; return elapsed seconds. Raises SystemExit on failure."""
    logger.info("\n%s", "=" * 60)
    logger.info("STEP: %s", name)
    logger.info("%s", "=" * 60)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    t0 = time.time()
    result = subprocess.run([PYTHON, str(script_path)], cwd=PAPER_ROOT, env=run_env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error("Step '%s' FAILED (exit %s) after %.0fs", name, result.returncode, elapsed)
        raise SystemExit(result.returncode)
    logger.info("Step '%s' completed in %.1f s (%.2f min).", name, elapsed, elapsed / 60)
    return elapsed


def run_pipeline(steps: list[str], env: dict | None = None, label: str = "run") -> list[dict]:
    records = []
    t_all = time.time()
    for step in steps:
        elapsed = run_step(step, STEPS[step], env=env)
        records.append({"mode": label, "step": step, "seconds": round(elapsed, 3)})
    total = time.time() - t_all
    records.append({"mode": label, "step": "__TOTAL__", "seconds": round(total, 3)})
    logger.info("\n✓ Pipeline '%s' complete in %.1f s (%.2f min).", label, total, total / 60)
    return records


def write_timing(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp_utc", "mode", "step", "seconds"])
        if write_header:
            w.writeheader()
        ts = datetime.now(timezone.utc).isoformat()
        for r in records:
            w.writerow({"timestamp_utc": ts, **r})
    logger.info("Appended timing → %s", path)


def main():
    parser = argparse.ArgumentParser(description="paper-solar validation pipeline (paper path)")
    parser.add_argument("--step", choices=list(STEPS.keys()), help="Run only a single step")
    parser.add_argument("--skip-cutout", action="store_true", help="Skip prepare_cutouts")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extract_data")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Cache-only: set OFFLINE_MODE=1, skip extract/cutout, refuse DB",
    )
    parser.add_argument(
        "--timed-both",
        action="store_true",
        help="DB-allowed paper path, then OFFLINE paper path; write pipeline_timing.csv",
    )
    args = parser.parse_args()

    timing_path = RESULTS_DIR / "pipeline_timing.csv"

    if args.step:
        env = {"OFFLINE_MODE": "1"} if args.offline else None
        if args.offline and args.step in ("extract_data", "prepare_cutouts"):
            logger.error("Cannot run %s in --offline mode", args.step)
            raise SystemExit(2)
        run_step(args.step, STEPS[args.step], env=env)
        return

    if args.timed_both:
        logger.info("\n##### TIMED RUN 1/2: DB-ALLOWED PAPER PATH #####")
        steps_db = [s for s in PAPER_PIPELINE if s != "prepare_cutouts"]
        germany = SCRIPT_DIR / "cutouts" / "germany_2023.nc"
        if not germany.exists():
            steps_db.insert(1, "prepare_cutouts")
        rec1 = run_pipeline(steps_db, env={"OFFLINE_MODE": "0"}, label="db_allowed")
        write_timing(rec1, timing_path)

        logger.info("\n##### TIMED RUN 2/2: OFFLINE CACHE-ONLY #####")
        rec2 = run_pipeline(OFFLINE_PIPELINE, env={"OFFLINE_MODE": "1"}, label="offline_cache")
        write_timing(rec2, timing_path)

        t1 = next(r["seconds"] for r in rec1 if r["step"] == "__TOTAL__")
        t2 = next(r["seconds"] for r in rec2 if r["step"] == "__TOTAL__")
        print("\n=== TIMING SUMMARY ===")
        print(f"DB-allowed total:     {t1:.1f} s ({t1/60:.2f} min)")
        print(f"Offline cache total:  {t2:.1f} s ({t2/60:.2f} min)")
        print(f"Details: {timing_path}")
        return

    if args.offline:
        rec = run_pipeline(OFFLINE_PIPELINE, env={"OFFLINE_MODE": "1"}, label="offline_cache")
        write_timing(rec, timing_path)
        return

    steps = []
    for step in PAPER_PIPELINE:
        if args.skip_cutout and step == "prepare_cutouts":
            logger.info("Skipping: %s (--skip-cutout)", step)
            continue
        if args.skip_extract and step == "extract_data":
            logger.info("Skipping: %s (--skip-extract)", step)
            continue
        steps.append(step)
    rec = run_pipeline(steps, env={"OFFLINE_MODE": "0"}, label="db_allowed")
    write_timing(rec, timing_path)


if __name__ == "__main__":
    main()
