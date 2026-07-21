"""Paper figure export: SVG + PDF for vector plots; PNG for scatters."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from utils import PAPER_ROOT, ensure_results_dir, result_path
from lib.manuscript import TEXT_DATA, copy_to_manuscript

logger = logging.getLogger("figures")

# Stems that go to SVG→PDF (included as PDF in paper.tex)
PAPER_VECTOR_STEMS = (
    "juelich_january_2weeks",
    "juelich_june_2weeks",
    "juelich_hourly_duration",
    "juelich_daily_max_duration",
    "annual_seasonal_summary",
    "national_daily_max_duration",
    "national_hourly_duration",
    "tso_yield_ratios",
    "wind_library_test",
    "solar_residual_budget",
    "tso_matched_week_wind",
    "tso_matched_week_solar",
)

# Scatters stay raster (PNG)
PAPER_RASTER_STEMS = (
    "juelich_scatter_comparison",
    "national_solar_scatter",
    "national_solar_scatter_scaled",
)


def svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    """Convert SVG → PDF. Prefer cairosvg, then inkscape/rsvg-convert."""
    svg_path = Path(svg_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cairosvg

        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        logger.info("SVG→PDF (cairosvg): %s", pdf_path.name)
        return
    except Exception as exc:  # noqa: BLE001 — try next backend
        logger.debug("cairosvg unavailable (%s)", exc)

    inkscape = shutil.which("inkscape")
    if inkscape:
        subprocess.run(
            [
                inkscape,
                str(svg_path),
                "--export-type=pdf",
                f"--export-filename={pdf_path}",
            ],
            check=True,
            capture_output=True,
        )
        logger.info("SVG→PDF (inkscape): %s", pdf_path.name)
        return

    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        subprocess.run(
            [rsvg, "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
            check=True,
            capture_output=True,
        )
        logger.info("SVG→PDF (rsvg-convert): %s", pdf_path.name)
        return

    raise RuntimeError(
        f"Cannot convert {svg_path.name} → PDF: install cairosvg, inkscape, or rsvg-convert"
    )


def save_vector_figure(fig, stem: str, *, also_png: bool = False) -> Path:
    """
    Save figure as SVG, convert to PDF, copy both into latex/data/.
    Returns the manuscript PDF path.
    """
    ensure_results_dir()
    TEXT_DATA.mkdir(parents=True, exist_ok=True)
    svg = Path(result_path(f"{stem}.svg"))
    pdf = Path(result_path(f"{stem}.pdf"))
    fig.savefig(svg, bbox_inches="tight", format="svg")
    try:
        svg_to_pdf(svg, pdf)
    except RuntimeError:
        # Fallback: native matplotlib PDF (still vector) if converters missing
        fig.savefig(pdf, bbox_inches="tight", format="pdf")
        logger.warning("No SVG converter; wrote matplotlib PDF for %s", stem)
    copy_to_manuscript(svg)
    copy_to_manuscript(pdf)
    if also_png:
        png = Path(result_path(f"{stem}.png"))
        fig.savefig(png, dpi=160, bbox_inches="tight")
        copy_to_manuscript(png)
    return TEXT_DATA / f"{stem}.pdf"


def save_raster_figure(fig, stem: str, *, dpi: int = 150) -> Path:
    """Save scatter (etc.) as PNG and copy to latex/data/."""
    ensure_results_dir()
    png = Path(result_path(f"{stem}.png"))
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    copy_to_manuscript(png)
    return TEXT_DATA / f"{stem}.png"


def save_paper_figure(fig, outfile: str | Path, *, dpi: int = 150) -> Path:
    """
    Route a plot to SVG+PDF or PNG from the destination stem.
    `outfile` may be a .png path (legacy callers) or a bare stem.
    """
    path = Path(outfile)
    stem = path.stem
    if stem in PAPER_RASTER_STEMS or "scatter" in stem:
        return save_raster_figure(fig, stem, dpi=dpi)
    if stem in PAPER_VECTOR_STEMS or path.suffix.lower() in {".svg", ".pdf", ""}:
        return save_vector_figure(fig, stem)
    # Default: vector for unknown paper-ish names, else PNG
    if path.suffix.lower() == ".png":
        return save_raster_figure(fig, stem, dpi=dpi)
    return save_vector_figure(fig, stem)
