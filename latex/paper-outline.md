# Paper Outline: Bottom-Up Renewables Modeling Validation (Atlite vs. PVLib/Windpowerlib)

**Target journal**: [Renewable Energy](https://www.sciencedirect.com/journal/renewable-energy) (Elsevier `elsarticle`, numbered citations)  
**Manuscript**: `latex/paper.tex`  
**Authoritative numbers**: `structured_research/results/*.csv`  
**Figures**: `latex/data/` (synced from `structured_research/results/`)

---

## Section 1: Introduction
* Context: VRE modeling for planning and markets (AMIRIS, ASSUME).
* Two paradigms: Atlite (ERA5, vectorized layouts) vs. OEDS/PVLib/Windpowerlib (ECMWF IFS, plant/class-specific).
* Contributions: Jülich PV, Kelmarsh wind farm, TSO/national January + full-year seasonal, \(z_0\) sensitivity, class aggregation runtime.

## Section 2: Methodology & Data Pipeline
* Weather: ERA5 (Atlite; 100 m wind) vs. ECMWF IFS (OEDS; 10 m wind + log shear).
* Plant metadata: MaStR via OEDS (orientation diversity in OEDS; Atlite PV often uniform \(30^\circ/180^\circ\)).
* Benchmarks: Campus Jülich PV (217 kWp, 2023); Kelmarsh wind (6×2.05 MW, UK SCADA); TSO zones; Germany ENTSO-E feed-in (January + full year).
* Note weather–model confounding; ENTSO-E is grid feed-in, not BTM generation.

## Section 3: Solar PV Validation
* **Jülich (full year)**: Actual 195,457 kWh; OEDS std 106.8% (corr 0.917); OEDS temp 105.5% (0.918); Atlite 116.8% (0.921).
* **SAPM**: Summer-noon bias 13.7→7.6 kW (~44%); empirical derate 6.4%→5.2% (`system_losses` 0.936→0.948).
* **DEF solar**: OEDS/Atlite yield 75.8%, corr 0.974 (orientation + weather, not registry forensics).
* **National solar gap**: January OEDS 151.4% / Atlite 212.1%; full-year OEDS 148.2% (multi-cause: BTM, snow, orientation, weather).

## Section 4: Wind Onshore Validation
* **Single V80**: OEDS ~72% of Atlite (10 m log vs ERA5 100 m); curve choice <1% for V80-like machines.
* **Kelmarsh**: Actual 31.50 GWh; Windpowerlib 107.4% (corr 0.877); Atlite 130.8% (0.902); per-turbine WPL 94–120%.
* **Class aggregation**: ~1200× speedup, <2% yield delta.
* **January national**: OEDS wind 84.9% (0.935); Atlite 104.0% (0.967).
* **January TSO**: Wind OEDS 50.7–99.5%; Atlite 69.2–116.5%; solar ≫100% both tools.
* **\(z_0\) sweep**: Discrete optima TenneT 0.15 / 50Hertz 0.40 / Amprion 0.50 / TransnetBW 0.80 → yield-weighted ~91.9% vs 83.0% default (calibration, not independent validation).
* **Granularity**: Single-location national scaling fails (corr 0.40–0.73) vs county-class OEDS (0.908).

## Section 5: Full-Year Seasonal Dynamics
* National seasonal table (wind 78–89% by season; solar 138–194%).
* TSO full-year: TenneT wind 96.7%; TransnetBW 47.6% (complex terrain).

## Section 6: Computational Trade-offs
* DEF benchmarks: Atlite wind/solar ~0.07/0.26 s; PVLib grouped 0.21 s; Windpowerlib sequential 12.9 s vs class-grouped 0.09 s.

## Section 7: Limitations
* Weather–model confound; static \(z_0\); ENTSO-E feed-in vs generation; wakes/availability; metadata uncertainty.

## Section 8: Conclusions
* Temperature physics helps locally; national PV gap is multi-factor; wind gaps are largely shear/weather height; terrain and spatial granularity dominate skill; class grouping makes libraries competitive.

---

### Out of scope for the manuscript
Capacity/registry engineering (postcode leading-zero, EEG join, free-area filters) is excluded from the paper text; keep only in exploratory archives if needed for reproducibility.
