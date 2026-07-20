# Why matched ERA5 Windpowerlib ≫ Atlite

## Verdict (gap decomposition vs Atlite = 100%)

| Case | Energy / Atlite | What it isolates |
|---|---:|---|
| WPL **4-class** (matched study) | **142%** | Full library+fleet mismatch |
| WPL **V80/2000**, hub 80 m, log←100 m | **108%** | Same turbine family as Atlite |
| Curve-WPL / Curve-Atlite on **identical** NUTS3 u80 | **110%** | Power-curve only |
| Curve-Atlite on NUTS3 u80 / Atlite layout | **99%** | Centroid vs plant layout |
| WPL V80 + force cut-out @25 m/s | **108%** (unchanged) | Storm cut-out (negligible here) |
| WPL V80 using raw wnd100m as hub | **120%** | Skipping 100→80 down-extrapolation |

**Most of the 142% gap is the turbine fleet, not weather or ModelChain:**
switching from four SP classes (hubs 80–120 m; ~42% of capacity at 120 m) to a single
V80 @ 80 m drops **142% → 108%**. The remaining **~8–10%** is almost entirely the
**oedb `V80/2000` power curve being fatter than Atlite’s `Vestas_V80_2MW_gridstreamer`**
at low–mid winds (4–5 m/s: WPL ~2.2–2.8× Atlite). Cut-out does not matter for DE 2023
at hub 80 (almost no hours ≥25 m/s). Spatial aggregation is a ~1% effect.

vs ENTSO-E: Atlite V80 ≈ **94%**, WPL V80 ≈ **102%**, 4-class WPL ≈ **134%**.

## Code differences found
1. **Turbine fleet:** matched study used WPL **4 SP classes** (hub 80–120 m). Atlite uses a single **Vestas V80** at **hub_height=80 m** for all capacity. Capacity mix: low/120 m 24.7 GW, med_low/105 m 12.6 GW, med/100 m 13.3 GW, high/80 m 8.4 GW (capacity-weighted mean hub ≈ **106 m** vs Atlite 80 m).
2. **Hub wind:** Atlite **down-extrapolates** ERA5 `wnd100m` → 80 m with logarithmic + `roughness`. Class WPL **up-extrapolated** with Hellman `wnd_shear_exp` to 105–120 m for many MW → systematically higher hub winds.
3. **Power curve cut-out:** Atlite `add_cutout_windspeed=True` forces **P=0 at max V (~25 m/s)**. WPL `V80/2000` stays at **rated power through 25 m/s** (no cut-out in oedb curve). Atlite V80 hub=80.0 m, P=2.0 MW. **In practice for DE 2023 this is negligible** at hub 80.
4. **Low-wind curve:** at 4–5 m/s WPL V80 power ≫ Atlite V80 (see powercurve CSV) — this drives the residual ~10% on matched V80.
5. **Spatial aggregation:** Atlite uses plant-level `layout_from_capacity_list`; WPL uses NUTS3-centroid weather × county capacity (~1% energy difference; r≈1.000).

## Capacity: 58.98 GW total
class
class_high        8.359297
class_low        24.672072
class_med        13.336809
class_med_low    12.616748

Cutout mean wnd100m ≈ 6.26 m/s; gridcell-hours with wnd100m≥25: 1938

## Results vs ENTSO-E
                         case  Ratio_vs_ENTSOE_%  Corr_ENTSOE  Sim_GWh  ENTSOE_GWh
          Atlite V80 (layout)              94.03       0.9667 110885.4    117931.3
       WPL V80 hub80 log←100m             101.93       0.9689 120211.8    117931.3
          WPL V80 + cutout@25             101.93       0.9689 120211.8    117931.3
    WPL V80 using wnd100m raw             113.08       0.9726 133351.4    117931.3
    Curve-Atlite on NUTS3 u80              92.89       0.9661 109547.7    117931.3
       Curve-WPL on NUTS3 u80             101.93       0.9689 120211.8    117931.3
Curve-WPL+cutout on NUTS3 u80             101.93       0.9689 120211.8    117931.3
  WPL 4-class (matched study)             133.83       0.9751 157827.4    117931.3

## Results vs Atlite (same ERA5 cutout)
                               case  Ratio_vs_Atlite_%  Corr_Atlite  WPL_GWh  Atlite_GWh
             WPL V80 hub80 log←100m             108.41       0.9996 120211.8    110885.4
                WPL V80 + cutout@25             108.41       0.9996 120211.8    110885.4
          WPL V80 using wnd100m raw             120.26       0.9982 133351.4    110885.4
          Curve-Atlite on NUTS3 u80              98.79       0.9999 109547.7    110885.4
             Curve-WPL on NUTS3 u80             108.41       0.9996 120211.8    110885.4
      Curve-WPL+cutout on NUTS3 u80             108.41       0.9996 120211.8    110885.4
        WPL 4-class (matched study)             142.33       0.9901 157827.4    110885.4
Curve-WPL / Curve-Atlite (same u80)             109.73       0.9996 120211.8    109547.7

## Reading
- **142% → 108%** by forcing V80@80 m: the big overshoot was **modern/taller class turbines + up-shear**, not a WPL bug.
- Residual **~8–10%** is the **power-curve catalogue difference** (oedb vs Atlite Vestas table), concentrated at low wind.
- Atlite’s single V80 for all DE capacity is a **conservative** fleet proxy vs MaStR-class WPL; neither is “wrong,” they answer different modelling questions.
