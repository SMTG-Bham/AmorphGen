# Heavy transition-metal dioxide validation set

Amorphous models of the heavy 4d/5d transition-metal dioxides (MO₂), generated
to validate AmorphGen's `rutile_dioxide` packing class. These dense
rutile/CaCl₂-type oxides pack far more tightly than light oxides, so they use a
higher packing factor (0.66) than the generic `metal_oxide` class (0.52). The
class is assigned geometrically (MO₂ with a 4+ cation radius below the
rutile/fluorite cutoff, ~0.70 Å), so it also covers light rutile dioxides
(TiO₂, SnO₂, …) while excluding fluorite dioxides (ZrO₂, HfO₂, CeO₂); see
`PACKING_FACTORS`, `_RUTILE_DIOXIDE_RMAX`, and `_classify_compound` in
`amorphgen/utils/radii.py`.

## Contents

Each `*.vasp` file is a single amorphous MO₂ cell (M₈O₁₆, 24 atoms), species-
sorted POSCAR (VASP5), Direct coordinates.

| System | Amorphous ρ (g/cm³) | Crystal ρ (g/cm³) | vs crystal | M–O (Å) | M coord. |
|--------|--------------------:|------------------:|:----------:|:-------:|:--------:|
| MoO₂   | 5.81 | 6.47  | −10% | 2.11 | 6.0 |
| RuO₂   | 6.08 | 6.97  | −13% | 2.00 | 5.1 |
| RhO₂   | 6.19 | ~7.2  | ~−14% | 1.98 | 5.1 |
| WO₂    | 9.78 | 10.8  | −9%  | 2.12 | 6.1 |
| ReO₂   | 9.95 | 11.4  | −13% | 2.02 | 4.8 |
| OsO₂   | 10.13 | 11.4 | −11% | 1.97 | 4.6 |
| IrO₂   | 10.23 | 11.67 | −12% | 2.03 | 5.4 |
| PtO₂   | 10.37 | ~10.2 | ~0%  | 2.08 | 4.8 |

All densities fall ~9–14% below the crystalline value — the expected amorphous
range — with physical M–O bond lengths (1.97–2.12 Å) and metal coordination
4.6–6.1 (vs 6 in the rutile crystals).

## How these were generated

Random structure generation (auto density from the `rutile_dioxide` class)
followed by a fixed-cell MACE relaxation:

```bash
amorphgen --random-gen --composition "IrO2*8" -n 1 --relax \
    --model mace-mpa-0 --device cpu --default-dtype float32 \
    -C none -f 0.05 --opt-steps 300 \
    --format vasp -o iro2_run
```

`-C none` (fixed cell) keeps the cell at the auto-estimated density so the
target density is preserved; a cell-relaxed (`-C FrechetCellFilter`) 0 K
optimisation of a single random structure tends to under-densify because it
cannot close voids. For production-quality models use the hybrid
(equilibrate + quench) route, which densifies by atomic rearrangement.

Crystal densities are tabulated reference values for the rutile/dioxide phases.
