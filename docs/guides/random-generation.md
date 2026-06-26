# Random structure generation

Generate ensembles of amorphous structures by constrained random sequential placement with automated minimum separation distances derived from Shannon ionic radii.

## How it works

Atoms are placed sequentially into a cubic cell. Each atom must satisfy pair-specific minimum interatomic distances computed automatically from Shannon ionic radii and bonding-type classification. No manual parameter tuning is required.

### Automated minsep

For each element pair, the bond type is classified and the appropriate radii are used:

| Bond type | Radii source | Scale factor | Example |
|-----------|-------------|--------------|---------|
| Ionic (M-O, M-Cl) | Shannon ionic (CN-aware) | 0.80 | In-O: (0.80+1.40)*0.80 = 1.76 |
| Metallic (M-M) | Metallic radii | 0.85 | Al-Al: (1.43+1.43)*0.85 = 2.43 |
| M-M in oxide | max(metallic, sqrt(2)*d(M-O)*0.85) | cap 2.80 | In-In: max(2.84, 2.64) = 2.80 |
| Metalloid (Si-Si) in oxide | max(metallic, sqrt(2)*d(Si-O)*0.85) | cap 2.80 | Si-Si: max(1.99, 2.12) = 2.12 |
| Small anion (O-O) | Shannon ionic | 0.80 | O-O: (1.40+1.40)*0.80 = 2.24 |
| Large anion (Cl-Cl) | Shannon ionic | 0.70 | Cl-Cl: (1.81+1.81)*0.70 = 2.53 |

When `--target-cn` is provided, CN-specific Shannon radii are used (e.g. Si CN=4: 0.26 A vs CN=6: 0.40 A), giving tighter minsep values.

#### Bond-type classifier

The classifier uses an element-type rule (nonmetal / metalloid / metal membership) with a **Pauling electronegativity refinement**: when the type rule would call a pair "ionic" but the Pauling Δχ is below 1.0, the pair is reclassified as covalent. This catches predominantly-covalent compounds whose Shannon ionic radii would otherwise give unrealistically small distances:

| Pair | Δχ | Type rule | Final class | Why it matters |
|------|----|-----------|-------------|----------------|
| Ga–As | 0.37 | ionic (metal+metalloid) | covalent | Shannon radii would give minsep ≈ 0.68 Å — atomic overlap |
| In–P  | 0.41 | ionic | covalent | III–V semiconductor |
| Si–C  | 0.65 | ionic | covalent | Carbide |
| B–N   | 1.00 | ionic | ionic (at threshold) | Sits right on the cutoff; Shannon radii give reasonable BN minsep |
| Ga–N  | 1.23 | ionic | ionic | Stays ionic |
| Li–F  | 3.00 | ionic | ionic | Stays ionic (clearly ionic) |

The Δχ refinement is applied to the metalloid–nonmetal, metalloid–metal, and metal–nonmetal branches. The metallic branch (metal–metal) and the pure-covalent branch (nonmetal–nonmetal, metalloid–metalloid) are unaffected because the type rule already gives the correct answer there. The 1.0 threshold is empirical — it cleanly separates III–V/II–VI semiconductors and carbides (covalent character ≥ 70% by Pauling's formula) from the polar-ionic borderline cases (GaN, ZnO, BeO) where Shannon ionic radii give sensible minsep values.

The bond classifier is exposed as `amorphgen.utils.radii.classify_bond(sym_a, sym_b)` for inspection.

**Citations.** The electronegativity scale follows Pauling's original definition from bond-dissociation energies (Pauling 1932) as standardised in his textbook (Pauling 1960), with the specific numerical values used in AmorphGen taken from Allred's 1961 thermochemical revision — the values now in the CRC Handbook and most chemistry textbooks. References:

- Pauling, L. *J. Am. Chem. Soc.* **54**, 3570–3582 (1932) — original EN derivation.
- Pauling, L. *The Nature of the Chemical Bond*, 3rd ed., Cornell Univ. Press (1960) — textbook scale.
- Allred, A. L. *J. Inorg. Nucl. Chem.* **17**, 215–221 (1961) — **revised values used in the code** (`PAULING_EN` table in `amorphgen/utils/radii.py`).

#### Edge cases at the classifier boundary

The Δχ = 1.0 threshold sits exactly where chemistry genuinely gets ambiguous — bonds with Δχ in the ~0.85–1.15 band have mixed ionic/covalent character. Testing the 50-system JOSS validation set, 54 of 55 cation–anion pairs (98%) agree between the per-pair Pauling classifier and the per-composition material-class radii bucket. The disagreement, and a few other compounds outside the 50-set that sit at the boundary, are listed below:

| System | Pair | Δχ | Material class expects | Pauling rule says | Reality |
|---|---|---|---|---|---|
| MgH₂ | Mg–H | 0.89 | ionic (hydride) | covalent | Rutile structure — predominantly ionic |
| MnS  | Mn–S | 1.03 | covalent (chalcogenide) | ionic | α-MnS is rocksalt — ionic-leaning |
| TiC  | Ti–C | 1.01 | covalent (carbide) | ionic | Rocksalt interstitial carbide — mixed bonding |
| BN   | B–N  | 1.00 | ionic (nitride) | ionic (stays — strict `<`) | Sits exactly on cutoff |

These disagreements are **benign**: the bond classifier governs which radii produce the per-pair `minsep`, while the material classifier governs which radii produce the density estimate. The two answer different questions, and any modest inconsistency at the boundary is absorbed by the subsequent MLIP relaxation. If you generate one of these systems and the auto-derived density looks off (typically ±15–20% from experiment), set `--target-density` explicitly to bypass the auto path for that composition.

### Automated density

Cell volume is estimated by **class-aware sphere packing**: the composition is
classified into a material class, each element is given a radius from the table
appropriate to that class's bonding (Shannon ionic, Cordero covalent, or
Goldschmidt metallic), and the cell is sized so the spheres fill it to the
class's packing factor. Cation oxidation states — needed to select the right
Shannon radius — are assigned automatically by charge balance, using a joint
solver that resolves multivalent cations in mixed-cation / mixed-anion
compounds (e.g. FeTiO3 -> Fe2+/Ti4+, SrTiO3 -> Sr2+/Ti4+) and sums the balance
over every anion-former (oxynitrides, oxyfluorides).

| Material class | Radius source | Packing factor | Examples |
|----------------|---------------|----------------|----------|
| `rutile_dioxide` | Shannon ionic CN6 | 0.66 | TiO2, SnO2, RuO2, IrO2 |
| `fluorite_dioxide` | Shannon ionic CN6 | 0.63 | ZrO2, HfO2, CeO2, ThO2, UO2 |
| `small_cation_nitride` | Shannon ionic CN6 | 0.62 | AlN, GaN, Si3N4, TiN |
| `high_valent_oxide` | Shannon ionic CN6 | 0.60 | V2O5, Nb2O5, MoO3, WO3 (OS ≥ 5) |
| `transition_metal_carbide` | Goldschmidt (cation) + Cordero (C) | 0.60 | TiC, WC, ZrC |
| `alloy` | Goldschmidt metallic | 0.60 | NiTi, CuZr, brass, pure metals |
| `halide` | Shannon ionic CN6 | 0.58 | LiF, NaCl, Li2ZrCl6 |
| `hydride` | Shannon ionic CN6 | 0.55 | LiH, MgH2, NaAlH4 |
| `metal_oxide` | Shannon ionic CN6 | 0.52 | In2O3, Al2O3, Ga2O3, MgO, ZnO |
| `nitride` | Shannon ionic CN6 | 0.52 | ZrN, HfN, ScN (large cation) |
| `covalent_oxide` | Shannon ionic CN6 | 0.50 | SiO2, GeO2, B2O3 |
| `boride` | Cordero covalent | 0.50 | TiB2, MgB2, ZrB2 |
| `covalent_network_oxide` | Cordero covalent | 0.35 | BeO (small polarizing cation) |
| `pnictide` | Cordero covalent | 0.32 | GaAs, InP, InAs |
| `covalent_carbide` | Cordero covalent | 0.32 | SiC, B4C |
| `group_iv` | Cordero covalent | 0.30 | Si, Ge, C |
| `chalcogenide` | Cordero covalent | 0.30 | ZnS, CdTe, GeTe |
| `elemental_semiconductor` | Cordero covalent | 0.28 | a-Se, a-Te, a-As, a-Sb, a-P |

The dioxide split (rutile vs fluorite) and the small- vs large-cation nitride
split are decided by a cation-radius rule; `high_valent_oxide` is gated on
oxidation state ≥ 5. Compositions that match no specific class fall back to a
generic packing factor of 0.52 (Shannon ionic).

Density is the weakest-calibrated part of the auto chain — it sizes a sensible
starting cell, but a subsequent MLIP cell relaxation will correct it. Use
`--target-density` to set the density explicitly when the experimental value is
known.

### Coordination-aware placement — "SC" (optional)

With `--target-cn`, AmorphGen uses **SC ("Seed-Coordinate")** placement: each new atom is added as a bonded neighbour of an existing *under-coordinated* site — the **seed** — by placing it within that seed's bonding shell (`minsep ≤ d ≤ dmax`), which **coordinates** it. Over-coordinating a neighbour is rejected. This builds short-range order directly into the placement instead of relying on relaxation alone, giving more physical structures.

**Where the name comes from.** SC is the placement half of the **Seed-Coordinate-Anneal (SCA)** algorithm of Youn et al., *Comput. Mater. Sci.* (2014). AmorphGen factors the *Anneal* step out into its own stages — the MLIP geometry optimisation (`--relax`) and the melt-quench / hybrid workflow — so the placement step is just **Seed-Coordinate** (hence "SC", not "SCA").

Disable with `--no-sc` to fall back to plain random rejection sampling (no coordination bias).

### Transparency: the auto-derive log line

Every `--random-gen` run writes a single one-line summary at the top of `random_gen.log` capturing every chemistry-informed decision the auto chain made — so you can see *why* a particular minsep / density / target CN was used without reading the code. Example for Ga₂O₃:

```
[auto-derive] Ga16O24 → metal_oxide, OS{Ga:+3}, CN{Ga:5}, minsep{Ga-Ga:2.43 metallic | Ga-O:1.62 ionic Δχ=1.63 | O-O:2.24 anion-pack}, ρ=4.44 g/cm³ L=8.25 Å
```

Each field:

| Field | What it means |
|---|---|
| `Ga16O24 → metal_oxide` | Composition routed to the `metal_oxide` material class |
| `OS{Ga:+3}` | Cation oxidation state inferred from charge balance against the anion |
| `CN{Ga:5}` | Target coordination auto-detected per cation |
| `minsep{pair:value class [Δχ=val]}` | Per-pair: bond class + Pauling Δχ (shown only when the ionic classification is at stake) + minsep value in Å |
| `ρ=… g/cm³  L=… Å` | Auto-estimated mass density and cubic cell length |

The line is grep-friendly: `grep "auto-derive" random_gen.log` retrieves it as a single line per generation run. Bond classes shown are `ionic`, `covalent`, `metallic`, and `anion-pack` (same-element nonmetal pairs use a separate anion-packing scale factor — see "Bond-type classifier" above).

## CLI examples

```bash
# Formula format (In2O3 * 16 formula units = 80 atoms)
amorphgen --random-gen --composition "In2O3*16" --n-structures 20

# Atom count format (equivalent)
amorphgen --random-gen --composition In=32,O=48 --n-structures 20

# With target CN and density
amorphgen --random-gen --composition "SiO2*16" --n-structures 20 \
    --target-cn Si=4,O=2 --target-density 2.2

# With relaxation
amorphgen --random-gen --composition "In2O3*8" --n-structures 10 \
    --relax --model mace-mpa-0 --cell-filter none
```

## Python API

```python
from amorphgen import generate_random, batch_random

# Single structure (auto minsep, auto density from Shannon radii)
atoms = generate_random(
    composition={"In": 16, "O": 24},   # atom counts (Python API)
    seed=42,
)

# Batch generation (20 structures of SiO2)
paths = batch_random(
    composition={"Si": 16, "O": 32},
    n_structures=20,
    target_density=2.2,                 # optional, auto if omitted
    target_cn={"Si": 4, "O": 2},       # optional, auto if omitted
)
```

> **Note:** The Python API takes atom counts as a dict. The CLI also accepts
> formula format: `--composition "SiO2*16"` is equivalent to `{"Si": 16, "O": 32}`.

## Accessing radii data

```python
from amorphgen.utils.radii import get_ionic_radius, classify_bond, default_minsep

# Shannon ionic radius
get_ionic_radius("In", cn=6)   # 0.80 A
get_ionic_radius("In", cn=4)   # 0.62 A

# Bond classification
classify_bond("In", "O")       # "ionic"
classify_bond("Si", "Si")      # "covalent"

# Auto minsep for a composition
ms = default_minsep(["In", "O"], target_cn={"In": 4})
# {"In-In": 3.11, "In-O": 1.72, "O-O": 2.24}
```

See {doc}`/api/random-gen` for the full API reference.
