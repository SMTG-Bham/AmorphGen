# Random structure generation

Generate ensembles of disordered structures by constrained random sequential placement with automated minimum separations from ionic, covalent and metallic radii.

## How it works

Atoms are placed sequentially into a cubic cell. Pair-specific minimum separations are derived from radii and bonding-type classification. Composition-derived defaults provide starting structures; validate their density and local structure for the system being studied.

### Automated minsep

For each element pair, the bond type is classified and the appropriate radii are used:

| Bond type | Radii source | Scale factor | Example |
|-----------|-------------|--------------|---------|
| Ionic (M-O, M-Cl) | Shannon ionic (CN-aware) | 0.80 | In-O: (0.80+1.40)*0.80 = 1.76 |
| Metallic (M-M) | Metallic radii | 0.85 | Al-Al: (1.43+1.43)*0.85 = 2.43 |
| M-M in oxide (anions with up to 6 cations) | max(metallic, sqrt(2)*d(M-O)*0.85) | cap 2.80 | In-In: min(2.80, max(2.84, 2.64)) = 2.80 |
| M-M in a cation-rich compound (anions with 7-12 cations) | 1.155*d(M-X)*0.85 (7-8), d(M-X)*0.85 (9-12) | cap 2.80 | Li-Li in Li3N: (0.59+1.46)*0.85 = 1.74 |
| M-M in a metal-rich compound (anions with over 12 cations) | Metallic radii | 0.85 | Ni-Ni in Ni80P20: (1.24+1.24)*0.85 = 2.11 |
| Metalloid (Si-Si) in oxide | max(metallic, sqrt(2)*d(Si-O)*0.85) | cap 2.80 | Si-Si: max(1.99, 2.12) = 2.12 |
| Small anion (O-O) | Shannon ionic | 0.80 | O-O: (1.40+1.40)*0.80 = 2.24 |
| Large anion (Cl-Cl) | Shannon ionic | 0.70 | Cl-Cl: (1.81+1.81)*0.70 = 2.53 |
| Nonmetal cation to anion (P-O, S-O, C-O) | Shannon cation radius (top state, lowest CN) | 0.80 | P-O: (0.17+1.40)*0.80 = 1.26 |
| Nonmetal cation to cation | sqrt(2)*d(X-O), or 2*d(X-O) for the same element | 0.85, cap 2.80 | P-P: 2*(0.17+1.40)*0.85 = 2.67 |
| Two H+ (water, hydroxides, acids) | sqrt(2)*d(H-O) | 0.85 | H-H: 1.414*(-0.38+1.40)*0.85 = 1.23 |
| Same-element anions that must bond (anion excess) | Cordero covalent | 0.80 (0.70 for C, N, O) | Se-Se in Ge20Se80: 2*1.20*0.80 = 1.92 |
| H-H of a hydride | 0.8 of the 2.1 Å Switendick limit | | H-H in TiH2: 1.68 |
| Metal-P bond with Δχ < 1 (Ni2P, CoP, Ni80P20) | Metallic + Cordero covalent | 0.80 | Ni-P: (1.24+1.07)*0.80 = 1.85 |
| O-O of a nitrate or carbonate | sqrt(3)*d(X-O), the triangle edge | 0.85 | O-O in NaNO3: 1.732*1.296*0.85 = 1.91 |

When `--target-cn` is provided, CN-specific Shannon radii are used (e.g. Si CN=4: 0.26 A vs CN=6: 0.40 A), giving tighter minsep values.

#### Why these values

- **Bonds at 0.80 of the radii sum.** Coordination-aware placement puts a bonded neighbour anywhere between minsep and 1.5 × minsep (the default `dmax`). At 0.80 that shell runs from 0.80 to 1.20 times the Shannon bond, centred on it, so placed bonds scatter ±20 % about the ideal length. AIRSS buildcell uses separations at 80-90 % of the equilibrium distances for the same reason. Measured against crystals, every bonded floor sits at 0.70-0.93 of the real bond: oxides, halides, nitrides, carbides, III-V and II-VI semiconductors, borides, hydrides, sulfides and alloys alike.
- **M-M at 0.85.** Two metals are kept a little stiffer than a bond, so placement does not build metal clusters.
- **Cations across an anion.** In a compound two cations meet across the anion they share. Where each anion has at most 6 cations (MO, M2O3, MO2), the closest approach is the shared edge of two octahedra: a 90° M-X-M angle, √2 × d(M-X). Where the anions average more cations (antifluorite Li2O and Li2S, anti-perovskite Li3OCl, Li3N, Cu2S), the cation polyhedra share edges of a cube (70.5°, 1.155 × d) or faces (60°, 1.0 × d). The metallic radius drops out there: it belongs to the neutral atom, and Li+ sits 2.11 Å from Li+ in Li3N against 3.04 Å in Li metal. The mean anion coordination is the cations' target CNs (the automatic ones when none are given) weighted by count, per anion. Beyond 12 no anion holds them all (Fe3C, Ni80P20): the metals touch, at their metallic contact.
- **Anions at packing distance.** Same-element anions are kept at 0.80 of twice their Shannon radius (0.70 for Cl, Br, I, S, Se, Te, which are larger and softer), so they never bond. That is right for oxides, where an O-O bond is a peroxide defect.
- **Unless they must bond.** When the cations, at their top oxidation states (3 for the lone-pair As, Sb and Bi), cannot reduce every anion to its nominal charge, the excess anions bond to each other (the Zintl-Klemm count): S2 2- in FeS2 and polysulfides, Se chains in Ge20Se80 and S-rich As-S glasses, C2 2- in CaC2, N3 - in NaN3, O2 2- and O2 - in Li2O2 and KO2, I3 - in CsI3. Their floor is then the X-X bond at 0.8 of twice the Cordero radius (0.7 for the multiple bonds of C, N and O). The least electronegative anion takes the bonds. With other anions present only S, Se, P or C qualify, and O needs an excess of 40 % rather than 10 %, so a doped oxide or a cation above the radii table (the Ni3+ of LiNiO2) does not read as a peroxide. Stoichiometric GeSe2 and As2S3 have no excess and stay chemically ordered.
- **Hydrogen.** Two H+ can share one O, as in a water molecule (H-H 1.52 Å), so they get the right-angle contact across the O rather than two bonds end to end. H- in a metal hydride stays above about 2.1 Å from H- (Switendick), and the geminal H of BH4- or ReH9 2- above 1.9 Å, so the hydride floor is 0.8 of 2.1 Å.
- **Covalent phosphides.** A metal-P bond with Δχ < 1 (Ni, Co, Fe, Mo, W phosphides) is covalent, and the 2.12 Å P3- radius put Ni-P at 2.25 Å against the real 2.2-2.3 Å. It takes metallic and Cordero radii instead, like a metalloid. Ionic phosphides (Li3P, Na3P) keep P3-.
- **Oxoanion roles.** Which nonmetals are cations comes from charge balance at the top oxidation states. Where that leaves a tie or a mismatch, a nonmetal becomes the cation if only that role balances exactly with the tabulated states: Ag2SO4 is Ag+ with a sulfate, Tl3PO4 a phosphate, CrOOH and MnOOH hydroxides.
- **The caps.** Random sequential placement jams once hard spheres fill about 38 % of the volume. Treating the floors as hard spheres, the oxides fill 26-36 % at their measured densities (O in SiO2 0.26, O in Al2O3 0.33, In in In2O3 0.36). The 2.80 Å and 3.00 Å caps keep large ions inside that limit.

#### Checked against real crystals

A floor must not exclude the contacts of the material it is meant to build. Against the shortest contacts of 30,616 experimentally observed Materials Project structures (ICSD-matched, within 25 meV/atom of the convex hull, up to 80 sites), 0.4 % of the cation-anion bond floors sit above the bond, with a median of 0.82 of it. Counting every pair, 5.7 % of the inorganic structures still have one floor above one of their own contacts (11.9 % before the anion-excess, hydrogen, phosphide, triangle-edge, cation-rich and role rules above). Those left need bonds the composition alone does not reveal:

| Family | Examples | Floor vs the real distance |
|---|---|---|
| Bonds between cations | Hg2 2+ (Hg2Cl2), Ga-Ga (GaSe), In-In (InSe), P-P (P4S3, the P2S6 4- of Li2S-P2S5 glasses), Ge-Ge and As-As in chalcogen-poor glasses, Mo, Re, W and Nb clusters | Hg-Hg 2.80 Å vs 2.53; P-P 2.80 vs 2.2; Ge-Ge 2.68 vs 2.45 |
| MnP-type phosphides | CoP, FeP, CrP, WP | P-P 2.97 Å vs 2.6-2.8; Co-Co 2.80 vs 2.6-2.7 |
| Anion excess hidden by the cation's top state | MoS3 (Mo4+ with S2 2-), CuS, CuSe, CeS2, PdSe2 | S-S 2.58 Å vs 2.04-2.1 |
| Close Cu+ and Li+ contacts | CuI, Cu2Se, LiOH, LiBO2 | 2.60-2.80 Å vs 2.4-2.7 |
| Uranyl and other actinide or Tc oxo-ions | UO2 2+ salts, NpO2 +, TcO4 - | U-O about 1.8 Å vs 1.75-1.8 (states missing from the radii table) |
| Molecular and organic solids (C-H, N-H) | 16 % of them blocked | outside AmorphGen's scope |

Stoichiometric chalcogenide glasses (GeSe2, GeS2, As2S3) also exclude their few homopolar bonds (Se-Se, Ge-Ge, As-As). That keeps the network chemically ordered on purpose; melt-quench creates those bonds.

For these systems, pass a full table with the offending pairs lowered to about 0.8 of their real distance. `--minsep` replaces the whole automatic table: a pair it leaves out falls back to 1.5 Å and gets no coordination-aware bonds. So copy every pair from the `[auto-derive]` line of `random_gen.log` and change only the ones you need:

```bash
amorphgen --random-gen --composition Ge=40,Se=60 --minsep Ge-Ge=1.95,Ge-Se=2.06,Se-Se=2.77
amorphgen --random-gen --composition Li=24,P=12,S=36 --minsep Li-Li=2.80,Li-P=2.77,Li-S=2.08,P-P=1.75,P-S=1.61,S-S=2.58
amorphgen --random-gen --composition Mo=16,S=48 --minsep Mo-Mo=2.20,Mo-S=1.94,S-S=1.65
amorphgen --random-gen --composition Co=48,P=48 --minsep Co-Co=2.20,Co-P=1.86,P-P=2.10
```

#### Bond-type classifier

The classifier uses an element-type rule (nonmetal / metalloid / metal membership) with a **Pauling electronegativity refinement**: when the type rule would call a pair "ionic" but the Pauling Δχ is below 1.0, the pair is reclassified as covalent. This catches predominantly-covalent compounds whose Shannon ionic radii would otherwise give unrealistically small distances:

| Pair | Δχ | Type rule | Final class | Why it matters |
|------|----|-----------|-------------|----------------|
| Ga–As | 0.37 | ionic (metal+metalloid) | covalent | Shannon radii would give minsep ≈ 0.68 Å, atomic overlap |
| In–P  | 0.41 | ionic | covalent | III–V semiconductor |
| Si–C  | 0.65 | ionic | covalent | Carbide |
| B–N   | 1.00 | ionic | ionic (at threshold) | Sits right on the cutoff; Shannon radii give reasonable BN minsep |
| Ga–N  | 1.23 | ionic | ionic | Stays ionic |
| Li–F  | 3.00 | ionic | ionic | Stays ionic (clearly ionic) |

The Δχ refinement is applied to the metalloid–nonmetal, metalloid–metal, and metal–nonmetal branches. The metallic branch (metal–metal) and the pure-covalent branch (nonmetal–nonmetal, metalloid–metalloid) are unaffected because the type rule already gives the correct answer there. The 1.0 threshold is empirical, it cleanly separates III–V/II–VI semiconductors and carbides (covalent character ≥ 70% by Pauling's formula) from the polar-ionic borderline cases (GaN, ZnO, BeO) where Shannon ionic radii give sensible minsep values.

The bond classifier is exposed as `amorphgen.utils.radii.classify_bond(sym_a, sym_b)` for inspection.

The electronegativity scale follows Pauling's original definition from bond-dissociation energies (Pauling 1932) as standardised in his textbook (Pauling 1960), with the specific numerical values used in AmorphGen taken from Allred's 1961 thermochemical revision, the values now in the CRC Handbook and most chemistry textbooks. References:

- Pauling, L. *J. Am. Chem. Soc.* **54**, 3570–3582 (1932), original EN derivation.
- Pauling, L. *The Nature of the Chemical Bond*, 3rd ed., Cornell Univ. Press (1960), textbook scale.
- Allred, A. L. *J. Inorg. Nucl. Chem.* **17**, 215–221 (1961), **revised values used in the code** (`PAULING_EN` table in `amorphgen/utils/radii.py`).

#### Edge cases at the classifier boundary

The Δχ = 1.0 threshold is a heuristic, so pairs near it can receive different
classifications from the per-pair bond rule and the composition-based density
rule. Examples include:

| System | Pair | Δχ | Material class expects | Pauling rule says | Context |
|---|---|---|---|---|---|
| MgH₂ | Mg–H | 0.89 | ionic (hydride) | covalent | Rutile structure, predominantly ionic |
| MnS  | Mn–S | 1.03 | covalent (chalcogenide) | ionic | α-MnS is rocksalt, ionic-leaning |
| TiC  | Ti–C | 1.01 | covalent (carbide) | ionic | Rocksalt interstitial carbide, mixed bonding |
| BN   | B–N  | 1.00 | ionic (nitride) | ionic (stays, strict `<`) | Sits exactly on cutoff |

The bond classifier selects radii for the per-pair `minsep`; the material
classifier selects radii for the density estimate. Check both derived values
for these boundary cases. Subsequent relaxation may change the local
structure and density, but does not guarantee agreement with experiment.
Set `--target-density` explicitly when a validated density is available.

#### Nonmetal cations: oxoanions and hydroxides

In a phosphate, a sulfate or a carbonate, the P, S or C is a nonmetal but it is the cation of its oxoanion, and in a hydroxide the H is. Which nonmetals are cations is decided by charge balance (`radii.cation_nonmetals`): an element of the anion table is promoted to cation when that brings the compound closer to neutrality, and C and P are cations when an anion more electronegative than them is present and they balance the charge better as cations than as C⁴⁻ / P³⁻. So the P of Li₃PO₄ and Li₃PS₄, the S of Li₂SO₄, the C of CaCO₃ and the H of Mg(OH)₂ are cations, while the carbide C of SiOC, the P of InP and the S of La₂O₂S stay anions. A nonmetal cation:

- bonds to its anions at its Shannon cation radius (P-O 1.26, S-O 1.22, C-O 1.06, O-H 0.82 Å, about 0.8 of the bond, as for Si-O);
- targets the ligand count of its oxoanion: 4 in PO₄³⁻, SO₄²⁻ and ClO₄⁻, 3 in CO₃²⁻, NO₃⁻, IO₃⁻ and a sulfite, 1 for H;
- takes its oxidation state from the same charge balance (S⁶⁺ in Li₂SO₄, which leaves Li⁺), and is sized as that cation in the density estimate.

With a composition, `classify_bond(sym_a, sym_b, composition)` applies these roles: a nonmetal cation and an anion are `ionic`, a nonmetal cation and another cation are `cation-cation` (a second-shell contact across the anion, never a bond), and two cations of a compound with anions are never `ionic`. That last rule is why Na-B in a borate and K-Si in a silicate are placed 2.16 and 2.75 Å apart, like Na-Si, rather than as the ionic bond their Δχ ≥ 1 would suggest.

#### Hydrogenated networks: a-Si:H, a-C:H

C, Si and Ge with H and nothing else, with at most one H per host atom, are the `hydrogenated_network` class: a-Si:H, a-Ge:H, a-C:H up to the polymer-like 50 % H, a-SiC:H and a-SiGe:H. H there caps a host atom through a covalent bond; it is not the H⁻ of LiH, MgH₂ or NaAlH₄, which stay `hydride`. The host keeps what it has without H: its Cordero radii and packing factor, its minimum separations (Si-Si 1.87 Å as in a-Si, C-C 1.22 Å as in a-C, the C-C anion packing of SiC) and its bonds. Each H targets one bond (the hosts target 4) and is kept at 0.8 of its bond from any host (C-H 0.86, Si-H 1.18 Å). Two H can share a host atom (H-H 1.21 Å in a-C:H, 1.67 Å in a-Si:H) but cannot form H₂.

In the density estimate H is sized at 0.90 Å, not its Cordero 0.31 Å, which would give it no volume: each H replaces a host-host bond and brings free volume with it, so the density falls as the H content rises. Si₆₄H₈ (11 % H) comes out at 2.30 g/cm³ (glow-discharge a-Si:H ≈ 2.2), and a-C:H at 2.19, 1.84, 1.52 and 1.24 g/cm³ for 20, 30, 40 and 50 % H (hard a-C:H 1.6–2.2 g/cm³ at 30–40 % H, polymer-like 1.2–1.6 g/cm³ at 40–50 %). The real density also depends on how the film was grown (sp³ fraction, voids), so use `--target-density` when the measured value is known.

### Automated density

Cell volume is estimated by **class-aware sphere packing**: the composition is
classified into a material class, each element is given a radius from the table
appropriate to that class's bonding (Shannon ionic, Cordero covalent, or
Goldschmidt metallic), and the cell is sized so the spheres fill it to the
class's packing factor. Cation oxidation states, needed to select the right
Shannon radius, are assigned automatically by charge balance, using a joint
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
| `oxyhalide` | Shannon ionic CN6 | 0.52–0.58 (interpolated by halogen fraction of the halogen+O anion pool; dopant-level halogens, <10%, keep the oxide class) | BiOCl, LaOCl, NaTaOCl4, ZrOCl2 |
| `hydride` | Shannon ionic CN6 | 0.55 | LiH, MgH2, NaAlH4 |
| `metal_oxide` | Shannon ionic CN6 | 0.52 | In2O3, Al2O3, Ga2O3, MgO, ZnO |
| `nitride` | Shannon ionic CN6 | 0.52 | ZrN, HfN, ScN (large cation) |
| `covalent_oxide` | Shannon ionic CN6 | 0.50 | SiO2, GeO2, B2O3 |
| `boride` | Goldschmidt cation + Cordero B | 0.60 | TiB2, MgB2, ZrB2 (LaB6-type cage borides run ~100% of crystal; use `--density-scale` <1 if placement struggles) |
| `covalent_network_oxide` | Cordero covalent | 0.35 | BeO (small polarizing cation) |
| `pnictide` | Cordero covalent | 0.32 | GaAs, InP, InAs |
| `covalent_carbide` | Cordero covalent | 0.32 | SiC, B4C |
| `group_iv` | Cordero covalent | 0.30 | Si, Ge, C |
| `hydrogenated_network` | Cordero covalent (host), H 0.90 Å | the H-free host's (0.28–0.32) | a-Si:H, a-Ge:H, a-C:H, a-SiC:H, a-SiGe:H |
| `chalcogenide` | Cordero covalent | 0.30 | ZnS, CdTe, GeTe |
| `chalcogenide_glass` | Cordero covalent | 0.23 | GeS2, GeSe2, As2S3, As2Se3 (network glasses; tellurides stay `chalcogenide`) |
| `elemental_semiconductor` | Cordero covalent | 0.28 | a-Se, a-Te, a-As, a-Sb, a-P |

The dioxide split (rutile vs fluorite) and the small- vs large-cation nitride
split are decided by a cation-radius rule; `high_valent_oxide` is gated on
oxidation state ≥ 5. Compositions that match no specific class fall back to a
generic packing factor of 0.52 (Shannon ionic).

The density estimate sets the starting cell. Subsequent cell relaxation can
change it according to the chosen potential; validate the relaxed density
separately. Use `--target-density` when an experimental value is known, and
`--retry-mode reduce-minsep` or `none` when placement must retain that density.

### Coordination-aware placement ("SC")

When coordination targets are available, AmorphGen uses **SC ("Seed-Coordinate")** placement: each new atom is added as a bonded neighbour of an existing *under-coordinated* site (the **seed**) by placing it within that seed's bonding shell (`minsep ≤ d ≤ dmax`), which **coordinates** it. Over-coordinating a neighbour is rejected. This builds short-range order directly into the placement instead of relying on relaxation alone, giving more physical structures.

SC is the placement half of the **Seed-Coordinate-Anneal (SCA)** algorithm of Youn et al., *Comput. Mater. Sci.* (2014). AmorphGen factors the *Anneal* step out into its own stages (the MLIP geometry optimisation (`--relax`) and the melt-quench / hybrid workflow) so the placement step is just **Seed-Coordinate** (hence "SC", not "SCA").

Targets are inferred from composition by default; `--target-cn` overrides them.
The targets guide placement and cap over-coordination but do not guarantee that
every atom reaches its target. Disable SC with `--no-sc` to fall back to plain
random rejection sampling (or pass `target_cn={}` to the Python API).

### Transparency: the auto-derive log line

Every `--random-gen` run writes a single one-line summary at the top of `random_gen.log` capturing every chemistry-informed decision the auto chain made, so you can see *why* a particular minsep / density / target CN was used without reading the code. Example for Ga₂O₃:

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

The line is grep-friendly: `grep "auto-derive" random_gen.log` retrieves it as a single line per generation run. Bond classes shown are `ionic`, `covalent`, `metallic`, `cation-cation` (a nonmetal cation and another cation, see "Nonmetal cations" above), and `anion-pack` (same-element anion pairs use a separate anion-packing scale factor, see "Bond-type classifier" above).

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

# Fixed-density study: hold the cell EXACTLY at the target density and
# soften non-bonded minseps on placement stalls instead of expanding
amorphgen --random-gen --composition "SiO2*16" --n-structures 10 \
    --target-density 2.6 --retry-mode reduce-minsep
```

## Placement-stall policy (`--retry-mode`)

Random sequential placement can stall when the requested density and minimum
separations leave too little space for the remaining atoms.

The first response to a stall, in the default `expand` mode, is a soft pack:
the structure is placed again in the same cell with every floor scaled by
0.72, then every pair closer than its floor is pushed apart iteratively until
all pairs reach at least 98.5 percent of their floors. A successful soft pack
keeps the current cell and sets `info["soft_pack"] = True`. If soft packing
fails, `expand` grows the cell. The other modes skip soft packing and follow
their own retry policies:

| Mode | Cell | Minseps | Use when |
|---|---|---|---|
| `expand` (default) | cell edge grows 5% per retry (up to 4 retries) | requested floors; soft packing accepts 98.5% of them | the starting density may change during placement |
| `reduce-minsep` | held fixed | same-element separations reduced 5% per retry (up to 4 retries, about 19% total); unlike-element cation–anion floors retained | density must stay fixed during placement |
| `none` | held fixed | requested floors retained | both cell and minimum separations must stay fixed; a stall raises an error |

Same-element separations may represent bonds in elemental networks or alloys,
so inspect the resulting contacts when using `reduce-minsep`. Run relaxation
at fixed cell (`--cell-filter none`) to preserve the density through that step.

The table describes retries inside `generate_random`. `batch_random` also
resamples seeds and, after repeated failures, can reduce the auto-estimated
density and then same-element metal/metalloid separations. Explicit density
or cell settings disable the batch density reductions, but `expand` can
still grow the cell inside `generate_random`. With `reduce-minsep`, the batch
never changes density; with `none`, it only resamples seeds before skipping
unplaceable indices. Inspect `random_gen.log` for adjustments and skipped
structures.

## Python API

```python
from amorphgen import generate_random, batch_random

# Single structure (auto minsep and material-class density estimate)
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
ms = default_minsep(["In"] * 16 + ["O"] * 24, target_cn={"In": 4})
# Pass the full symbol list so the rules see the actual composition.
print(ms)
```

See {doc}`/api/random-gen` for the full API reference.


## Selecting structure indices

When using `--resume`, keep the composition, seed, density and placement
settings, output format, and relaxation settings the same. Schema 2 of
`run_metadata.json` records those settings, including model identity and local
model file hashes. AmorphGen refuses changed settings before modifying existing
outputs or logs, and checks the elements and atom counts in saved structures.
Reusing a random-generation directory with incompatible settings is also refused
without `--resume`, so a partial rerun cannot relabel older structures.
Missing, unreadable or older metadata without complete settings causes an
error; restore compatible metadata or use a separate output directory.
You can increase the requested structure count or change the selected indices
while retaining the same settings. Use `--batch-opt` to relax structures from
an earlier generation run.

An exclusive `.amorphgen.lock` covers generation and optional CLI relaxation.
A concurrent writer fails immediately. The OS releases the lock after normal
exit, failure or process termination; leave the persistent lock file in place.

`--indices SPEC` restricts a run to given structure indices; inclusive ranges
and lists both work (`80-90`, `0,5,7-9`). Every index has a seed derived from
`--seed`. With the same placement settings, split jobs reproduce the placements
from a full run unless batch-level retry escalation changes those settings.
Escalation state is not saved across resumes, so a resumed or split run that
hits that path is not guaranteed to match a continuous run.

Give concurrent jobs separate output directories to keep their logs and
metadata separate, then collect their non-overlapping structure files:

```bash
amorphgen --random-gen --composition "InGaZnO4*50" -n 100 --seed 2026 --indices 0-49  -o igzo_a
amorphgen --random-gen --composition "InGaZnO4*50" -n 100 --seed 2026 --indices 50-99 -o igzo_b

mkdir -p igzo/random_initial
cp igzo_a/random_initial/*.xyz igzo_b/random_initial/*.xyz igzo/random_initial/

# relax only some of them later (files whose name ends in the index)
amorphgen --batch-opt --input-dir igzo/random_initial --indices 80-90 -m mace-mpa-0 -o igzo/random_opt
```

`--batch-opt` also takes `--pattern GLOB` to choose the input files. Both
combine with `--resume` and `--engine torchsim`.
