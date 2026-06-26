"""
amorphgen.utils.radii
----------------------
Atomic radii data, bonding classification, minsep calculation,
and density estimation for random structure generation.

This module contains:
- Shannon ionic radii (CN=4 and CN=6)
- Metallic radii (Goldschmidt CN=12)
- Pauling electronegativities
- Element classification (nonmetal, metalloid)
- Elemental solid densities
- Bond type classification
- Minsep calculation from radii
- Cell volume / density estimation
"""

from __future__ import annotations

import itertools
import logging
from functools import lru_cache

import numpy as np
from ase.data import covalent_radii, atomic_numbers, atomic_masses

logger = logging.getLogger(__name__)


# ==============================================================================
# Radii data tables
# ==============================================================================

# Shannon effective ionic radii (A).
# Keyed as {symbol: {oxidation_state: {CN: radius}}}.
# CN=6 is the default; CN=4 used when target_cn is specified.
# Source: Shannon, R.D. Acta Cryst. A32, 751-767 (1976).
SHANNON_IONIC_RADII = {
    # Cations
    "Li": {1: {4: 0.59, 6: 0.76}},
    "Na": {1: {4: 0.99, 6: 1.02}},
    "K":  {1: {6: 1.38}},
    "Rb": {1: {6: 1.52}},
    "Cs": {1: {6: 1.67}},
    "Mg": {2: {4: 0.57, 6: 0.72}},
    "Ca": {2: {6: 1.00}},
    "Sr": {2: {6: 1.18}},
    "Ba": {2: {6: 1.35}},
    "B":  {3: {4: 0.11, 6: 0.27}},
    "Al": {3: {4: 0.39, 6: 0.535}},
    "Ga": {3: {4: 0.47, 6: 0.620}},
    "In": {3: {4: 0.62, 6: 0.800}},
    "Tl": {3: {6: 0.885}},
    "Si": {4: {4: 0.26, 6: 0.400}},
    "Ge": {4: {4: 0.39, 6: 0.530}},
    "Sn": {2: {6: 0.930}, 4: {4: 0.55, 6: 0.690}},
    "Pb": {2: {6: 1.190}, 4: {6: 0.775}},
    "Ti": {3: {6: 0.670}, 4: {4: 0.42, 6: 0.605}},
    "Zr": {4: {4: 0.59, 6: 0.720}},
    "Hf": {4: {4: 0.58, 6: 0.710}},
    "V":  {3: {6: 0.640}, 4: {6: 0.580}, 5: {4: 0.355, 6: 0.540}},
    "Nb": {4: {6: 0.680}, 5: {4: 0.48, 6: 0.640}},
    "Ta": {5: {6: 0.640}},
    "Cr": {3: {6: 0.615}, 4: {6: 0.550}, 6: {4: 0.26, 6: 0.440}},
    "Mo": {4: {6: 0.650}, 6: {4: 0.41, 6: 0.590}},
    "W":  {4: {6: 0.660}, 6: {4: 0.42, 6: 0.600}},
    "Ir": {3: {6: 0.680}, 4: {6: 0.625}, 5: {6: 0.570}},
    "Ru": {3: {6: 0.680}, 4: {6: 0.620}, 5: {6: 0.565}},
    "Rh": {3: {6: 0.665}, 4: {6: 0.600}},
    "Pd": {2: {4: 0.64, 6: 0.860}, 4: {6: 0.615}},
    "Ag": {1: {6: 1.150}, 2: {6: 0.940}, 3: {6: 0.750}},
    "Re": {4: {6: 0.630}, 6: {6: 0.550}, 7: {6: 0.530}},
    "Os": {4: {6: 0.630}, 6: {6: 0.545}, 7: {6: 0.525}, 8: {6: 0.390}},
    "Pt": {2: {4: 0.60, 6: 0.800}, 4: {6: 0.625}},
    "Au": {3: {6: 0.850}, 5: {6: 0.570}},
    "Hg": {2: {4: 0.96, 6: 1.020}},
    "Mn": {2: {4: 0.66, 6: 0.830}, 3: {6: 0.645}, 4: {4: 0.39, 6: 0.530}},
    "Fe": {2: {4: 0.63, 6: 0.780}, 3: {4: 0.49, 6: 0.645}},
    "Co": {2: {4: 0.58, 6: 0.745}, 3: {6: 0.610}},
    "Ni": {2: {4: 0.55, 6: 0.690}},
    "Cu": {1: {4: 0.60, 6: 0.770}, 2: {4: 0.57, 6: 0.730}},
    "Zn": {2: {4: 0.60, 6: 0.740}},
    "Cd": {2: {4: 0.78, 6: 0.950}},
    "Y":  {3: {6: 0.900}},
    "La": {3: {6: 1.032}},
    "Ce": {3: {6: 1.010}, 4: {6: 0.870}},
    # Lanthanides (Ln3+ CN6; plus Eu2+/Yb2+ and Pr4+/Tb4+ where relevant)
    "Pr": {3: {6: 0.990}, 4: {6: 0.850}},
    "Nd": {3: {6: 0.983}},
    "Pm": {3: {6: 0.970}},
    "Sm": {3: {6: 0.958}},
    "Eu": {2: {6: 1.170}, 3: {6: 0.947}},
    "Gd": {3: {6: 0.938}},
    "Tb": {3: {6: 0.923}, 4: {6: 0.760}},
    "Dy": {3: {6: 0.912}},
    "Ho": {3: {6: 0.901}},
    "Er": {3: {6: 0.890}},
    "Tm": {3: {6: 0.880}},
    "Yb": {2: {6: 1.020}, 3: {6: 0.868}},
    "Lu": {3: {6: 0.861}},
    "Bi": {3: {6: 1.030}},
    "Sc": {3: {6: 0.745}},
    # Actinide dioxide formers (An4+, Shannon CN6/CN8). Without these, ThO2 /
    # UO2 / PuO2 had no tabulated 4+ radius, so (a) the MO2 dioxide test could
    # not fire and they fell to metal_oxide, and (b) the radius fell back to the
    # oversized Cordero value (Th 2.06 A) — together under-predicting their
    # density by ~60%. With these entries they route to fluorite_dioxide (large
    # 4+ cation) and use the proper ionic radius (ThO2  rho -> ~10.4 vs 10.0).
    "Th": {4: {6: 0.940, 8: 1.050}},
    "U":  {4: {6: 0.890, 8: 1.000}},
    "Pu": {4: {6: 0.860, 8: 0.960}},
    # Metalloid in high oxidation state oxides (e.g. As2O5).
    # As(3+) is not tabulated by Shannon and Sb(III/V) Shannon radii
    # are too small (lone-pair distortion), giving unphysical sphere
    # volumes; for Sb compounds AmorphGen falls back to the Cordero
    # covalent radius which matches experimental Sb2O3 density better.
    "As": {5: {4: 0.335, 6: 0.460}},
    # Anions (CN-independent)
    "O":  {-2: {6: 1.400}},
    "S":  {-2: {6: 1.840}},
    "Se": {-2: {6: 1.980}},
    "Te": {-2: {6: 2.210}, 4: {6: 0.970}, 6: {6: 0.560}},
    "F":  {-1: {6: 1.330}},
    "Cl": {-1: {6: 1.810}},
    "Br": {-1: {6: 1.960}},
    "I":  {-1: {6: 2.200}},
    "N":  {-3: {6: 1.460}},
    "H":  {-1: {6: 1.400}},
    "C":  {-4: {6: 1.400}},  # Approximate; Shannon does not list C4-
    "P":  {-3: {6: 2.120}, 5: {4: 0.17, 6: 0.380}},
}

# Metallic radii (A, CN=12 Goldschmidt radii).
# Source: Greenwood & Earnshaw, Chemistry of the Elements (1997).
METALLIC_RADII = {
    "Li": 1.52, "Na": 1.86, "K": 2.27, "Rb": 2.48, "Cs": 2.65,
    "Mg": 1.60, "Ca": 1.97, "Sr": 2.15, "Ba": 2.17,
    "Al": 1.43, "Ga": 1.35, "In": 1.67,
    "Tl": 1.70, "Sn": 1.58, "Pb": 1.75, "Ti": 1.47, "Zr": 1.60,
    "Hf": 1.59, "V": 1.34, "Nb": 1.46, "Ta": 1.46, "Cr": 1.28,
    "Mo": 1.39, "W": 1.39, "Mn": 1.27, "Fe": 1.26, "Co": 1.25,
    "Ni": 1.24, "Cu": 1.28, "Zn": 1.37, "Cd": 1.52, "Y": 1.80,
    "La": 1.87, "Ce": 1.83, "Sc": 1.64, "Bi": 1.56, "Ge": 1.37,
    "Si": 1.17, "Ir": 1.36,
    "Ru": 1.34, "Rh": 1.34, "Pd": 1.37, "Ag": 1.44, "Re": 1.37,
    "Os": 1.35, "Pt": 1.39, "Au": 1.44, "Hg": 1.51,
    # Lanthanides (CN=12)
    "Pr": 1.82, "Nd": 1.82, "Pm": 1.81, "Sm": 1.80, "Eu": 2.04,
    "Gd": 1.80, "Tb": 1.78, "Dy": 1.77, "Ho": 1.77, "Er": 1.76,
    "Tm": 1.75, "Yb": 1.94, "Lu": 1.72,
    # Metalloids (rhombohedral / metallic phase radii)
    "As": 1.39, "Sb": 1.59,
}

# Pauling electronegativities for bonding classification.
# Values are the Allred (1961) revised scale, which retains Pauling's
# original methodology but uses updated thermochemical bond-dissociation
# data.  These are the values that appear in modern chemistry textbooks
# and the CRC Handbook.
#   Allred, A. L.  "Electronegativity values from thermochemical data."
#                  J. Inorg. Nucl. Chem. 17, 215-221 (1961).
#   Originally:  Pauling, L.  The Nature of the Chemical Bond, 3rd ed.,
#                Cornell Univ. Press (1960).
PAULING_EN = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63,
    "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91,
    "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18,
    "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22,
    "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Cd": 1.69, "In": 1.78,
    "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Cs": 0.79,
    "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Hf": 1.30, "Ta": 1.50,
    "W": 2.36, "Pb": 2.33, "Bi": 2.02, "Tl": 1.62, "Ir": 2.20,
    "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Re": 1.90,
    "Os": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00,
    "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.20,
    "Gd": 1.20, "Tb": 1.10, "Dy": 1.22, "Ho": 1.23, "Er": 1.24,
    "Tm": 1.25, "Yb": 1.10, "Lu": 1.27,
}

# Conventional anion charges, used to infer the cation oxidation state
# from a compound's stoichiometry by charge balance (e.g. Ti=+4 in
# TiO_2, Sb=+5 in Sb_2O_5). Limited to clear-cut anion-formers; complex
# / mixed-anion / non-stoichiometric systems fall through to the
# default "highest positive Shannon state" fallback.
ANION_CHARGES = {
    "O":  -2,
    "S":  -2,
    "Se": -2,
    "Te": -2,
    "F":  -1,
    "Cl": -1,
    "Br": -1,
    "I":  -1,
    "N":  -3,
    "H":  -1,  # treated as hydride; only valid for binary hydrides
}


# Cations that, although Shannon tabulates several oxidation states, take one
# strongly dominant state in oxide / mixed-anion solids. Used ONLY to break a
# tie when more than one charge-balanced assignment exists (never to force an
# unbalanced one). Kept deliberately conservative: d0 / main-group cases with
# little real ambiguity in compounds. Genuinely variable cations (Fe, Mn, Co,
# Cu, Ce, Ag, ...) are omitted, so a compound with two such cations stays
# ambiguous and resolves to None (honest) rather than guessing.
_DOMINANT_OS = {
    "Ti": 4, "Zr": 4, "Hf": 4, "Nb": 5, "Ta": 5, "V": 5,
    "W": 6, "Mo": 6, "Cr": 3, "Sn": 4, "Pb": 2,
    "Al": 3, "Ga": 3, "In": 3, "Sc": 3, "Y": 3,
}


@lru_cache(maxsize=512)
def _solve_oxidation_states(comp_items: tuple) -> dict | None:
    """Jointly assign integer oxidation states to every cation by charge
    balance. Returns a {cation: state} dict, or None if the assignment is not
    uniquely determined.

    Works for arbitrary cation/anion mixtures: the anion charge is summed over
    *all* anion-formers (so oxynitrides, oxyfluorides, oxysulfides balance the
    same way as simple oxides). Cation states are enumerated from the Shannon
    table; at most one cation may be absent from Shannon (it becomes the single
    free variable solved by the remaining balance). When several assignments
    balance, the one with the fewest :data:`_DOMINANT_OS` violations wins; if
    that is still tied the system is genuinely ambiguous and None is returned.
    """
    composition = dict(comp_items)
    total_neg = sum(n * ANION_CHARGES[s] for s, n in composition.items()
                    if s in ANION_CHARGES)
    if total_neg == 0:
        return None
    cation_charge = -total_neg  # total positive charge the cations must supply
    cations = {s: n for s, n in composition.items() if s not in ANION_CHARGES}
    if not cations:
        return None

    known, unknown = {}, []
    for s in cations:
        states = sorted(k for k in SHANNON_IONIC_RADII.get(s, {}) if k > 0)
        if states:
            known[s] = states
        else:
            unknown.append(s)
    if len(unknown) > 1:
        return None  # two un-tabulated cations -> underdetermined

    known_syms = list(known)
    sizes = 1
    for s in known_syms:
        sizes *= len(known[s])
    if sizes > 20000:
        return None  # guard against pathological enumeration

    solutions = []
    for combo in itertools.product(*(known[s] for s in known_syms)):
        assign = dict(zip(known_syms, combo))
        known_pos = sum(composition[s] * o for s, o in assign.items())
        if unknown:
            (u,) = unknown
            rem = cation_charge - known_pos
            if rem <= 0 or rem % composition[u] != 0:
                continue
            assign[u] = rem // composition[u]
        elif known_pos != cation_charge:
            continue
        solutions.append(assign)

    if not solutions:
        return None
    if len(solutions) == 1:
        return solutions[0]

    def penalty(sol):
        viol = sum(1 for s, o in sol.items()
                   if s in _DOMINANT_OS and o != _DOMINANT_OS[s])
        return (viol, sum(sol.values()))

    solutions.sort(key=penalty)
    if penalty(solutions[0]) == penalty(solutions[1]):
        return None  # still ambiguous after the dominant-state tie-break
    return solutions[0]


def infer_oxidation_state(sym: str, composition: dict) -> int | None:
    """Infer the oxidation state of *sym* in *composition* by charge balance.

    Sums anion charge from :data:`ANION_CHARGES` and assigns the
    remaining positive charge equally across cations of the queried
    element when only one cation type is present, or for any element
    that appears as the only non-anion type. Returns ``None`` when the
    inference is ambiguous (multiple cation species, non-integer
    oxidation state, or no anions at all), letting the caller fall back
    to a default Shannon entry.
    """
    if sym in ANION_CHARGES or sym in NONMETALS:
        return None  # not a cation we resolve here

    total_neg = 0
    for s, n in composition.items():
        q = ANION_CHARGES.get(s)
        if q is not None and s != sym:
            total_neg += n * q
    if total_neg == 0:
        return None  # no anions -> can't balance

    # Identify cations (everything that isn't a known anion-former).
    cations = {s: n for s, n in composition.items() if s not in ANION_CHARGES}
    if sym not in cations:
        return None

    # Single-cation case: balance directly.
    if len(cations) == 1:
        n_cat = cations[sym]
        ox = -total_neg / n_cat
        if abs(ox - round(ox)) < 1e-6 and ox > 0:
            return int(round(ox))
        return None

    # Multi-cation case: solve the whole system jointly. This resolves
    # multivalent cations whose state is fixed once the other cations are
    # pinned (FeTiO3 -> Fe2+/Ti4+, LiNbO3 -> Li+/Nb5+, SrTiO3 -> Sr2+/Ti4+),
    # and handles mixed anions (oxynitrides, oxyfluorides) since the balance
    # sums over every anion-former. Returns None when genuinely ambiguous.
    solution = _solve_oxidation_states(tuple(sorted(composition.items())))
    if solution is None:
        return None
    return solution.get(sym)


# ==============================================================================
# Element classification
# ==============================================================================
#
# References for the three element-type sets used by ``classify_bond``:
#
#   IUPAC, "Nomenclature of Inorganic Chemistry — IUPAC Recommendations
#       2005" (Red Book), RSC Publishing, ISBN 0-85404-438-8.  Section
#       IR-3 — defines the standard nonmetal / metalloid / metal columns.
#
# AmorphGen follows IUPAC's classification exactly.  The METALS set is the
# implicit complement of NONMETALS ∪ METALLOIDS — no separate list is
# maintained, so any element outside those two sets (transition metals,
# lanthanides, actinides, ...) is treated as metallic.
#
# Notes on borderline elements (intentionally NOT in METALLOIDS):
#
#   Sn  — IUPAC post-transition metal.  β-Sn (the room-T phase) is
#         metallic.  Amorphous Sn is metallic with liquid-like CN ≈ 6–8
#         and superconducting Tc ≈ 5–7 K, higher than crystalline β-Sn
#         (Bückel & Hilsch, Z. Phys. 138, 109 (1954); Bergmann, Phys.
#         Rep. 27, 159 (1976)).  α-Sn (diamond cubic, semimetal) only
#         exists below 13 °C and is not a glass-former, so the metallic
#         classification gives the correct behaviour for every case
#         AmorphGen would generate (SnO2 rutile CN=6, β-Sn alloys,
#         a-Sn).
#   Se  — IUPAC nonmetal.  Although a-Se forms covalent chalcogenide
#         chains, the Pauling-Δχ override in classify_bond already
#         routes Ge–Se, As–Se etc. to "covalent" without requiring
#         metalloid status; the chalcogenide compound-class rule
#         likewise fires on the presence of S/Se/Te without checking
#         metalloid membership.  Promoting Se to metalloid would be
#         redundant.

# Standard nonmetals + noble gases.  IUPAC Red Book (2005), Section IR-3.
# Se is included here — although it sits on the chalcogen/metalloid border
# and some textbooks promote it to metalloid, IUPAC's recommendation is
# "nonmetal".  Se-Se covalency in trigonal Se and a-Se is handled by the
# nonmetal-nonmetal rule in classify_bond (returns "covalent").
NONMETALS = frozenset({
    "H", "He", "C", "N", "O", "F", "Ne", "P", "S", "Cl", "Ar",
    "Se", "Br", "Kr", "I", "Xe", "At", "Rn",
})

# IUPAC's six recognised metalloids — B, Si, Ge, As, Sb, Te.
# (IUPAC Red Book 2005, Section IR-3.)
METALLOIDS = frozenset({"B", "Si", "Ge", "As", "Sb", "Te"})

# Typical coordination numbers for coordination-aware placement.
# Metalloids are always CN=4. For metals, CN depends on the anion context:
# oxides → CN=4-6, nitrides → CN=4, halides/sulfides → CN=6.
_ALWAYS_TETRAHEDRAL = frozenset({
    "Si", "Ge", "B", "P", "As",      # metalloids — always CN=4
    "Be",                              # tetrahedral in BeF2, BeO (like Si in SiO2)
    # Group-12 post-transition metals form zincblende-structured II-VI
    # semiconductors with chalcogens (ZnS, ZnSe, ZnTe, CdS, CdSe, CdTe,
    # HgS, HgSe, HgTe — all CN=4 in their stable phases).  Without this,
    # the chalcogenide branch of auto_target_cn defaults them to octahedral
    # CN=6, which is wrong for these specific compounds.
    "Zn", "Cd", "Hg",
})

# d-block transition metals that form *interstitial* carbides (rocksalt or
# close-related dense structures: TiC, ZrC, HfC, VC, NbC, TaC, CrC, MoC,
# WC, ...).  These compounds want metallic + Cordero radii and a higher
# packing factor (≈ 0.60) than covalent carbides (SiC, B4C, P4C3, ...).
# Pauling Δχ does NOT cleanly separate these from covalent carbides
# because W–C (Δχ = 0.19) and Mo–C (Δχ = 0.39) are interstitial but have
# small Δχ — the discriminator has to be cation group, not electronegativity.
# Group-11 (Cu, Ag, Au) and Group-12 (Zn, Cd, Hg) do not form stable
# carbides and are excluded.
_TRANSITION_METAL_CARBIDE_CATIONS = frozenset({
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",         # 3d
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",         # 4d
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt",               # 5d
})

# Rutile-type dioxides (MO2 with a small, octahedrally-coordinated metal cation:
# TiO2, VO2, CrO2, MnO2, NbO2, SnO2, RuO2, IrO2, OsO2, PtO2, ...) pack far more
# densely than light oxides and get a higher packing factor (see
# PACKING_FACTORS["rutile_dioxide"]).  They are identified *geometrically* by the
# radius-ratio rule rather than an element list: a 4+ cation radius below
# ~0.70 A (r/r_O < ~0.5) gives octahedral rutile/CaCl2-type packing, while larger
# cations form 8-coordinate fluorite/baddeleyite oxides (ZrO2 0.72, HfO2 0.71,
# CeO2 0.87) that stay `metal_oxide`.  Metalloid dioxides (SiO2, GeO2) are
# excluded as covalent network formers; see _classify_compound.
# Elements that form diatomic / small-molecule gases rather than covalent
# solid networks — excluded from the pure-element semiconductor rule.
_DIATOMIC_NONMETALS = frozenset({"H", "N", "O", "F", "Cl", "Br", "I"})

_RUTILE_DIOXIDE_RMAX = 0.70      # A; Shannon 4+ CN6 cation-radius cutoff
# Cation-radius cutoff (A) separating small-cation nitrides (denser packing,
# placeable at higher pf) from large-cation nitrides (placement-limited by the
# cation-cation distance, kept at the safe default pf).
_SMALL_CATION_NITRIDE_RMAX = 0.70
# Covalent rutile formers whose 4+ radius sits above the cutoff (lone-pair /
# covalency), kept as explicit exceptions.
_RUTILE_DIOXIDE_EXCEPTIONS = frozenset({"Pb"})

# Cations whose oxide is a covalent tetrahedral network rather than an ionic
# solid. Beryllium is the textbook case: Be2+ is so small and polarizing (and
# not even tabulated by Shannon) that BeO is a covalent wurtzite network. The
# ionic sphere model under-predicts its density by ~50% (1.42 vs the measured
# amorphous 3.01 g/cm3); treating it with Cordero covalent radii for BOTH atoms
# + an open-network packing factor recovers ~2.96. Kept as an explicit element
# set (not a radius gate) so it cannot misfire on small-but-ionic cations such
# as Al3+. Distinct from "covalent_oxide" (SiO2/GeO2/B2O3), whose metalloid
# cations have usable small Shannon radii and stay on the ionic path.
_COVALENT_OXIDE_CATIONS = frozenset({"Be"})


def auto_target_cn(composition: dict) -> tuple[dict | None, int]:
    """
    Auto-detect target coordination numbers and tolerance from composition.

    Returns (target_cn, cn_tolerance) tuple.

    Rules:
    - Metalloids (Si, Ge, B): CN=4, tolerance=0 (strict tetrahedral)
    - Pure Si/Ge: CN=4, tolerance=0
    - Metals in nitrides: CN=4, tolerance=0 (wurtzite tetrahedral)
    - Metals in halides/sulfides: CN=6, tolerance=0 (strict octahedral)
    - Metals in oxides: CN=5, tolerance=1 (flexible 4-6 range)

    Parameters
    ----------
    composition : dict
        e.g. {"In": 32, "O": 48}

    Returns
    -------
    (target_cn, cn_tolerance) : tuple
        target_cn: dict or None
        cn_tolerance: int (0=strict, 1=flexible)
    """
    elems = set(composition.keys())
    anions = elems & NONMETALS
    cations = elems - anions

    # Check compound class first (handles semiconductors, chalcogenides)
    cls = _classify_compound(composition)

    if cls == "group_iv":
        # Pure group IV (Si, Ge, SiGe) — tetrahedral CN=4
        target_cn = {s: 4 for s in elems}
        return target_cn, 0

    if cls == "pnictide":
        # III-V compounds (GaAs, InP, InAs) — tetrahedral CN=4
        target_cn = {s: 4 for s in elems}
        return target_cn, 0

    if cls == "chalcogenide":
        # Chalcogenides (ZnS, Sb2Te3, CdTe, GeTe)
        chalcogens = elems & {"S", "Se", "Te"}
        target_cn = {}
        for s in elems:
            if s not in chalcogens:
                if s in _ALWAYS_TETRAHEDRAL or s in METALLOIDS:
                    target_cn[s] = 4
                else:
                    target_cn[s] = 6
        return (target_cn if target_cn else None), 0

    if cls in {"covalent_carbide", "transition_metal_carbide"}:
        # Covalent carbides (SiC, B4C) — metalloid CN=4
        # Transition-metal carbides (TiC, WC, ZrC) — transition-metal CN=6 (rocksalt)
        target_cn = {}
        for s in elems:
            if s != "C":
                if s in _ALWAYS_TETRAHEDRAL or s in METALLOIDS:
                    target_cn[s] = 4
                else:
                    target_cn[s] = 6
        return (target_cn if target_cn else None), 0

    if cls == "hydride":
        # Hydrides (LiH, MgH2) — metal CN=6
        target_cn = {}
        for s in elems:
            if s != "H":
                target_cn[s] = 6
        return (target_cn if target_cn else None), 0

    if cls == "boride":
        # Borides (TiB2, LaB6) — metal CN=6
        target_cn = {}
        for s in elems:
            if s != "B":
                target_cn[s] = 6
        return (target_cn if target_cn else None), 0

    if cls == "alloy":
        # Metal alloys / intermetallics — typical CN=8-12
        # Use CN=8 (BCC-like) with tolerance=2 for flexibility
        target_cn = {s: 8 for s in elems}
        return target_cn, 2

    # Dioxides / high-valent oxides — use the crystallographic cation
    # coordination (octahedral rutile, 8-coordinate fluorite) instead of the
    # generic oxide CN=5. The anion CN follows from stoichiometry (e.g. Ir=6 in
    # IrO2 gives O=3). Tolerance 1 allows the amorphous spread (e.g. Ir 5-6).
    if cls == "rutile_dioxide":
        return {s: 6 for s in cations}, 1   # octahedral MO2: TiO2, IrO2, RuO2...
    if cls == "fluorite_dioxide":
        return {s: 8 for s in cations}, 1   # 8-coordinate MO2: ZrO2, HfO2, CeO2
    if cls == "high_valent_oxide":
        return {s: 6 for s in cations}, 1   # octahedral M(V)/M(VI): Nb2O5, WO3...

    if not anions:
        # Pure metals — typical CN=12 (FCC/HCP) or 8 (BCC)
        target_cn = {s: 8 for s in elems}
        return target_cn, 2

    # Determine anion context
    chalcogens = elems & {"S", "Se", "Te"}
    pnictogens = elems & {"P", "As", "Sb"}
    has_nitrogen = "N" in anions
    has_oxide = "O" in anions
    has_halide = bool(anions & {"Cl", "Br", "I", "F"})
    has_chalcogen = bool(chalcogens)
    has_halide_sulfide = has_halide or has_chalcogen

    target_cn = {}

    if has_oxide and not has_nitrogen and not has_halide_sulfide:
        # Pure oxides: metals CN=5 (flexible 4-6), metalloids CN=4 (strict)
        has_metal_cation = any(s not in _ALWAYS_TETRAHEDRAL and s not in METALLOIDS for s in cations)
        for s in cations:
            if s in _ALWAYS_TETRAHEDRAL or s in METALLOIDS:
                target_cn[s] = 4
            else:
                target_cn[s] = 5
        # tolerance=1 only if metal cations present (flexible 4-6)
        # pure metalloid oxides (SiO2, GeO2, B2O3): tolerance=0 (strict CN=4)
        return target_cn, 1 if has_metal_cation else 0

    elif has_nitrogen and not has_oxide:
        # Nitrides: all CN=4 strict
        for s in cations:
            target_cn[s] = 4
        return target_cn, 0

    elif has_halide and not has_oxide:
        # Halides: CN=6 strict, metalloids CN=4
        for s in cations:
            if s in _ALWAYS_TETRAHEDRAL or s in METALLOIDS:
                target_cn[s] = 4
            else:
                target_cn[s] = 6
        return target_cn, 0

    # Chalcogenides already handled above via cls == "chalcogenide"

    else:
        # Mixed or other: metals CN=5 flexible, metalloids CN=4
        for s in cations:
            if s in _ALWAYS_TETRAHEDRAL or s in METALLOIDS:
                target_cn[s] = 4
            else:
                target_cn[s] = 5
        return target_cn, 1

# Elemental solid densities (kg/m3).
# Source: CRC Handbook of Chemistry and Physics, 104th edition (2023).
# Gaseous elements (O, N, Cl, etc.) are excluded.
# (ASE's ase.data.covalent_radii — used elsewhere in this module — is
# sourced from Cordero et al. 2008 for elements 1-96.)
ELEMENTAL_DENSITIES = {
    # Solid nonmetals / metalloids (kg/m3)
    "B": 2340, "C": 2267, "P": 1823, "S": 2070,
    "Se": 4809, "Te": 6240, "I": 4930,
    # Metals
    "Li": 534, "Be": 1850, "Na": 971, "Mg": 1738, "Al": 2700,
    "Si": 2330, "K": 862, "Ca": 1550, "Sc": 2985, "Ti": 4507,
    "V": 6110, "Cr": 7190, "Mn": 7440, "Fe": 7874, "Co": 8900,
    "Ni": 8908, "Cu": 8960, "Zn": 7134, "Ga": 5904, "Ge": 5323,
    "As": 5727, "Sr": 2640, "Y": 4472, "Zr": 6506,
    "Nb": 8570, "Mo": 10220, "Cd": 8650, "In": 7310, "Sn": 7287,
    "Sb": 6685, "Ba": 3510, "La": 6145, "Ce": 6770,
    "Hf": 13310, "Ta": 16654, "W": 19250, "Pb": 11340, "Bi": 9807,
    "Tl": 11850,
}


# ==============================================================================
# Scale factors
# ==============================================================================

# Applied to sum of radii to give minimum interatomic distance.
# Scientific basis:
#   - Shannon (1976): ionic radii from ~900 oxide/fluoride crystal structures
#   - AIRSS buildcell: uses 80-90% of equilibrium distances
SCALE_FACTORS = {
    "ionic":              0.80,  # M-X bonds (shorter -> better coordination)
    "covalent":           0.80,  # Directional bonds
    "metallic":           0.85,  # M-M (keep higher to prevent M-M clustering)
    "anion_packing":      0.80,  # Small anion packing (O-O, F-F, N-N)
    "anion_packing_large": 0.70,  # Large anion packing (Cl-Cl, Br-Br, I-I, S-S)
}

# Shannon ionic radius threshold for small vs large anions.
_LARGE_ANION_RADIUS_THRESHOLD = 1.5

# Maximum minsep caps (A). Prevents overly large distances
# that make random placement impossible.
_MAX_SAME_ELEMENT_MINSEP = 2.80   # M-M, X-X same-element
_MAX_IONIC_MINSEP = 3.00          # M-X ionic bonds
_MAX_ANION_MINSEP = 3.00          # X-X anion packing


# ==============================================================================
# Bonding classification
# ==============================================================================

def classify_bond(sym_a: str, sym_b: str) -> str:
    """
    Classify a pair of elements by bonding type.

    Uses an element-type classification rule (nonmetal / metalloid / metal
    membership) with a Pauling-electronegativity refinement: pairs that
    the type rules would call "ionic" but whose Pauling electronegativity
    difference is small (Δχ < 1.0) are reclassified as "covalent".
    This catches III–V semiconductors (GaAs Δχ=0.37, AlAs ~0.6) and related
    compounds where the chemistry is predominantly covalent and Shannon
    ionic radii give unrealistically small inter-atomic distances.

    Returns one of: "ionic", "covalent", "metallic".
    """
    a_is_nonmetal = sym_a in NONMETALS
    b_is_nonmetal = sym_b in NONMETALS
    a_is_metalloid = sym_a in METALLOIDS
    b_is_metalloid = sym_b in METALLOIDS

    # Helper: small Δχ ⇒ predominantly covalent regardless of type rules.
    def _is_covalent_by_en():
        en_a = PAULING_EN.get(sym_a)
        en_b = PAULING_EN.get(sym_b)
        if en_a is None or en_b is None:
            return False
        return abs(en_a - en_b) < 1.0

    if a_is_metalloid and b_is_metalloid:
        return "covalent"
    if (a_is_metalloid and b_is_nonmetal) or (b_is_metalloid and a_is_nonmetal):
        # Was unconditionally "ionic" — but BN, BC etc. are covalent.
        return "covalent" if _is_covalent_by_en() else "ionic"
    if a_is_metalloid or b_is_metalloid:
        # Metal + metalloid (e.g.\ Ga + As).  Was unconditionally "ionic" —
        # but III-V semiconductors are predominantly covalent.
        return "covalent" if _is_covalent_by_en() else "ionic"
    if a_is_nonmetal != b_is_nonmetal:
        # Metal + nonmetal (e.g.\ Ga + N).  Was unconditionally "ionic" —
        # check Δχ for borderline cases (GaN Δχ=1.23 stays ionic; carbides
        # of low-EN metals like Ti-C Δχ=1.0 may flip).
        return "covalent" if _is_covalent_by_en() else "ionic"
    if a_is_nonmetal and b_is_nonmetal:
        return "covalent"
    return "metallic"


# ==============================================================================
# Radius lookup functions
# ==============================================================================

def get_ionic_radius(sym: str, cn: int | None = None,
                     oxidation_state: int | None = None) -> float | None:
    """
    Get Shannon ionic radius for an element.

    Parameters
    ----------
    sym : str
        Element symbol.
    cn : int, optional
        Coordination number. If provided, use CN-specific radius.
        Defaults to CN=6.
    oxidation_state : int, optional
        Specific oxidation state to look up (e.g.\\ ``5`` for Sb in
        Sb_2O_5). If supplied and present in the Shannon table the
        matching radius is returned. If supplied but absent, falls
        through to the default selection.  When ``None`` (default),
        anions return their most-negative state and cations return
        their **highest** positive state - usually the right choice for
        binary oxides / halides / nitrides where the higher state
        dominates (TiO_2, V_2O_5, WO_3 etc.).

    Returns
    -------
    float or None
    """
    if sym not in SHANNON_IONIC_RADII:
        return None
    states = SHANNON_IONIC_RADII[sym]

    if oxidation_state is not None and oxidation_state in states:
        ox = oxidation_state
    elif oxidation_state is not None and oxidation_state > 0:
        # Requested cation state is missing from Shannon; pick the
        # closest available positive state (preferring same-or-higher).
        # This handles cases like ZrN, where charge balance suggests
        # Zr(III) but Shannon only lists Zr(IV); using Zr(IV) is closer
        # than dropping all the way to a covalent fallback.
        positive = sorted(k for k in states.keys() if k > 0)
        if not positive:
            return None
        ox = min(positive, key=lambda k: (abs(k - oxidation_state), -k))
    elif sym in NONMETALS:
        ox = min(states.keys())  # most-negative anion state
    else:
        positive = {k: v for k, v in states.items() if k > 0}
        if not positive:
            return None
        ox = max(positive.keys())  # highest cation state when unspecified

    cn_dict = states[ox]
    target_cn = cn if cn is not None else 6

    if target_cn in cn_dict:
        return cn_dict[target_cn]
    elif 6 in cn_dict:
        return cn_dict[6]
    else:
        return next(iter(cn_dict.values()))


def get_metallic_radius(sym: str) -> float | None:
    """Get metallic radius for an element. Returns None if unavailable."""
    return METALLIC_RADII.get(sym)


def get_effective_radius(sym: str) -> float:
    """
    Get the most appropriate radius for volume estimation.

    - Metals -> metallic radius
    - Anion-formers (O, S, Cl, ...) -> Shannon ionic radius
    - Metalloids (Si, Ge, ...) -> metallic radius if available
    - Fallback -> covalent radius
    """
    if sym not in NONMETALS:
        r = get_metallic_radius(sym)
        if r is not None:
            return r
    if sym in NONMETALS and sym not in {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}:
        r = get_ionic_radius(sym)
        if r is not None:
            return r
    return covalent_radii[atomic_numbers[sym]]


# ==============================================================================
# Minsep calculation
# ==============================================================================

def default_minsep(symbols: list[str], scale: float = 0.85,
                   target_cn: dict | None = None) -> dict:
    """
    Build a minsep dict for all element pairs.

    Uses bonding-type-aware radii. When target_cn is provided, uses
    CN-specific Shannon radii (e.g. CN=4 for Si in SiO2).

    Parameters
    ----------
    symbols : list of str
    scale : float
        Fallback scale factor (default 0.85).
    target_cn : dict, optional
        e.g. {"Si": 4, "O": 2}. Uses CN-specific radii when available.

    Returns
    -------
    dict
        Minimum separations keyed as "A-B" with A <= B alphabetically.
    """
    unique = sorted(set(symbols))
    minsep = {}
    cn_map = target_cn or {}

    def _cn_radius(sym):
        cn = cn_map.get(sym)
        return get_ionic_radius(sym, cn=cn)

    for i, s1 in enumerate(unique):
        for s2 in unique[i:]:
            key = f"{s1}-{s2}" if s1 <= s2 else f"{s2}-{s1}"

            bond_type = classify_bond(s1, s2)
            sf = SCALE_FACTORS.get(bond_type, scale)

            if bond_type == "ionic":
                r1 = _cn_radius(s1)
                r2 = _cn_radius(s2)
                if r1 is not None and r2 is not None:
                    d_ionic = min((r1 + r2) * sf, _MAX_IONIC_MINSEP)
                    minsep[key] = d_ionic
                    logger.info(
                        "  minsep %s = %.2f A  (ionic: Shannon %.3f + %.3f, "
                        "scale=%.2f)", key, minsep[key], r1, r2, sf
                    )
                    continue

            elif bond_type == "metallic":
                r1 = get_metallic_radius(s1)
                r2 = get_metallic_radius(s2)
                if r1 is not None and r2 is not None:
                    d_metallic = (r1 + r2) * sf

                    has_anion = any(s in NONMETALS for s in unique)
                    if has_anion:
                        ri1 = _cn_radius(s1)
                        ri2 = _cn_radius(s2)
                        anion_syms = [s for s in unique if s in NONMETALS]
                        r_anion = max(
                            (_cn_radius(a) or 0.0) for a in anion_syms
                        ) if anion_syms else None

                        # Geometric estimate for edge-sharing polyhedra:
                        # d(M-M) = sqrt(2) * d(M-X) * sf
                        # Uses the same scale factor as ionic/metallic (0.85).
                        if ri1 is not None and r_anion and r_anion > 0:
                            d_geom_1 = 2**0.5 * (ri1 + r_anion) * sf
                        else:
                            d_geom_1 = 0.0

                        if ri2 is not None and r_anion and r_anion > 0:
                            d_geom_2 = 2**0.5 * (ri2 + r_anion) * sf
                        else:
                            d_geom_2 = 0.0

                        d_geometric = (d_geom_1 + d_geom_2) / 2 if (
                            d_geom_1 > 0 and d_geom_2 > 0
                        ) else max(d_geom_1, d_geom_2)

                        d_mm = min(max(d_metallic, d_geometric),
                                   _MAX_SAME_ELEMENT_MINSEP)
                        minsep[key] = d_mm
                        source = "metallic" if d_metallic >= d_geometric else "geometric"
                        capped = " (capped)" if d_mm < max(d_metallic, d_geometric) else ""
                        logger.info(
                            "  minsep %s = %.2f A  (M-M ionic context, %s: "
                            "met=%.2f, geom=%.2f, scale=%.2f)%s",
                            key, minsep[key], source,
                            d_metallic, d_geometric, sf, capped
                        )
                    else:
                        minsep[key] = min(d_metallic, _MAX_SAME_ELEMENT_MINSEP)
                        logger.info(
                            "  minsep %s = %.2f A  (metallic: %.3f + %.3f, "
                            "scale=%.2f)", key, minsep[key], r1, r2, sf
                        )
                    continue

            elif bond_type == "covalent":
                a_is_metalloid = s1 in METALLOIDS
                b_is_metalloid = s2 in METALLOIDS

                if a_is_metalloid or b_is_metalloid:
                    # Use metallic radii if available (better for pure elements),
                    # fall back to covalent radii
                    r1 = get_metallic_radius(s1) or covalent_radii[atomic_numbers[s1]]
                    r2 = get_metallic_radius(s2) or covalent_radii[atomic_numbers[s2]]
                    d_covalent = (r1 + r2) * sf

                    # In anion context (oxide, nitride, etc.), metalloid-metalloid
                    # distances are much larger due to bridging anions (Si-O-Si).
                    # Use geometric estimate: sqrt(2) * d(X-anion)
                    has_anion = any(s in NONMETALS for s in unique)
                    if has_anion and s1 == s2:
                        ri = _cn_radius(s1)
                        anion_syms = [s for s in unique if s in NONMETALS]
                        r_anion = max(
                            (_cn_radius(a) or 0.0) for a in anion_syms
                        ) if anion_syms else None

                        if ri is not None and r_anion and r_anion > 0:
                            d_geometric = 2**0.5 * (ri + r_anion) * sf
                            d_xx = min(max(d_covalent, d_geometric),
                                       _MAX_SAME_ELEMENT_MINSEP)
                            minsep[key] = d_xx
                            source = "covalent" if d_covalent >= d_geometric else "geometric"
                            capped = " (capped)" if d_xx < max(d_covalent, d_geometric) else ""
                            logger.info(
                                "  minsep %s = %.2f A  (metalloid anion context, %s: "
                                "cov=%.2f, geom=%.2f, scale=%.2f)%s",
                                key, minsep[key], source,
                                d_covalent, d_geometric, sf, capped
                            )
                            continue

                    minsep[key] = d_covalent
                    logger.info(
                        "  minsep %s = %.2f A  (covalent/metalloid: "
                        "%.3f + %.3f, scale=%.2f)",
                        key, minsep[key], r1, r2, sf
                    )
                    continue

                # Pure single-element nonmetal composition (e.g.\ pure C,
                # pure S, pure P). This is a genuine covalent homo-element
                # bond, not an anion-anion contact in a compound, so use
                # Cordero covalent radii rather than the much larger
                # Shannon anion radius.  Heteroelement nonmetal pairs and
                # same-element pairs in compounds (O-O in TiO_2, Cl-Cl in
                # LiCl, ...) still go through anion-packing below.
                if s1 == s2 and len(unique) == 1:
                    r = covalent_radii[atomic_numbers[s1]]
                    d_cov_homo = 2 * r * sf
                    minsep[key] = min(d_cov_homo, _MAX_SAME_ELEMENT_MINSEP)
                    logger.info(
                        "  minsep %s = %.2f A  (covalent homo-element: "
                        "Cordero %.3f, scale=%.2f)",
                        key, minsep[key], r, sf
                    )
                    continue

                r1 = _cn_radius(s1)
                r2 = _cn_radius(s2)
                if r1 is not None and r2 is not None:
                    avg_r = (r1 + r2) / 2
                    if avg_r >= _LARGE_ANION_RADIUS_THRESHOLD:
                        sf_anion = SCALE_FACTORS.get("anion_packing_large", 0.70)
                        label = "large anion"
                    else:
                        sf_anion = SCALE_FACTORS.get("anion_packing", 0.80)
                        label = "small anion"
                    minsep[key] = min((r1 + r2) * sf_anion, _MAX_ANION_MINSEP)
                    logger.info(
                        "  minsep %s = %.2f A  (%s packing: Shannon "
                        "%.3f + %.3f, scale=%.2f)",
                        key, minsep[key], label, r1, r2, sf_anion
                    )
                    continue

            # Fallback: covalent radii (with cap)
            r1 = covalent_radii[atomic_numbers[s1]]
            r2 = covalent_radii[atomic_numbers[s2]]
            d_fallback = (r1 + r2) * scale
            if s1 == s2:
                d_fallback = min(d_fallback, _MAX_SAME_ELEMENT_MINSEP)
            else:
                d_fallback = min(d_fallback, _MAX_IONIC_MINSEP)
            minsep[key] = d_fallback
            logger.info(
                "  minsep %s = %.2f A  (fallback: covalent %.3f + %.3f, "
                "scale=%.2f)", key, minsep[key], r1, r2, scale
            )

    return minsep


# ==============================================================================
# Density / cell volume estimation
# ==============================================================================

def estimate_density(composition: dict, amorphous_factor: float = 0.80) -> float | None:
    """
    Estimate density from elemental solid densities.

    Returns density in g/cm3, or None if any element lacks density data.
    """
    total_mass = 0.0
    total_vol = 0.0
    for sym, count in composition.items():
        dens = ELEMENTAL_DENSITIES.get(sym)
        if dens is None or dens == 0:
            return None
        dens_gcm3 = dens / 1000.0
        mass = atomic_masses[atomic_numbers[sym]] * count
        total_mass += mass
        total_vol += mass / dens_gcm3

    if total_vol == 0:
        return None

    crystal_density = total_mass / total_vol
    return crystal_density * amorphous_factor


# Packing factors for density estimation by material class.
#
# The same formula V_cell = sum(sphere_vol) / f_pack is used for all
# classes, but the radius type underneath the spheres differs by
# bonding regime (see :func:`_radius_for_density`):
#   - Ionic (oxides, halides, nitrides, hydrides) -> Shannon ionic CN=6
#     - typical f_pack range 0.50-0.58
#   - Covalent (group-IV, pnictides, chalcogenides, borides, carbides)
#     -> Cordero covalent. Covalent radii reflect bond lengths, so
#     the sphere is much smaller than the actual atomic volume; f_pack
#     is correspondingly smaller (~0.20-0.40).
#   - Metallic (alloys) -> Goldschmidt metallic. Metallic radii are
#     close to atomic centres in close-packed metals; f_pack ~0.65-0.75.
#
# Values calibrated to give experimental amorphous densities for
# representative systems in each class.
PACKING_FACTORS = {
    # Ionic — Shannon ionic radii
    "covalent_oxide":  0.50,  # SiO2, GeO2, B2O3 (network formers)
    "metal_oxide":     0.52,  # In2O3, Al2O3, Ga2O3 (and fluorite ZrO2/HfO2/CeO2)
    "rutile_dioxide":  0.66,  # TiO2, SnO2, RuO2, IrO2, OsO2 — dense rutile-type MO2
    "fluorite_dioxide": 0.63,  # ZrO2, HfO2, CeO2 — 8/7-coordinate fluorite/baddeleyite MO2
    "high_valent_oxide": 0.60,  # V2O5, Nb2O5, Ta2O5, Sb2O5, MoO3, WO3 — cation OS>=5, dense
    "halide":          0.58,  # Li2ZrCl6, LiF
    "nitride":         0.52,  # large-cation nitrides (ZrN, HfN, ScN). Kept low:
                              # the cation-cation distance limits placement, so a
                              # denser cell fails to generate. Densities run low;
                              # use --target-density for these.
    "small_cation_nitride":   0.62,  # small-cation nitrides (AlN, GaN, Si3N4, TiN) —
                              # small cations fit the N sublattice, so they pack
                              # denser and still place. Cation-radius-gated.
    "hydride":         0.55,  # LiH, MgH2, NaAlH4
    # Covalent — Cordero covalent radii
    "group_iv":        0.30,  # Si, Ge, C (tetrahedral covalent network)
    "elemental_semiconductor": 0.28,  # a-Se, a-Te, a-As, a-Sb, a-P (chain/layer, Cordero)
    "pnictide":        0.32,  # GaAs, InP, InAs (III-V compounds)
    "chalcogenide":    0.30,  # ZnS, CdTe, GeTe (II-VI / IV-VI)
    "boride":          0.50,  # TiB2, MgB2, ZrB2
    "covalent_network_oxide": 0.35,  # BeO — covalent wurtzite oxide network;
                                     # Cordero radii for both atoms (Be 0.96,
                                     # O 0.66) + 0.35 -> rho ~2.96 vs measured
                                     # amorphous 3.01 g/cm3 (ionic model: 1.42)
    # Carbides split by cation chemistry (2026-05-11) — see
    # _TRANSITION_METAL_CARBIDE_CATIONS and _classify_compound for the routing.
    "covalent_carbide":     0.32,  # SiC, B4C — Cordero radii, open network
    "transition_metal_carbide": 0.60,  # TiC, WC, ZrC — rocksalt-like; uses
                                   # Goldschmidt metallic for cation,
                                   # Cordero for C (see _radius_for_density)
    # Metallic — Goldschmidt metallic radii.  Lowered from 0.70 to 0.60
    # (2026-05-10) so random sequential placement has enough headroom:
    # random close packing of equal hard spheres caps at ~0.64, and 0.70
    # left no room for the placement attempt budget — Cu/Au/Ti/Ni etc.
    # all failed single-call generate_random with auto density.  At 0.60
    # the cell is ~5% larger; predicted density for a-Cu is ~7.2 g/cm³
    # vs experimental amorphous ~7.7-8.5 (~10% low — within design-target
    # accuracy and easily overridden via target_density for production runs).
    "alloy":           0.60,  # NiTi, CuZr, brass, pure metals
    "default":         0.52,  # fallback (ionic)
}

# NOTE: ``AMORPHOUS_DENSITY_FACTORS`` and the associated elemental
# density-mixing branch were retired in favour of the unified
# class-aware sphere-packing model. ``estimate_density()`` (which
# implements the mixing rule) is kept as a public helper for users
# who want the mass-averaged crystal density directly.


def _classify_compound(composition: dict) -> str:
    """
    Classify a compound by material class for packing factor selection.

    Returns one of: "group_iv", "pnictide", "chalcogenide",
    "covalent_oxide", "covalent_network_oxide", "metal_oxide", "halide",
    "nitride", "carbide", "hydride", "boride", "alloy", "default".
    """
    elems = set(composition.keys())
    has_metal = any(s not in NONMETALS and s not in METALLOIDS for s in elems)
    has_metalloid = any(s in METALLOIDS for s in elems)

    anions = elems & NONMETALS
    chalcogens = elems & {"S", "Se", "Te"}
    pnictogens = elems & {"P", "As", "Sb"}

    # Group IV: pure Si, Ge, SiGe (all metalloid, no chalcogen/pnictogen)
    all_metalloid = all(s in METALLOIDS for s in elems)
    if all_metalloid and not chalcogens and not pnictogens:
        return "group_iv"

    # Pure single-element covalent / semimetal solids that group-IV doesn't
    # catch (a-Se, a-Te, a-As, a-Sb, a-P, a-S). A one-element system has no
    # electronegativity difference, so it bonds covalently — it must use Cordero
    # covalent radii, not the ionic "highest positive" fallback, which gives
    # nonsense (e.g. pure As → As(5+) radius → ~150 g/cm3). This is the bonding
    # analogue of the radius rule: ΔEN = 0 ⇒ covalent.
    if len(elems) == 1:
        (only_el,) = elems
        if ((only_el in NONMETALS or only_el in METALLOIDS)
                and only_el not in _DIATOMIC_NONMETALS):
            return "elemental_semiconductor"

    # Pnictides: III-V compounds (GaAs, InP, InAs, GaSb)
    # Must have at least one non-pnictogen element (the group III metal)
    # Pure pnictogens (Sb, As) are not III-V compounds
    # P is both NONMETAL and pnictogen — exclude from anion check here
    non_pnictogen = elems - pnictogens
    other_anions = anions - {"P", "As"}  # P/As can act as pnictogen anion
    if pnictogens and non_pnictogen and not other_anions and not chalcogens:
        return "pnictide"

    # Chalcogenides (S, Se, Te) — check before oxides
    if chalcogens and "O" not in anions:
        return "chalcogenide"

    # Oxides
    if "O" in anions:
        cations = elems - anions
        # Anomalous small-cation covalent oxides (BeO): the cation is too small
        # and polarizing for the ionic model, so the oxide is a covalent
        # tetrahedral network. Routed before the ionic branches; uses Cordero
        # radii (see _radius_for_density) + a low packing factor.
        if cations and all(c in _COVALENT_OXIDE_CATIONS for c in cations):
            return "covalent_network_oxide"
        # High-valent oxides (all cations in oxidation state >= 5): V2O5, Nb2O5,
        # Ta2O5, Sb2O5, MoO3, WO3, ... Small, highly-charged cations form short
        # M-O bonds and pack denser than the generic oxide factor predicts, so
        # they get a higher packing factor. Restricted to metal/metalloid
        # cations (excludes molecular P2O5 / SO3 etc.).
        if cations and all(c not in NONMETALS for c in cations):
            os_list = [infer_oxidation_state(c, composition) for c in cations]
            if os_list and all(o is not None and o >= 5 for o in os_list):
                return "high_valent_oxide"
        # Rutile-type dioxides (MO2) pack much denser than light oxides; route
        # them to a higher packing factor. Identified by the radius-ratio rule:
        # a metal MO2 whose 4+ cation radius is below the rutile/fluorite cutoff
        # (or a known covalent exception, e.g. PbO2). Fluorite dioxides
        # (ZrO2, HfO2, CeO2 — larger cations) and metalloid oxides (SiO2, GeO2)
        # fall through to metal_oxide / covalent_oxide.
        metal_cations = [c for c in cations
                         if c not in NONMETALS and c not in METALLOIDS]
        if cations and len(metal_cations) == len(cations):
            n_cat = sum(composition[c] for c in cations)
            if composition.get("O", 0) == 2 * n_cat:
                # MO2 dioxide: split rutile (small cation, octahedral) vs
                # fluorite/baddeleyite (large cation, 8/7-coordinate) by the 4+
                # cation radius, with PbO2 as a covalent rutile exception.
                r4 = {c: get_ionic_radius(c, cn=6, oxidation_state=4)
                      for c in cations}
                if all(c in _RUTILE_DIOXIDE_EXCEPTIONS or
                       (r4[c] is not None and r4[c] < _RUTILE_DIOXIDE_RMAX)
                       for c in cations):
                    return "rutile_dioxide"
                if all(r4[c] is not None and r4[c] >= _RUTILE_DIOXIDE_RMAX
                       for c in cations):
                    return "fluorite_dioxide"
        if has_metal:
            return "metal_oxide"
        elif has_metalloid:
            return "covalent_oxide"

    # Halides
    if anions & {"Cl", "Br", "I", "F"}:
        return "halide"

    # Nitrides — small-cation nitrides (AlN, GaN, Si3N4, TiN, ...) pack densely
    # and tolerate a higher packing factor; large-cation nitrides (ZrN, HfN,
    # ScN) are placement-limited by the cation-cation distance, so they keep the
    # safe default. Gate on cation radius (the same radius-rule idea as the
    # rutile/fluorite dioxide split).
    if "N" in anions:
        cations = elems - anions
        crs = [get_ionic_radius(c, cn=6,
                                oxidation_state=infer_oxidation_state(c, composition))
               for c in cations]
        if cations and all(cr is not None and cr < _SMALL_CATION_NITRIDE_RMAX
                           for cr in crs):
            return "small_cation_nitride"
        return "nitride"

    # Carbides — split by cation chemistry
    #   Transition-metal carbides (TiC, WC, ZrC, ...) form rocksalt-like dense
    #   structures with d-block transition metals and need a higher packing
    #   factor + Goldschmidt metallic radii for the cation.
    #   Covalent carbides (SiC, B4C, ...) keep Cordero radii + low packing
    #   factor.  See _TRANSITION_METAL_CARBIDE_CATIONS for the cation set.
    if "C" in anions:
        cations = elems - anions
        if cations & _TRANSITION_METAL_CARBIDE_CATIONS:
            return "transition_metal_carbide"
        return "covalent_carbide"

    # Hydrides
    if "H" in anions:
        return "hydride"

    # Borides (B as anion-like element with metals)
    if "B" in elems and has_metal and not anions and not chalcogens:
        return "boride"

    # Metal alloys / intermetallics (no anions)
    if not anions and not chalcogens and has_metal:
        return "alloy"

    return "default"


def _radius_for_density(sym: str, cls: str,
                        composition: dict | None = None) -> float:
    """Pick the radius type appropriate for the composition's bonding character.

    The density estimator uses sphere-packing on per-element spheres;
    different bonding regimes need different radius tables so that the
    sphere volume is a faithful proxy for the volume the atom occupies
    in the corresponding amorphous structure:

    * Ionic compounds (oxides, halides, nitrides, hydrides) -> Shannon
      ionic radii at CN=6 (cation-anion contact length).
    * Covalent compounds (group-IV, pnictides, chalcogenides, borides,
      carbides) -> Cordero covalent radii (bond-length-based).
    * Metallic alloys -> Goldschmidt metallic radii (close-packed
      atomic centres).
    """
    ionic_classes = {"covalent_oxide", "metal_oxide", "rutile_dioxide",
                     "fluorite_dioxide", "high_valent_oxide", "halide", "nitride",
                     "small_cation_nitride", "hydride"}
    covalent_classes = {"group_iv", "elemental_semiconductor", "pnictide",
                        "chalcogenide", "boride", "covalent_carbide",
                        "covalent_network_oxide"}
    # Infer oxidation state from charge balance when a composition is
    # supplied; falls back to "highest positive" inside get_ionic_radius
    # if inference returns None.
    ox = infer_oxidation_state(sym, composition) if composition else None
    # Antimony is oxidation-state split: Sb(V) is d0 and octahedral with no lone
    # pair, so it uses its small ionic radius like Nb(V)/Ta(V) (dense oxides such
    # as Sb2O5). Sb(III) has a stereochemically active lone pair giving open
    # structures (Sb2O3) where the small ionic radius badly over-predicts the
    # density, so it falls back to the Cordero covalent radius.
    if sym == "Sb":
        return 0.60 if ox == 5 else covalent_radii[atomic_numbers["Sb"]]
    if cls in ionic_classes:
        r = get_ionic_radius(sym, cn=6, oxidation_state=ox)
        if r is None:
            r = covalent_radii[atomic_numbers[sym]]
        return r
    if cls in covalent_classes:
        return covalent_radii[atomic_numbers[sym]]
    if cls == "transition_metal_carbide":
        # TiC, WC, ZrC etc. — cation uses Goldschmidt metallic, C uses
        # Cordero.  This pair of radii + pf=0.60 reproduces a-TiC density
        # ≈ 4.0 g/cm³ (vs crystal 4.93) and a-WC ≈ 14 g/cm³ (vs 15.6),
        # within the documented 5-20 % amorphous-vs-crystal range.
        if sym == "C":
            return covalent_radii[atomic_numbers[sym]]
        r = get_metallic_radius(sym)
        if r is None:
            r = covalent_radii[atomic_numbers[sym]]
        return r
    if cls == "alloy":
        r = get_metallic_radius(sym)
        if r is None:
            r = covalent_radii[atomic_numbers[sym]]
        return r
    # Default: ionic with covalent fallback
    r = get_ionic_radius(sym, cn=6, oxidation_state=ox)
    if r is None:
        r = covalent_radii[atomic_numbers[sym]]
    return r


def estimate_cell_length(composition: dict, target_density: float | None = None,
                         packing_factor: float | None = None,
                         density_scale: float = 1.0) -> float:
    """Estimate cubic cell length from composition via sphere packing.

    A single class-aware sphere-packing model handles all amorphous
    compositions:

    .. math::
       V_{\\text{cell}} = \\frac{\\sum_i N_i \\cdot \\tfrac{4}{3}\\pi r_i^3}
                              {f_{\\text{pack}} \\cdot s},

    where :math:`r_i` is the radius for element :math:`i` chosen
    according to the compound's bonding regime (Shannon ionic for ionic
    compounds, Cordero covalent for covalent compounds, Goldschmidt
    metallic for alloys; see :func:`_radius_for_density`).
    :math:`f_{\\text{pack}}` is the class-aware packing fraction stored
    in :data:`PACKING_FACTORS`, and :math:`s` is the user-supplied
    ``density_scale`` (default 1.0).

    If ``target_density`` is supplied, the cell is sized directly from
    it and ``density_scale`` / ``packing_factor`` are ignored.
    """
    if density_scale <= 0:
        raise ValueError(f"density_scale must be > 0 (got {density_scale}).")

    total_mass = sum(atomic_masses[atomic_numbers[s]] * n
                     for s, n in composition.items())

    if target_density is not None:
        vol_cm3 = (total_mass / 6.022e23) / target_density
        vol_A3 = vol_cm3 * 1e24
    else:
        cls = _classify_compound(composition)
        if packing_factor is None:
            packing_factor = PACKING_FACTORS.get(cls, PACKING_FACTORS["default"])

        total_sphere_vol = 0.0
        for sym, count in composition.items():
            r = _radius_for_density(sym, cls, composition=composition)
            total_sphere_vol += count * (4.0 / 3.0) * np.pi * r ** 3
        vol_A3 = total_sphere_vol / (packing_factor * density_scale)

    return vol_A3 ** (1.0 / 3.0)


# ==============================================================================
# Auto-derivation log line
# ==============================================================================

def format_auto_derive_summary(
    composition: dict,
    target_cn: dict | None,
    minsep: dict,
    est_density: float,
    cell_length: float,
) -> str:
    """Single-line summary of the auto-derivation chain for the random_gen log.

    Captures the chemistry-informed decisions AmorphGen makes from a bare
    composition: material class, oxidation states (where inferable), per-cation
    target coordination, per-pair bond classification + Pauling Δχ + minsep,
    and the final estimated density / cell length.  Designed to fit on one
    log line for typical 2–4 element systems (e.g. Ga₂O₃ ≈ 200 chars).

    Format:
        [auto-derive] <formula> → <class>, OS{...}, CN{...},
                       minsep{pair:val class [Δχ=val] | ...},
                       ρ=<val> g/cm³, L=<val> Å

    For ionic pairs (where the Pauling Δχ < 1.0 refinement fires) the Δχ
    value is shown so the user can see exactly why the pair was classified
    as ionic vs covalent.  For metallic and anion-packing pairs the Δχ is
    omitted (the type rule alone decides those).
    """
    # Composition string in the order the user provided
    formula = "".join(f"{s}{n}" for s, n in composition.items())

    # Material class
    cls = _classify_compound(composition)

    # Oxidation states — only for cations (positive OS, where inferable)
    os_parts = []
    for elem in composition:
        os_val = infer_oxidation_state(elem, composition)
        if os_val is not None and os_val > 0:
            os_parts.append(f"{elem}:+{os_val}")
    os_str = f"OS{{{', '.join(os_parts)}}}" if os_parts else ""

    # Target CN per element
    cn_str = ""
    if target_cn:
        cn_parts = [f"{e}:{cn}" for e, cn in target_cn.items()]
        cn_str = f"CN{{{', '.join(cn_parts)}}}"

    # Per-pair: minsep + bond class label (+ Δχ for ionic)
    pair_parts = []
    for pair in sorted(minsep):
        a, b = pair.split("-")
        d = minsep[pair]
        if a == b and a in NONMETALS:
            label = "anion-pack"
        else:
            pair_cls = classify_bond(a, b)
            if pair_cls == "ionic":
                dchi = abs(PAULING_EN.get(a, 0.0) - PAULING_EN.get(b, 0.0))
                label = f"ionic Δχ={dchi:.2f}"
            elif pair_cls == "metallic":
                label = "metallic"
            elif pair_cls == "covalent":
                label = "covalent"
            else:
                label = pair_cls
        pair_parts.append(f"{pair}:{d:.2f} {label}")
    minsep_str = f"minsep{{{' | '.join(pair_parts)}}}"

    # Assemble — comma-separated top-level fields
    parts = [f"[auto-derive] {formula} → {cls}"]
    if os_str:
        parts.append(os_str)
    if cn_str:
        parts.append(cn_str)
    parts.append(minsep_str)
    parts.append(f"ρ={est_density:.2f} g/cm³ L={cell_length:.2f} Å")

    return ", ".join(parts)
