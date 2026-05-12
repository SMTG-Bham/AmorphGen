"""
StructureAnalyser — main class that delegates to submodules.

Usage
-----
    from amorphgen.analysis import StructureAnalyser

    sa = StructureAnalyser("output_dir/", cutoff="auto")
    sa.summary()
    sa.plot(output_dir="plots/")
"""

from __future__ import annotations

import os
import glob
import numpy as np
from collections import defaultdict
from ase.io import read

from .cutoff import auto_cutoff_minsep, auto_cutoff_rdf
from .structure import (compute_density, compute_coordination,
                        compute_bond_distances, compute_all_angles,
                        compute_bond_angle_stats, build_neighbour_dict)
from .rdf import (compute_rdf, compute_structure_factor,
                  compute_averaged_rdf)
from .rings import compute_ring_statistics
from .voronoi import compute_voronoi
from .energy import compute_energy_ranking
from .plotting import plot_analysis


class StructureAnalyser:
    """
    Analyse amorphous structures: density, coordination, distances,
    angles, RDF, S(q), rings, Voronoi, energy ranking.

    Parameters
    ----------
    source : str or list of str
        Path to a structure file, a directory, or a list of file paths.
    cutoff : float, dict, or str
        - float: single cutoff (A) for all pairs
        - dict: pair-specific, e.g. {"Si-O": 2.2}
        - "auto": from bonding radii (fast)
        - "auto-rdf": from RDF first minimum (accurate)
    """

    def __init__(self, source, cutoff="auto"):
        """
        Parameters
        ----------
        source : str, list of str, or list of Atoms
            Path to a structure file, a directory of structure files,
            a list of file paths, or a list of ASE Atoms objects.
        cutoff : float, dict, or str
            Neighbour cutoff for CN, bond distances, and angles.
            - float: single cutoff (A) for all pairs
            - dict: pair-specific, e.g. {"Si-O": 2.2, "O-O": 2.8}
            - "auto": from bonding radii (fast, default)
            - "auto-rdf": from RDF first minimum (more accurate)
        """
        self.atoms_list = self._load(source)
        if not self.atoms_list:
            raise ValueError("No structures loaded. Check the source path.")

        # Warn if structures have different compositions
        formulas = set(a.get_chemical_formula(mode="hill")
                       for a in self.atoms_list)
        if len(formulas) > 1:
            import warnings
            warnings.warn(
                f"Structures have different compositions: {formulas}. "
                f"Analysis assumes uniform composition; energy/atom and "
                f"CN statistics may mix incompatible systems."
            )

        self._cutoff_mode = cutoff

        if cutoff == "auto":
            self.cutoff = auto_cutoff_minsep(self.atoms_list)
            self._cutoff_mode = "auto (minsep)"
        elif cutoff == "auto-rdf":
            self.cutoff = auto_cutoff_rdf(self.atoms_list)
            self._cutoff_mode = "auto (RDF)"
        else:
            self.cutoff = cutoff

        self._max_cutoff = (
            max(self.cutoff.values()) if isinstance(self.cutoff, dict)
            else self.cutoff
        )

    @staticmethod
    def _load(source):
        from ase import Atoms
        if isinstance(source, list):
            if source and isinstance(source[0], Atoms):
                return source
            return [read(f) for f in source]
        if os.path.isdir(source):
            files = sorted(
                glob.glob(os.path.join(source, "*.xyz"))
                + glob.glob(os.path.join(source, "*.extxyz"))
                + glob.glob(os.path.join(source, "*.cif"))
                + glob.glob(os.path.join(source, "*.vasp"))
            )
            if not files:
                raise FileNotFoundError(f"No structure files in {source}/")
            return [read(f) for f in files]
        return [read(source)]

    def _get_cutoff(self, s1, s2):
        if isinstance(self.cutoff, (int, float)):
            return self.cutoff
        key1 = f"{s1}-{s2}"
        key2 = f"{s2}-{s1}"
        return self.cutoff.get(key1, self.cutoff.get(key2, self._max_cutoff))

    def _build_neighbour_dict(self, atoms):
        return build_neighbour_dict(atoms, self._max_cutoff, self._get_cutoff)

    # ── Core analysis methods ────────────────────────────────────────────

    def density(self):
        """Compute density (g/cm3) for each structure.

        Returns
        -------
        dict
            {"values": list[float], "mean": float, "std": float}
        """
        return compute_density(self.atoms_list)

    def coordination(self, pair=None):
        """Compute coordination numbers across all structures.

        Parameters
        ----------
        pair : str, optional
            Pair to analyse, e.g. "Si-O". If None, all pairs computed.

        Returns
        -------
        dict
            Keyed by pair string. Each value is a dict with
            "mean", "std", "min", "max", "distribution", "total_atoms".
        """
        return compute_coordination(self.atoms_list, self._max_cutoff,
                                    self._get_cutoff, pair)

    def bond_distances(self, pair=None):
        """Compute bond distance statistics.

        Parameters
        ----------
        pair : str, optional
            Pair to analyse, e.g. "Si-O". If None, all pairs computed.

        Returns
        -------
        dict
            Keyed by pair string. Each value is a dict with
            "mean", "std", "min", "max", "count".
        """
        return compute_bond_distances(self.atoms_list, self._max_cutoff,
                                      self._get_cutoff, pair)

    def _compute_all_angles(self, triplet=None, bonding_only=True):
        return compute_all_angles(self.atoms_list, self._max_cutoff,
                                  self._get_cutoff, triplet, bonding_only)

    def bond_angles(self, triplet=None):
        """Compute bond angle statistics.

        Parameters
        ----------
        triplet : str, optional
            Triplet to analyse, e.g. "O-Si-O". If None, all triplets computed.

        Returns
        -------
        dict
            Keyed by triplet string. Each value is a dict with
            "mean", "std", "min", "max", "count" (angles in degrees).
        """
        raw = self._compute_all_angles(triplet)
        return compute_bond_angle_stats(raw)

    # ── RDF and S(q) ────────────────────────────────────────────────────

    def rdf(self, pair=None, rmax=None, nbins=200, sigma=0.0):
        """Compute the radial distribution function g(r).

        Parameters
        ----------
        pair : str, optional
            Pair to analyse, e.g. "Si-O". If None, total RDF.
        rmax : float, optional
            Maximum radius in A. Auto-detected from cell if None.
        nbins : int
            Number of histogram bins (default 200).
        sigma : float
            Gaussian smearing width in A (default 0.0 = no smearing).
            Use 0.02-0.05 for comparison with experiment.

        Returns
        -------
        dict
            {"r": list[float], "g_r": list[float]}
        """
        return compute_rdf(self.atoms_list, pair, rmax, nbins, sigma=sigma)

    def structure_factor(self, pair=None, qmax=15.0, nq=300, rmax=None):
        """Compute the structure factor S(q) from g(r) via Fourier transform.

        Parameters
        ----------
        pair : str, optional
            Pair to analyse. If None, total S(q).
        qmax : float
            Maximum q in 1/A (default 15.0).
        nq : int
            Number of q points (default 300).
        rmax : float, optional
            Maximum radius for RDF used in the transform.

        Returns
        -------
        dict
            {"q": list[float], "s_q": list[float]}
        """
        return compute_structure_factor(self.atoms_list, pair, qmax, nq, rmax)

    def averaged_rdf(self, pair=None, rmax=None, nbins=200):
        """Compute RDF per structure with mean and standard deviation.

        Parameters
        ----------
        pair : str, optional
            Pair to analyse, e.g. "Si-O". If None, total RDF.
        rmax : float, optional
            Maximum radius in A. Auto-detected from cell if None.
        nbins : int
            Number of histogram bins (default 200).

        Returns
        -------
        dict
            {"r": list, "g_r_mean": list, "g_r_std": list, "n_structures": int}
        """
        return compute_averaged_rdf(self.atoms_list, pair, rmax, nbins)

    # ── Advanced analysis ───────────────────────────────────────────────

    def ring_statistics(self, bond_pair=None, cutoff=None, max_ring=12):
        """Compute ring size statistics for network-forming structures.

        Parameters
        ----------
        bond_pair : tuple of str, optional
            Bond pair to trace, e.g. ("Si", "O"). Auto-detected if None.
        cutoff : float or dict, optional
            Bond cutoff. Uses analyser cutoff if None.
        max_ring : int
            Maximum ring size to search (default 12).

        Returns
        -------
        dict
            {"ring_sizes": list, "counts": list, "fractions": list,
             "bond_pair": tuple, "total_rings": int}
        """
        return compute_ring_statistics(
            self.atoms_list, bond_pair, cutoff, max_ring, self._get_cutoff)

    def voronoi(self, element=None):
        """Compute Voronoi tessellation indices (n3, n4, n5, n6).

        Parameters
        ----------
        element : str, optional
            Element to analyse. If None, all atoms included.

        Returns
        -------
        dict
            {"indices": list, "distribution": dict, "top_10": list,
             "mean_faces": float, "total_atoms": int}
        """
        return compute_voronoi(self.atoms_list, element)

    def energy_ranking(self):
        """Rank structures by potential energy per atom.

        Reads energy from atoms.info or attached calculator.

        Returns
        -------
        dict
            {"energies_per_atom": dict, "ranking": list, "best": int,
             "worst": int, "best_energy": float, "worst_energy": float,
             "spread": float}. Returns "warning" key if no energy data.
        """
        return compute_energy_ranking(self.atoms_list)

    # ── Multi-structure averaging ───────────────────────────────────────

    def averaged_cn(self, pair=None):
        """Compute per-structure mean CN with overall mean and std.

        Parameters
        ----------
        pair : str, optional
            Pair to analyse, e.g. "Si-O". If None, all pairs computed.

        Returns
        -------
        dict
            Keyed by pair string. Each value is a dict with
            "mean_per_structure", "overall_mean", "overall_std", "n_structures".
        """
        cn_per_structure = defaultdict(list)

        for atoms in self.atoms_list:
            nbr_dict, syms = self._build_neighbour_dict(atoms)
            unique = sorted(set(syms))
            n = len(atoms)

            for s1 in unique:
                for s2 in unique:
                    if pair is not None:
                        p = pair.split("-")
                        if not ((s1 == p[0] and s2 == p[1]) or
                                (s1 == p[1] and s2 == p[0])):
                            continue
                    key = f"{s1}-{s2}"
                    cns = []
                    for a in range(n):
                        if syms[a] != s1:
                            continue
                        cn = sum(1 for _, sj, _, _ in nbr_dict[a] if sj == s2)
                        cns.append(cn)
                    if cns:
                        cn_per_structure[key].append(np.mean(cns))

        result = {}
        for key, means in cn_per_structure.items():
            means = np.array(means)
            result[key] = {
                "mean_per_structure": means.tolist(),
                "overall_mean": float(np.mean(means)),
                "overall_std": float(np.std(means)),
                "n_structures": len(means),
            }
        return result

    # ── Summary and reporting ───────────────────────────────────────────

    def summary(self, show_angles=True):
        lines = []
        bar = "=" * 65

        n_structs = len(self.atoms_list)
        formula = self.atoms_list[0].get_chemical_formula(mode="hill")
        n_atoms = len(self.atoms_list[0])

        lines.append(f"\n{bar}")
        lines.append(f"  Structural Analysis: {formula} ({n_atoms} atoms)")
        lines.append(f"  N structures: {n_structs}")
        lines.append(f"  Cutoff mode: {self._cutoff_mode}")
        if isinstance(self.cutoff, dict):
            for pair in sorted(self.cutoff):
                lines.append(f"    {pair}: {self.cutoff[pair]:.2f} A")
        else:
            lines.append(f"    All pairs: {self.cutoff} A")
        lines.append(bar)

        d = self.density()
        lines.append(f"\n  Density: {d['mean']:.2f} +/- {d['std']:.2f} g/cm3")

        bd = self.bond_distances()
        if bd:
            lines.append(f"\n  Bond distances:")
            lines.append(f"  {'Pair':<10} {'Mean (A)':>10} {'Std':>8} "
                         f"{'Min':>8} {'Max':>8} {'Count':>8}")
            lines.append(f"  {'-'*54}")
            for pair, data in bd.items():
                lines.append(
                    f"  {pair:<10} {data['mean']:>10.3f} {data['std']:>7.3f} "
                    f"{data['min']:>8.3f} {data['max']:>8.3f} "
                    f"{data['count']:>8}")

        cn = self.coordination()
        if cn:
            try:
                from ..pipeline.random_gen import _classify_bond
            except ImportError:
                from amorphgen.pipeline.random_gen import _classify_bond

            bonding_cn = {}
            nonbonded_cn = {}
            for pair, data in cn.items():
                s1, s2 = pair.split("-")
                bt = _classify_bond(s1, s2)
                if bt == "ionic" or (bt == "covalent" and s1 != s2):
                    bonding_cn[pair] = data
                else:
                    nonbonded_cn[pair] = data

            def _fmt(pair, data):
                l1 = (f"  {pair}: mean={data['mean']:.1f} +/- "
                      f"{data['std']:.1f} [{data['min']},{data['max']}]")
                parts = [f"CN={cn}: {pct:.1f}%"
                         for cn, pct in sorted(data["distribution"].items())
                         if pct >= 0.5]
                l2 = "    Distribution: " + ", ".join(parts)
                return [l1, l2]

            if bonding_cn:
                lines.append(f"\n  Bonding coordination numbers:")
                for pair, data in bonding_cn.items():
                    lines.extend(_fmt(pair, data))

            if nonbonded_cn:
                nb = {k: v for k, v in nonbonded_cn.items()
                      if v["mean"] > 0.05}
                if nb:
                    lines.append(f"\n  Non-bonded contacts:")
                    for pair, data in nb.items():
                        lines.extend(_fmt(pair, data))

        if show_angles:
            ba = self.bond_angles()
            if ba:
                lines.append(f"\n  Bond angles:")
                lines.append(f"  {'Triplet':<12} {'Mean (deg)':>10} "
                             f"{'Std':>8} {'Count':>8}")
                lines.append(f"  {'-'*42}")
                for triplet, data in ba.items():
                    lines.append(
                        f"  {triplet:<12} {data['mean']:>10.1f} "
                        f"{data['std']:>7.1f} {data['count']:>8}")

        lines.append(f"\n{bar}\n")

        text = "\n".join(lines)
        print(text)
        return text

    def per_structure_summary(self) -> str:
        """
        Analyse each structure individually and produce a comparison table.

        Returns
        -------
        str — formatted table (also printed).
        """
        lines = []
        bar = "=" * 80
        formula = self.atoms_list[0].get_chemical_formula(mode="hill")
        n_atoms = len(self.atoms_list[0])

        lines.append(f"\n{bar}")
        lines.append(f"  Per-Structure Analysis: {formula} ({n_atoms} atoms)")
        lines.append(f"  N structures: {len(self.atoms_list)}")
        lines.append(bar)

        # Determine bonding pairs for CN
        try:
            from ..pipeline.random_gen import _classify_bond
        except ImportError:
            from amorphgen.pipeline.random_gen import _classify_bond

        unique = sorted(set(self.atoms_list[0].get_chemical_symbols()))
        bonding_pairs = []
        for s1 in unique:
            for s2 in unique:
                bt = _classify_bond(s1, s2)
                if bt == "ionic" or (bt == "covalent" and s1 != s2):
                    bonding_pairs.append(f"{s1}-{s2}")

        # If single element, use same-species
        if not bonding_pairs:
            bonding_pairs = [f"{unique[0]}-{unique[0]}"]

        # Header
        cn_headers = [f"CN({p})" for p in bonding_pairs[:3]]
        header = f"  {'#':<5} {'Density':>8} {'E/atom':>10}"
        for h in cn_headers:
            header += f" {h:>10}"
        lines.append(f"\n{header}")
        lines.append(f"  {'-'*(len(header)-2)}")

        # Per-structure data
        all_densities = []
        all_energies = []
        all_cns = {p: [] for p in bonding_pairs[:3]}

        for i, atoms in enumerate(self.atoms_list):
            sa_single = StructureAnalyser([atoms], cutoff=self.cutoff)

            # Density
            d = sa_single.density()
            dens = d["mean"]
            all_densities.append(dens)

            # Energy (per atom, using THIS structure's atom count)
            n_at = len(atoms)
            e_str = ""
            for key in ['energy', 'Energy', 'potential_energy']:
                if key in atoms.info:
                    e = atoms.info[key] / n_at
                    all_energies.append(e)
                    e_str = f"{e:.4f}"
                    break
            if not e_str:
                try:
                    e = atoms.get_potential_energy() / n_at
                    all_energies.append(e)
                    e_str = f"{e:.4f}"
                except Exception:
                    e_str = "N/A"

            # CN for bonding pairs
            cn = sa_single.coordination()
            row = f"  {i:<5} {dens:>8.2f} {e_str:>10}"
            for p in bonding_pairs[:3]:
                cn_val = cn.get(p, {}).get("mean", 0)
                all_cns[p].append(cn_val)
                row += f" {cn_val:>10.1f}"

            lines.append(row)

        # Summary statistics
        lines.append(f"  {'-'*(len(header)-2)}")
        mean_row = f"  {'Mean':<5} {np.mean(all_densities):>8.2f}"
        std_row = f"  {'Std':<5} {np.std(all_densities):>8.2f}"
        if all_energies:
            mean_row += f" {np.mean(all_energies):>10.4f}"
            std_row += f" {np.std(all_energies):>10.4f}"
        else:
            mean_row += f" {'N/A':>10}"
            std_row += f" {'N/A':>10}"
        for p in bonding_pairs[:3]:
            mean_row += f" {np.mean(all_cns[p]):>10.1f}"
            std_row += f" {np.std(all_cns[p]):>10.1f}"
        lines.append(mean_row)
        lines.append(std_row)
        lines.append(f"\n{bar}\n")

        text = "\n".join(lines)
        print(text)
        return text

    def save_report(self, filepath, text=None, show_angles=True):
        """Save the summary report to a text file.

        Parameters
        ----------
        filepath : str
            Output file path.
        text : str, optional
            Pre-computed report text. If None, calls summary().
        show_angles : bool
            Include bond angles in the report (default True).
        """
        if text is None:
            text = self.summary(show_angles=show_angles)
        with open(filepath, "w") as f:
            f.write(text)
        print(f"  Report saved: {filepath}")

    def plot(self, **kwargs):
        """Generate and save analysis plots (RDF, CN distribution, bond angles).

        Parameters
        ----------
        output_dir : str
            Directory for output files (default ".").
        prefix : str
            Filename prefix (default "analysis").
        rdf_pairs : list of str, optional
            Pairs to plot, e.g. ["Si-O", "O-O"]. If None, all pairs.
        rmax : float, optional
            Maximum radius for RDF. Auto-detected if None.
        save_csv : bool
            Also save raw data as CSV files (default True).
        """
        return plot_analysis(self, **kwargs)
