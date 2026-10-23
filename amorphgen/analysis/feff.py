"""Forward-model EXAFS chi(k) from an ensemble with FEFF.

What this does
--------------
:mod:`amorphgen.analysis.exafs` computes the quantities an EXAFS fit
*reports* (N, R, sigma^2, C3) from the geometry. This module goes one
step further and produces the quantity an EXAFS experiment *measures*:
the fine-structure function chi(k), configurationally averaged over
absorber sites and over the members of an ensemble.

The approach is **forward modelling, not fitting**. Every absorber site
in the ensemble contributes one FEFF calculation, and the average over
sites is the prediction. No N or sigma^2 is fitted in this forward calculation. Cluster size, path
selection, potential settings and S0^2 remain modelling choices. The
structural disorder that an experimentalist has to model with a
Debye-Waller factor is supplied directly by the ensemble.

Coordination and static disorder are supplied by the structural ensemble.
Comparisons with experiment still depend on amplitude calibration, energy
alignment, potential settings and thermal disorder; forward modelling does
not remove those sources of uncertainty.

FEFF is not a dependency
------------------------
FEFF is an external Fortran code with its own distribution terms; it is
not pip-installable and AmorphGen does not ship or require it. The split
here keeps that out of the critical path:

- **Writing** ``feff.inp`` needs nothing but ASE and NumPy. It always
  works, and it is what CI tests.
- **Running** needs a ``feff`` executable on PATH. That step is
  optional, and :func:`run_feff_jobs` fails immediately with a clear
  message if the binary is absent rather than half-way through a job
  array.
- **Reading and averaging** the output needs nothing but NumPy, and
  works on whatever ``chi.dat`` / ``xmu.dat`` files exist, however they
  were produced.

So the usual workflow is: write the job tree on a login node, run the
array on the cluster (or anywhere else), collect the results locally.
On a machine that has FEFF, :func:`feff_pipeline` (CLI: ``--feff-write
DIR --feff-run``) does all three in one call, in parallel and resumably.

The generated input targets the FEFF8L / FEFF10 card set (``EDGE`` and
``S02``). FEFF6 uses a different header (``HOLE``) and is not written
here; ``extra_cards`` is the escape hatch for anything else.

Cost
----
One FEFF run per absorber site. For a 400-atom cell with 160 Ga and 20
structures that is 3200 runs for one edge -- seconds each for an
EXAFS-only calculation, and embarrassingly parallel, so it maps onto a
job array. ``n_sites`` sub-samples absorbers per structure; convergence must be checked by varying the site and structure counts.
:func:`collect_feff_results` reports SEM across independent structure means;
this does not test path truncation, cluster size or model accuracy.

S0^2
----
``S02`` is written as 1.0 to leave the amplitude unreduced. FEFF8.5L can
calculate its intrinsic overlap reduction when S02=0; that special setting
does not mean zero spectral amplitude. Experimental amplitude calibration
must be stated separately from the forward-model settings.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import hashlib
import sys
import tempfile
import signal
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list

#: Starting cluster radius (A); establish adequacy by convergence tests.
DEFAULT_RADIUS = 6.0

#: FEFF's compiled-in cluster size limit (``natx``). Exceeding it makes
#: FEFF reject the input, which would otherwise only surface after a whole
#: job tree had been written and queued.
MAX_CLUSTER_ATOMS = 500

#: Paths FEFF's final sum can hold: ``npx`` in FEFF8.5L's FF2X module. Its
#: readers stop at this many paths without any message, so a longer path
#: list (loose CRITERIA, long RPATH, high NLEG) gives a truncated chi(k).
FF2X_MAX_PATHS = 15000

#: Default header cards. Every value can be overridden per call, and
#: anything not listed here goes in via ``extra_cards``.
DEFAULT_CARDS = {
    "EDGE": "K",
    "S02": "1.0",              # deliberately unreduced -- see module docstring
    "CONTROL": "1 1 1 1 1 1",
    "PRINT": "1 0 0 0 0 0",
    "EXCHANGE": "0",           # Hedin-Lundqvist
    "RPATH": "6.0",           # overridden to track the cluster radius
    "EXAFS": "20.0",
    "NLEG": "4",
}


# ── cluster extraction ───────────────────────────────────────────────────

def structure_neighbours(atoms, radius=DEFAULT_RADIUS):
    """One neighbour list for a whole structure, to share across absorbers.

    :func:`feff_cluster` builds the list itself when it is not given one,
    which is O(N) work repeated for every absorber. A 400-atom cell with
    160 absorbers would pay for it 160 times over; computing it once per
    structure and passing it in is what :func:`write_feff_jobs` does.

    Returns
    -------
    tuple
        ``(i, j, d, D)`` as returned by ``ase.neighborlist.neighbor_list``
        with ``"ijdD"``.
    """
    return neighbor_list("ijdD", atoms, cutoff=float(radius))


def feff_cluster(atoms, absorber_index, radius=DEFAULT_RADIUS,
                 neighbours=None):
    """Finite cluster around one absorber, with the absorber at the origin.

    FEFF works on a cluster, not a periodic cell. Periodic images are
    resolved through ASE's neighbour list, so an image of the absorber
    itself is included as a scatterer when it falls inside ``radius`` --
    which is correct, and is what a naive minimum-image extraction gets
    wrong.

    Parameters
    ----------
    atoms : ase.Atoms
        The periodic (or finite) structure.
    absorber_index : int
        Index of the absorbing atom in ``atoms``.
    radius : float
        Cluster radius in A.
    neighbours : tuple, optional
        A pre-built ``(i, j, d, D)`` from :func:`structure_neighbours` for
        this structure and radius. Supply it when extracting clusters for
        many absorbers in the same structure.

    Returns
    -------
    dict
        ``{"symbols": [...], "positions": (n, 3) array, "distances":
        (n,) array}``, sorted by distance, with the absorber first at
        (0, 0, 0). ``symbols[0]`` is the absorbing element.
    """
    if not 0 <= absorber_index < len(atoms):
        raise IndexError(
            f"absorber_index {absorber_index} outside 0..{len(atoms) - 1}")

    syms = atoms.get_chemical_symbols()
    i, j, d, D = (structure_neighbours(atoms, radius) if neighbours is None
                  else neighbours)
    mask = i == absorber_index
    order = np.argsort(d[mask])

    positions = np.vstack([np.zeros(3), D[mask][order]])
    distances = np.concatenate([[0.0], d[mask][order]])
    symbols = [syms[absorber_index]] + [syms[k] for k in j[mask][order]]

    return {"symbols": symbols, "positions": positions, "distances": distances}


def _potential_table(symbols):
    """Map cluster symbols to FEFF unique-potential indices.

    ipot 0 is reserved for the absorbing atom. Every element that occurs
    among the *scatterers* gets its own ipot >= 1, ordered by decreasing
    Z. The absorber's own element gets an ipot >= 1 only when other atoms
    of that element are present, so no potential is declared unused.

    Returns
    -------
    tuple
        ``(rows, ipot_of_symbol)`` where ``rows`` is a list of
        ``(ipot, Z, symbol)`` starting with the absorber's ipot 0.
    """
    absorber = symbols[0]
    scatterers = sorted(set(symbols[1:]),
                        key=lambda s: -atomic_numbers[s])

    rows = [(0, atomic_numbers[absorber], absorber)]
    ipot_of = {}
    for n, sym in enumerate(scatterers, start=1):
        ipot_of[sym] = n
        rows.append((n, atomic_numbers[sym], sym))
    return rows, ipot_of


# ── feff.inp ─────────────────────────────────────────────────────────────

def write_feff_input(atoms, absorber_index, path=None, radius=DEFAULT_RADIUS,
                     edge="K", title=None, cards=None, extra_cards=None,
                     neighbours=None, stoichiometry="cluster"):
    """Build a ``feff.inp`` for one absorber site.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure containing the absorber.
    absorber_index : int
        Index of the absorbing atom.
    path : str, optional
        Where to write the file. When ``None`` the text is returned
        without touching the filesystem.
    radius : float
        Cluster radius in A (:data:`DEFAULT_RADIUS`).
    edge : str
        Absorption edge: ``"K"``, ``"L1"``, ``"L2"``, ``"L3"``, ...
    title : str, optional
        TITLE line. Defaults to the formula, the absorber and the radius.
    cards : dict, optional
        Overrides for :data:`DEFAULT_CARDS`, e.g. ``{"EXAFS": "18.0"}``.
        Pass ``{"SCF": "4.0"}`` to switch on self-consistent potentials
        (much slower; non-SCF muffin-tin potentials are the normal choice
        for EXAFS amplitudes over many sites).
    extra_cards : list of str, optional
        Raw lines inserted verbatim before POTENTIALS, for cards this
        module does not model.
    neighbours : tuple, optional
        A pre-built neighbour list for this structure, passed through to
        :func:`feff_cluster`.
    stoichiometry : {"cluster", "structure"}
        ``cluster`` preserves FEFF's inferred potential populations.
        ``structure`` writes reduced parent-cell element counts as xnatph,
        with absorber population 0.001 and automatic angular-momentum limits.
        This avoids spherical-cluster composition bias in a periodic solid.
        It requires three-dimensional periodicity and a cluster containing
        a scattering potential for every element in the parent structure.

    Returns
    -------
    str
        The file contents (also written to ``path`` when given).
    """
    if stoichiometry not in ("cluster", "structure"):
        raise ValueError("stoichiometry must be 'cluster' or 'structure'")
    if stoichiometry == "structure" and not np.all(atoms.pbc):
        raise ValueError("structure stoichiometry requires three-dimensional periodicity")
    cluster = feff_cluster(atoms, absorber_index, radius,
                           neighbours=neighbours)
    symbols = cluster["symbols"]

    if len(symbols) > MAX_CLUSTER_ATOMS:
        import warnings
        warnings.warn(
            f"cluster around atom {absorber_index} has {len(symbols)} atoms, "
            f"over FEFF's usual limit of {MAX_CLUSTER_ATOMS}; FEFF will "
            f"reject this input with that compiled limit. Use a suitable "
            f"FEFF build or reassess radius (currently {radius:.1f} A).")
    rows, ipot_of = _potential_table(symbols)
    populations = None
    if stoichiometry == "structure":
        elements, counts = np.unique(atoms.get_chemical_symbols(), return_counts=True)
        if set(elements) != set(ipot_of):
            raise ValueError("cluster omits a parent-element scattering potential; "
                             "increase radius for structure stoichiometry")
        populations = dict(zip(elements, counts // np.gcd.reduce(counts)))

    header = dict(DEFAULT_CARDS)
    header["EDGE"] = edge
    # Path enumeration beyond the cluster finds nothing, and an RPATH well
    # under the radius silently throws away shells that were paid for. Tie
    # the two together unless the caller says otherwise.
    header["RPATH"] = f"{radius:.1f}"
    if cards:
        header.update({k: str(v) for k, v in cards.items()})

    if title is None:
        title = (f"{atoms.get_chemical_formula(mode='hill')} "
                 f"absorber {symbols[0]}[{absorber_index}] "
                 f"cluster {radius:.1f} A")

    lines = [f"TITLE {title}", "*",
             "* Written by AmorphGen (amorphgen.analysis.feff)."]
    if header["S02"] == DEFAULT_CARDS["S02"]:
        lines += ["* S02 is 1.0 by design: this is a forward model, not a "
                  "fit.",
                  "* Apply the experimental S02 when comparing amplitudes."]
    else:
        lines += [f"* S02 set to {header['S02']} by the caller, overriding "
                  "AmorphGen's",
                  "* unreduced default; FEFF interprets special values such as zero."]
    lines.append("")
    for key, value in header.items():
        lines.append(f"{key:<10}{value}")
    if extra_cards:
        lines.extend(extra_cards)

    lines += ["", "POTENTIALS", "*    ipot     Z    tag" +
              (" lmax1 lmax2 xnatph" if populations is not None else "")]
    for ipot, z, sym in rows:
        suffix = ""
        if populations is not None:
            population = 0.001 if ipot == 0 else populations[sym]
            suffix = f" -1 -1 {population:g}"
        lines.append(f"    {ipot:<8d}{z:<6d}{sym}{suffix}")

    lines += ["", "ATOMS",
              "*          x           y           z   ipot  tag      distance"]
    for n, sym in enumerate(symbols):
        x, y, z = cluster["positions"][n]
        ipot = 0 if n == 0 else ipot_of[sym]
        lines.append(f"  {x:11.5f} {y:11.5f} {z:11.5f}   {ipot:<5d}{sym:<8s}"
                     f"{cluster['distances'][n]:9.5f}")
    lines += ["END", ""]

    text = "\n".join(lines)
    if path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    return text


# ── job tree ─────────────────────────────────────────────────────────────

def write_feff_jobs(atoms_list, absorber, output_dir, n_sites=None,
                    radius=DEFAULT_RADIUS, edge="K", seed=0, cards=None,
                    extra_cards=None, stoichiometry="cluster"):
    """Write one ``feff.inp`` per sampled absorber site, plus a manifest.

    Layout::

        output_dir/
          manifest.json                  every job, with its site index
          joblist.txt                    one directory per line
          run_all.sh                     serial fallback runner
          s0000_a0012/feff.inp
          s0000_a0045/feff.inp
          ...

    ``joblist.txt`` is the file to drive a scheduler array with; the
    index of a line is the array task id::

        DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" joblist.txt)
        python -m amorphgen.analysis.feff --jobs-dir . --job "$DIR"

    Parameters
    ----------
    atoms_list : list of ase.Atoms
        The ensemble.
    absorber : str
        Absorbing element symbol, e.g. ``"Ga"``.
    output_dir : str
        Directory to create.
    n_sites : int, optional
        Absorber sites per structure. ``None`` uses every one. Sampling
        is without replacement and deterministic given ``seed``.
    radius, edge, cards, extra_cards, stoichiometry
        Passed to :func:`write_feff_input`.
    seed : int
        Seed for the site sampling.

    Returns
    -------
    dict
        The manifest: ``{"absorber", "edge", "radius", "n_structures",
        "n_jobs", "jobs": [{"dir", "structure", "site"}, ...]}``.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if n_sites is not None and (not isinstance(n_sites, int) or n_sites < 1):
        raise ValueError("n_sites must be a positive integer")
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    jobs, inputs = [], []
    for s_idx, atoms in enumerate(atoms_list):
        sites = [n for n, sym in enumerate(atoms.get_chemical_symbols())
                 if sym == absorber]
        if not sites:
            continue
        if n_sites is not None and n_sites < len(sites):
            sites = sorted(rng.choice(sites, size=n_sites, replace=False))

        neighbours = structure_neighbours(atoms, radius)
        for site in sites:
            name = f"s{s_idx:04d}_a{site:04d}"
            job_dir = os.path.join(output_dir, name)
            text = write_feff_input(atoms, site, radius=radius, edge=edge,
                                    cards=cards, extra_cards=extra_cards,
                                    neighbours=neighbours, stoichiometry=stoichiometry)
            path = Path(job_dir) / "feff.inp"
            # Preflight the entire tree before changing any inputs or manifest.
            if path.exists() and path.read_text() != text:
                raise ValueError(f"{path}: input changed; use a new job directory")
            if not path.exists() and Path(job_dir).exists() and any(Path(job_dir).iterdir()):
                raise ValueError(f"{job_dir}: nonempty directory without feff.inp")
            inputs.append((path, text))
            jobs.append({"dir": name, "structure": s_idx, "site": int(site),
                         "input_sha256": hashlib.sha256(text.encode()).hexdigest()})

    if not jobs:
        raise ValueError(
            f"no {absorber} atoms in any of the {len(atoms_list)} structures")

    manifest = {
        "schema_version": 2,
        "absorber": absorber,
        "edge": str((cards or {}).get("EDGE", edge)),
        "radius": radius,
        "stoichiometry": stoichiometry,
        "cards": {**DEFAULT_CARDS, "EDGE": edge, "RPATH": f"{radius:.1f}",
                  **{k: str(v) for k, v in (cards or {}).items()}},
        "extra_cards": list(extra_cards or []),
        "n_structures": len(atoms_list),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    for path, text in inputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(text)
    with open(os.path.join(output_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    with open(os.path.join(output_dir, "joblist.txt"), "w") as fh:
        fh.write("\n".join(j["dir"] for j in jobs) + "\n")

    runner = os.path.join(output_dir, "run_all.sh")
    with open(runner, "w") as fh:
        fh.write(
            "#!/bin/bash\n"
            "# Serial fallback. For a cluster, drive joblist.txt from a\n"
            "# scheduler array instead -- these jobs are independent.\n"
            f"# {len(jobs)} jobs\n"
            'cd "$(dirname "$0")" || exit 1\n'
            'exec "${PYTHON:-python}" -m amorphgen.analysis.feff --jobs-dir .\n'
        )
    os.chmod(runner, 0o755)

    return manifest


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_status(directory, status):
    """Replace the status atomically, so interrupted runs cannot look complete."""
    path = Path(directory) / "run_status.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as fh:
        json.dump(status, fh, indent=2)
        temporary = fh.name
    os.replace(temporary, path)


def _job_path(jobs_dir, job):
    root = Path(jobs_dir).resolve()
    path = (root / job["dir"]).resolve()
    if path.parent != root:
        raise ValueError(f"job directory must be directly inside {root}: {job['dir']}")
    return path


def _input_hash(job_dir, job):
    current = _sha256(job_dir / "feff.inp")
    if job.get("input_sha256", current) != current:
        raise ValueError(f"{job_dir}: feff.inp differs from its manifest")
    return current


def _count_listed_paths(path):
    """Number of paths in a FEFF ``list.dat``; None when there is no file."""
    if not path.exists():
        return None
    count = 0
    with path.open(errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0].isdigit():
                count += 1
    return count


def _paths_read(chi_path):
    """``ntotal`` from the ``nused/ntotal paths used`` line of ``chi.dat``.

    None when the line is absent or the four-digit field overflowed.
    """
    with chi_path.open(errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if "paths used" in line and "/" in line:
                total = line.split("/")[1].split()[0]
                return int(total) if total.isdigit() else None
    return None


def _check_path_sum(scratch):
    """Reject a run whose final sum read fewer paths than FEFF listed.

    Returns the number of listed paths, or None when FEFF wrote no list.
    """
    listed = _count_listed_paths(scratch / "list.dat")
    if listed is None:
        return None
    read = _paths_read(scratch / "chi.dat")
    if (read is not None and read < listed) or (read is None and listed > FF2X_MAX_PATHS):
        raise ValueError(
            f"FEFF listed {listed} paths but its final sum holds at most "
            f"{FF2X_MAX_PATHS} and drops the rest silently, so chi(k) is "
            f"truncated. Raise CRITERIA or reduce RPATH / NLEG.")
    return listed


def run_feff_jobs(jobs_dir, executable="feff", jobs=None, timeout=None,
                  workers=1, skip_complete=False):
    """Run FEFF in each job directory. Requires the binary on PATH.

    Parameters
    ----------
    jobs_dir : str
        Directory written by :func:`write_feff_jobs`.
    executable : str
        FEFF binary name or path (default ``"feff"``).
    jobs : list of str, optional
        Job directory names to run. Default: every job in the manifest.
    timeout : float, optional
        Per-job timeout in seconds.
    workers : int
        Jobs run at once (default 1). FEFF is single-threaded and each job
        has its own directory, lock and scratch space, so independent jobs
        can share a multi-core machine.
    skip_complete : bool
        Leave jobs whose managed run is complete and whose input and output
        fingerprints still match (default False: run every job). Jobs never
        run, failed, changed or imported from outside are run.

    Returns
    -------
    dict
        ``{"ran": int, "skipped": int, "failed": [(dir, reason), ...]}``.
        A failure reason is FEFF's own diagnostic line from ``feff.log``
        where it printed one -- a muffin-tin overlap, say, which in an
        amorphous ensemble almost always means a too-close pair that
        ``--analyse --check-dimers`` would have flagged.

    Raises
    ------
    FileNotFoundError
        If the executable is not on PATH. Checked once, before any job
        starts, so a missing binary fails in a second rather than part
        way through an array.
    """
    binary = shutil.which(executable) or (
        executable if os.path.isfile(executable) and
        os.access(executable, os.X_OK) else None)
    if binary is None:
        raise FileNotFoundError(
            f"FEFF executable {executable!r} not found on PATH.\n"
            "AmorphGen writes FEFF input but does not ship FEFF: it is an\n"
            "external Fortran code with its own distribution terms.\n"
            "Obtain FEFF10 (or FEFF8L) from the FEFF Project, build it, and\n"
            "put the binary on PATH -- or copy the job tree to a machine\n"
            "that has it and run joblist.txt there, then point\n"
            "collect_feff_results() at the output."
        )

    if (isinstance(workers, (bool, np.bool_))
            or not isinstance(workers, (int, np.integer)) or workers < 1):
        raise ValueError("workers must be a positive integer")
    binary = os.path.abspath(binary)
    manifest = read_manifest(jobs_dir)
    entries = {job["dir"]: job for job in manifest["jobs"]}
    names = jobs if jobs is not None else [j["dir"] for j in manifest["jobs"]]
    if any(name not in entries for name in names):
        raise ValueError("requested job is not in the manifest")

    todo = [name for name in names
            if not (skip_complete and _managed_complete(jobs_dir, entries[name]))]
    tasks = [(str(jobs_dir), entries[name], binary, timeout) for name in todo]
    if workers == 1 or len(tasks) < 2:
        outcomes = [_run_feff_job(*task) for task in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_run_feff_job, *zip(*tasks)))
    failed = [(name, reason) for name, reason in outcomes if reason is not None]
    return {"ran": len(outcomes) - len(failed),
            "skipped": len(names) - len(todo), "failed": failed}


def _managed_complete(jobs_dir, job):
    """True for a managed run that finished and still matches its record."""
    job_dir = _job_path(jobs_dir, job)
    if not (job_dir / "run_status.json").exists() or not (job_dir / "chi.dat").exists():
        return False
    try:
        return not _check_output(job_dir, job, "chi.dat", False)
    except (OSError, ValueError):
        return False


def _run_feff_job(jobs_dir, job, binary, timeout):
    """Run one job in a scratch directory; return ``(dir, failure or None)``."""
    name = job["dir"]
    job_dir = _job_path(jobs_dir, job)
    fingerprint = _input_hash(job_dir, job)
    status = {"state": "running", "input_sha256": fingerprint,
              "executable": binary}
    lock = job_dir / ".amorphgen_feff.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return name, "another run holds .amorphgen_feff.lock"
    os.close(lock_fd)
    try:
        _write_status(job_dir, status)
        # Fresh spectra are required even for a partial-stage rerun. Carry
        # intermediate files so CONTROL-selected stages can still be used.
        with tempfile.TemporaryDirectory(prefix=".feff-", dir=job_dir) as tmp:
            scratch = Path(tmp)
            for path in job_dir.iterdir():
                if path.is_file() and path.name not in {
                    "chi.dat", "xmu.dat", "feff.log", "run_status.json",
                    ".amorphgen_feff.lock",
                }:
                    shutil.copy2(path, scratch / path.name)
            with (job_dir / "feff.log").open("w") as log:
                with subprocess.Popen([binary], cwd=scratch, stdout=log,
                                      stderr=subprocess.STDOUT,
                                      start_new_session=True) as proc:
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        # FEFF drivers launch child executables. Stop the
                        # entire group before removing its scratch files.
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        raise
            if proc.returncode:
                raise ValueError(_feff_failure_reason(
                    str(job_dir), f"exit code {proc.returncode}"))
            if not (scratch / "chi.dat").exists():
                raise ValueError(_feff_failure_reason(
                    str(job_dir), "no chi.dat written"))
            listed = _check_path_sum(scratch)
            if listed is not None:
                status["paths_listed"] = listed
            outputs = {}
            for filename in ("chi.dat", "xmu.dat"):
                path = scratch / filename
                if path.exists():
                    read_chi(path)
                    outputs[filename] = _sha256(path)
            if _sha256(job_dir / "feff.inp") != fingerprint:
                raise ValueError("feff.inp changed during execution")
            for filename in ("chi.dat", "xmu.dat"):
                (job_dir / filename).unlink(missing_ok=True)
            for path in scratch.iterdir():
                if path.is_file():
                    os.replace(path, job_dir / path.name)
            status.update(state="complete", outputs=outputs)
            _write_status(job_dir, status)
            return name, None
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        reason = (f"timed out after {timeout} s"
                  if isinstance(exc, subprocess.TimeoutExpired) else str(exc))
        status.update(state="failed", reason=reason)
        _write_status(job_dir, status)
        return name, reason
    finally:
        lock.unlink(missing_ok=True)


#: Lines FEFF prints when it gives up, most specific first. A failing run
#: says something precise in its log and then exits; reporting only "no
#: chi.dat" throws that away and leaves the user with nothing to act on.
_FEFF_ERROR_PATTERNS = (
    # Most informative first: this one is printed on line 3 of the log,
    # long before the crash it eventually causes.
    "TWO ATOMS VERY CLOSE TOGETHER",
    "Rmt >= distance to nearest neighbor",
    "FOLP for POTENTIAL",
    "MOVRLP",
    "too many atoms",
    "Fortran runtime error",
    "Segmentation fault",
    "Error termination",
)

#: Patterns whose cause is a too-close pair, which get the dimer hint.
_OVERLAP_PATTERNS = frozenset((
    "TWO ATOMS VERY CLOSE TOGETHER",
    "Rmt >= distance to nearest neighbor",
    "FOLP for POTENTIAL",
    "MOVRLP",
))

#: Overlap failures have one overwhelmingly common cause in an amorphous
#: ensemble, and AmorphGen already ships the tool that finds it.
_OVERLAP_HINT = ("muffin-tin overlap: two atoms in the cluster are closer "
                 "than is physical. Run `amorphgen --analyse --check-dimers` "
                 "on the ensemble -- this is what a dimer defect looks like "
                 "to FEFF")


def _feff_failure_reason(job_dir, fallback):
    """Pull FEFF's own diagnostic out of ``feff.log``, falling back to a
    generic message when it says nothing recognisable."""
    log = os.path.join(job_dir, "feff.log")
    try:
        with open(log, errors="replace") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    except OSError:
        return fallback

    for pattern in _FEFF_ERROR_PATTERNS:
        for line in lines:
            if pattern in line:
                if pattern in _OVERLAP_PATTERNS:
                    return f"{line} -- {_OVERLAP_HINT}"
                return line
    return fallback


def find_feff_trees(path):
    """Job trees under ``path``: itself if it holds ``manifest.json``, else
    each immediate subdirectory that does (the per-absorber layout that
    several absorbers produce), in sorted order."""
    path = Path(path)
    if (path / "manifest.json").is_file():
        return [str(path)]
    trees = sorted(str(d) for d in path.iterdir()
                   if d.is_dir() and (d / "manifest.json").is_file()) if path.is_dir() else []
    if not trees:
        raise FileNotFoundError(
            f"no FEFF job tree (manifest.json) in {str(path)!r} or its subdirectories")
    return trees


def read_manifest(jobs_dir):
    """Load ``manifest.json`` from a job tree."""
    path = os.path.join(jobs_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no manifest.json in {jobs_dir!r}; is this a directory written "
            "by write_feff_jobs()?")
    with open(path) as fh:
        return json.load(fh)


# ── reading FEFF output ──────────────────────────────────────────────────

#: Column indices of (k, chi) in the two output files FEFF writes.
#: chi.dat is ``k chi mag phase``; xmu.dat is ``omega e k mu mu0 chi``.
_COLUMNS = {"chi.dat": (0, 1), "xmu.dat": (2, 5)}


def read_chi(path, kcol=None, chicol=None):
    """Read chi(k) from a FEFF ``chi.dat`` or ``xmu.dat``.

    Columns are taken from the file name (:data:`_COLUMNS`) unless
    ``kcol`` / ``chicol`` say otherwise. Header prose before the data and
    comment lines (``#`` or ``*``) are skipped. Once data begin, malformed
    rows fail with a file/line diagnostic. k must be finite and increasing.

    Returns
    -------
    tuple
        ``(k, chi)`` as float arrays.
    """
    base = os.path.basename(path).lower()
    if kcol is None or chicol is None:
        default = _COLUMNS.get(base)
        if default is None:
            raise ValueError(
                f"cannot infer columns for {base!r}; pass kcol and chicol "
                f"(known: {', '.join(sorted(_COLUMNS))})")
        kcol = default[0] if kcol is None else kcol
        chicol = default[1] if chicol is None else chicol

    ks, chis = [], []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("*"):
                continue
            parts = line.split()
            try:
                # Header prose is allowed before the first numeric row.
                float(parts[0].replace("D", "E").replace("d", "e"))
            except ValueError:
                if not ks:
                    continue
                raise ValueError(f"{path}:{lineno}: malformed spectrum row")
            try:
                kval = float(parts[kcol].replace("D", "E").replace("d", "e"))
                value = float(parts[chicol].replace("D", "E").replace("d", "e"))
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}:{lineno}: malformed spectrum row") from exc
            if not np.isfinite([kval, value]).all():
                raise ValueError(f"{path}:{lineno}: non-finite spectrum value")
            ks.append(kval)
            chis.append(value)

    if not ks:
        raise ValueError(f"no numeric data found in {path!r}")
    return _validate_spectrum(ks, chis, str(path))


def _validate_spectrum(k, chi, label="spectrum"):
    k, chi = np.asarray(k, dtype=float), np.asarray(chi, dtype=float)
    if k.ndim != 1 or chi.ndim != 1 or k.shape != chi.shape or k.size < 2:
        raise ValueError(f"{label}: need matching one-dimensional arrays with at least two points")
    if not np.isfinite(k).all() or not np.isfinite(chi).all():
        raise ValueError(f"{label}: non-finite spectrum value")
    if np.any(np.diff(k) <= 0):
        raise ValueError(f"{label}: k must be strictly increasing")
    return k, chi


def average_chi(sources, kmax=20.0, kstep=0.05, structure_ids=None):
    """Configurational average of chi(k) over absorber sites.

    Use only the intersection of the source k ranges. With structure_ids,
    average sites within each structure, then give each structure equal
    weight. The SEM is estimated across independent structure means.
    Without grouping, return the pooled site mean and undefined (NaN) SEM.
    chi_std always describes the pooled site spread, not uncertainty.

    Parameters
    ----------
    sources : iterable
        Paths to ``chi.dat`` / ``xmu.dat`` files, or ``(k, chi)`` tuples.
    kmax : float
        Upper end of the common grid (A^-1).
    kstep : float
        Grid spacing (A^-1). 0.05 is the usual EXAFS convention and what
        :func:`chi_fourier_transform` expects.

    Returns
    -------
    dict
        ``{"k", "chi", "chi_std", "chi_sem", "n_sites", "n_structures",
        "chi_per_structure"}``. SEM is NaN for fewer than two contributing
        structures. ``chi_per_structure`` holds one site-averaged curve per
        structure (rows in first-appearance order), or None without
        grouping. Independent structures are assumed; correlated trajectory
        frames need blocking.
    """
    if not np.isfinite([kmax, kstep]).all() or kmax <= 0 or kstep <= 0:
        raise ValueError("kmax and kstep must be positive and finite")

    spectra = []
    for src in sources:
        if isinstance(src, (tuple, list)) and len(src) == 2:
            k, chi = _validate_spectrum(src[0], src[1])
        else:
            k, chi = read_chi(src)
        spectra.append((k, chi))

    if not spectra:
        raise ValueError("no chi(k) sources to average")

    low = max(0.0, max(k[0] for k, _ in spectra))
    high = min(kmax, min(k[-1] for k, _ in spectra))
    start = int(np.ceil(low / kstep - 1e-10))
    stop = int(np.floor(high / kstep + 1e-10))
    if stop <= start:
        raise ValueError("spectra have no shared k range on the requested grid")
    grid = np.arange(start, stop + 1) * kstep
    arr = np.asarray([np.interp(grid, k, chi) for k, chi in spectra])
    n = arr.shape[0]
    std = arr.std(axis=0)
    mean = arr.mean(axis=0)
    sem = np.full(grid.shape, np.nan)
    n_structures = 0
    means = None
    if structure_ids is not None:
        ids = list(structure_ids)
        if len(ids) != n:
            raise ValueError("structure_ids must have one entry per spectrum")
        groups = list(dict.fromkeys(ids))
        means = np.asarray([arr[[i for i, sid in enumerate(ids) if sid == group]].mean(axis=0)
                            for group in groups])
        n_structures = len(means)
        mean = means.mean(axis=0)
        if n_structures > 1:
            sem = means.std(axis=0, ddof=1) / np.sqrt(n_structures)
    return {
        "k": grid,
        "chi": mean,
        "chi_std": std,
        "chi_sem": sem,
        "n_sites": n,
        "n_structures": n_structures,
        "chi_per_structure": means,
    }


# ── k -> R ───────────────────────────────────────────────────────────────

def ft_window(k, kmin=3.0, kmax=None, dk=1.0):
    """Hanning window with a ``dk``-wide taper at each end.

    The standard EXAFS window: zero below ``kmin - dk/2``, rising as a
    raised cosine to one at ``kmin + dk/2``, flat, then falling back to
    zero at ``kmax + dk/2``.
    """
    k = np.asarray(k, dtype=float)
    kmax = float(k.max()) if kmax is None else float(kmax)

    x1, x2 = kmin - dk / 2.0, kmin + dk / 2.0
    x3, x4 = kmax - dk / 2.0, kmax + dk / 2.0

    win = np.zeros_like(k)
    win[(k >= x2) & (k <= x3)] = 1.0

    rise = (k > x1) & (k < x2)
    if x2 > x1:
        win[rise] = 0.5 * (1.0 - np.cos(np.pi * (k[rise] - x1) / (x2 - x1)))

    fall = (k > x3) & (k < x4)
    if x4 > x3:
        win[fall] = 0.5 * (1.0 + np.cos(np.pi * (k[fall] - x3) / (x4 - x3)))

    return win


def chi_fourier_transform(k, chi, kweight=2, kmin=3.0, kmax=None, dk=1.0,
                          kstep=0.05, nfft=2048):
    """Fourier transform k^w chi(k) to R space.

    The standard EXAFS forward transform, in the convention Athena and
    Larch use, so that ``|chi(R)|`` peaks can be read off against
    published figures::

        chi(R) = (kstep / sqrt(pi)) * FFT[ k^w chi(k) W(k) ]

    Note that peaks sit below the true interatomic distance by roughly
    0.2-0.5 A because the scattering phase shift is not removed -- this
    is the usual "uncorrected R" of an EXAFS figure, not an error.

    Parameters
    ----------
    k, chi : array_like
        chi(k) on any grid; it is re-interpolated onto a uniform one.
    kweight : int
        k-weighting exponent (2 is conventional).
    kmin, kmax, dk : float
        Window range and taper width (A^-1). ``kmax`` defaults to the
        largest k present, and is pulled in to ``k.max() - dk/2`` when it
        lies beyond that, so the whole taper falls inside the data.
    kstep : float
        Uniform grid spacing used for the transform.
    nfft : int
        FFT length (zero-padded).

    Returns
    -------
    dict
        ``{"r", "chir_mag", "chir_re", "chir_im", "window", "k", "kwin",
        "kmin", "kmax", "dk"}`` -- ``window``, ``k`` and ``kwin`` let the
        window actually used be plotted, and ``kmin``/``kmax``/``dk`` say
        what it was (``kmax`` after any clamping).
    """
    k, chi = _validate_spectrum(k, chi)
    if (not np.isfinite([kmin, dk, kstep, kweight]).all()
            or dk < 0 or kstep <= 0 or kweight < 0
            or not isinstance(nfft, (int, np.integer)) or nfft < 2):
        raise ValueError("invalid Fourier-transform grid or window settings")
    kmax = float(k.max()) if kmax is None else float(kmax)
    if not np.isfinite(kmax):
        raise ValueError("kmax must be finite")
    kmin = max(kmin, float(k.min()) + dk / 2.0)
    # The taper is centred on kmax and runs dk/2 past it. Beyond the last
    # data point there is nothing left to taper, so the signal would be cut
    # off mid-fall -- where k^w weighting is largest -- and ring in R.
    kmax = min(kmax, float(k.max()) - dk / 2.0)
    if kmax <= kmin:
        raise ValueError(
            f"FT window is empty: kmax {kmax:.2f} <= kmin {kmin:.2f} "
            f"(data end at k = {float(k.max()):.2f}, dk = {dk})")

    grid = np.arange(0.0, k.max() + 0.5 * kstep, kstep)
    chi_u = np.interp(grid, k, chi, left=0.0, right=0.0)
    win = ft_window(grid, kmin=kmin, kmax=kmax, dk=dk)

    if len(grid) > nfft:
        raise ValueError("nfft is too small for the spectrum; increase nfft")
    n = len(grid)
    buf = np.zeros(nfft, dtype=complex)
    buf[:n] = chi_u[:n] * win[:n] * grid[:n] ** kweight

    out = np.fft.fft(buf)[: nfft // 2] * kstep / np.sqrt(np.pi)
    rstep = np.pi / (kstep * nfft)
    r = rstep * np.arange(nfft // 2)

    return {
        "r": r,
        "chir_mag": np.abs(out),
        "chir_re": out.real,
        "chir_im": out.imag,
        "window": win,
        "k": grid,
        "kwin": chi_u * win * grid ** kweight,
        "kmin": float(kmin),
        "kmax": kmax,
        "dk": float(dk),
    }


# ── end to end ───────────────────────────────────────────────────────────

def _check_output(job_dir, job, filename, trust_external):
    fingerprint = _input_hash(job_dir, job)
    path = job_dir / "run_status.json"
    if not path.exists():
        if not trust_external:
            raise ValueError(
                f"{job_dir}: output has no run provenance; rerun with run_feff_jobs "
                "or explicitly import using trust_external=True "
                "(CLI: --feff-trust-external)")
        # Explicit adoption records the current files, not a claim that their
        # historical calculation has been independently verified.
        outputs = {}
        for name in ("chi.dat", "xmu.dat"):
            if (job_dir / name).exists():
                read_chi(job_dir / name)
                outputs[name] = _sha256(job_dir / name)
        status = {"state": "complete", "input_sha256": fingerprint,
                  "outputs": outputs, "external_import": True}
        _write_status(job_dir, status)
    else:
        status = json.loads(path.read_text())
    if status.get("state") != "complete":
        raise ValueError(f"{job_dir}: run is {status.get('state', 'unknown')}; output is unusable")
    if status.get("input_sha256") != fingerprint:
        raise ValueError(f"{job_dir}: output belongs to a different feff.inp")
    if status.get("outputs", {}).get(filename) != _sha256(job_dir / filename):
        raise ValueError(f"{job_dir}: {filename} differs from the recorded output")
    return bool(status.get("external_import"))


def collect_feff_results(jobs_dir, filename="chi.dat", kmax=20.0, kstep=0.05,
                         kweight=2, kmin=3.0, dk=1.0, rmax=6.0,
                         trust_external=False):
    """Average every finished job in a tree and transform to R space.

    Managed output must match recorded input/output fingerprints. Output run
    externally requires explicit trust_external=True to adopt the current
    files; this cannot verify their historical correspondence. Failed or
    altered managed output is never accepted by that option.

    Jobs with no output are reported rather than silently dropped: a
    partially finished array would otherwise bias the average towards
    whichever sites happened to converge.

    Returns
    -------
    dict
        ``{"k", "chi", "chi_std", "chi_sem", "n_sites", "n_structures",
        "chi_per_structure", "r", "chir_mag", "chir_re", "chir_im",
        "absorber", "edge", "n_jobs", "missing", ...}`` where ``missing``
        lists the job directories with no output file and
        ``chi_per_structure`` holds the site-averaged curve of each
        structure (used for the bootstrap bands of :func:`plot_feff`).
    """
    if filename not in _COLUMNS:
        raise ValueError("filename must be chi.dat or xmu.dat")
    if not np.isfinite(rmax) or rmax <= 0:
        raise ValueError("rmax must be positive and finite")
    manifest = read_manifest(jobs_dir)

    found, missing, structure_ids, external = [], [], [], []
    for job in manifest["jobs"]:
        job_dir = _job_path(jobs_dir, job)
        _input_hash(job_dir, job)
        if (job_dir / ".amorphgen_feff.lock").exists():
            raise ValueError(f"{job_dir}: calculation is currently running")
        path = job_dir / filename
        if path.exists():
            if _check_output(job_dir, job, filename, trust_external):
                external.append(job["dir"])
            found.append(path)
            structure_ids.append(job["structure"])
        else:
            missing.append(job["dir"])

    if not found:
        raise FileNotFoundError(
            f"no {filename} under {jobs_dir!r}: none of the "
            f"{manifest['n_jobs']} jobs have run yet.")

    # Keep enough data for the full upper taper. kmax denotes its centre,
    # consistent with chi_fourier_transform; coverage may still clamp it.
    avg = average_chi(found, kmax=kmax + dk / 2.0, kstep=kstep,
                      structure_ids=structure_ids)
    ft = chi_fourier_transform(avg["k"], avg["chi"], kweight=kweight,
                               kmin=kmin, kmax=kmax, dk=dk, kstep=kstep)

    keep = ft["r"] <= rmax
    avg.update({
        "r": ft["r"][keep],
        "chir_mag": ft["chir_mag"][keep],
        "chir_re": ft["chir_re"][keep],
        "chir_im": ft["chir_im"][keep],
        "absorber": manifest["absorber"],
        "edge": manifest["edge"],
        "n_jobs": manifest["n_jobs"],
        "missing": missing,
        "external_imports": external,
        "n_structures_requested": manifest["n_structures"],
        "kweight": kweight,
        "ft_kmin": ft["kmin"],
        "ft_kmax": ft["kmax"],
        "ft_dk": ft["dk"],
    })
    return avg


def format_feff_report(result):
    """Human-readable summary for :func:`collect_feff_results` output."""
    bar = "=" * 78
    lines = ["", bar,
             f"  Configurationally averaged EXAFS  "
             f"({result['absorber']} {result['edge']} edge)",
             bar,
             f"  Sites averaged : {result['n_sites']} of {result['n_jobs']}"]

    if result["missing"]:
        lines.append(f"  Not yet run    : {len(result['missing'])} "
                     f"(e.g. {result['missing'][0]})")
        lines.append("  The average covers finished jobs only; a partly "
                     "finished array biases it.")

    lines.append(f"  Structures     : {result['n_structures']} contributing; "
                 "equal weight per structure")
    if result.get("external_imports"):
        lines.append(f"  External imports: {len(result['external_imports'])}; "
                     "historical input/output correspondence trusted by caller")
    lines.append("  SEM assumes independent structures; correlated trajectory "
                 "frames require blocking.")
    k, chi, sem = result["k"], result["chi"], result["chi_sem"]
    band = (k >= 3.0) & (k <= 12.0)
    if not np.isfinite(sem).all():
        lines.append("  Ensemble SEM unavailable: need at least two independent structures.")
    elif band.any():
        rms = float(np.sqrt(np.mean(chi[band] ** 2)))
        mean_sem = float(np.mean(sem[band]))
        lines.append(f"  RMS chi        : {rms:.5f} over k = 3-12 A^-1")
        if rms > 0:
            pct = 100.0 * mean_sem / rms
            lines.append(f"  Mean std error : {mean_sem:.5f} "
                         f"({pct:.1f}% of RMS chi)")
            if pct > 10.0:
                lines.append("  The average is not converged across structures; "
                             "sample more independent configurations.")
        else:
            lines.append(f"  Mean std error : {mean_sem:.5f}")

    peak = int(np.argmax(result["chir_mag"]))
    lines.append(f"  |chi(R)| peak  : {result['r'][peak]:.2f} A "
                 f"(k^{result['kweight']} weighted, phase-uncorrected --")
    lines.append("                   0.2-0.5 A below the true distance, as "
                 "in any EXAFS figure)")
    if "ft_kmax" in result:
        # The k-range sets the peak widths in R. State it, or a reader
        # comparing against a published figure on a shorter range sees
        # narrower peaks with no way to tell why.
        lines.append(f"  FT window      : k = {result['ft_kmin']:.1f}-"
                     f"{result['ft_kmax']:.1f} A^-1 (Hanning, dk = "
                     f"{result['ft_dk']:.1f}); match a published figure's")
        lines.append("                   k-range before comparing peak "
                     "widths or heights")
    lines.append(bar)
    return "\n".join(lines)


def feff_uncertainty(result, confidence=0.95, n_bootstrap=1000, seed=0):
    """Whole-structure bootstrap bands for the two plotted curves.

    Structures are the sampling unit, as in the other ensemble analyses
    (:mod:`amorphgen.analysis.uncertainty`). ``k^w chi(k)`` is linear in the
    structure means, so :func:`~amorphgen.analysis.uncertainty.summarize_structures`
    applies directly. ``|chi(R)|`` is not linear: each bootstrap draw of
    structures is averaged in k, transformed with the collected window and
    then reduced to its magnitude, so the band describes the plotted curve.
    A magnitude is not linear in the mean, so with few structures the
    percentile band can sit above the curve near a peak.

    Returns
    -------
    tuple
        ``(chik, chir)``. ``chik`` is a ``summarize_structures`` summary of
        ``k^w chi(k)``; ``chir`` holds pointwise ``bootstrap_low`` /
        ``bootstrap_high`` for ``|chi(R)|``. Either is None when fewer than
        two structures contribute or the transform window is unknown.
    """
    per = result.get("chi_per_structure")
    if per is None or len(per) < 2:
        return None, None
    from .uncertainty import summarize_structures
    k = np.asarray(result["k"], dtype=float)
    per = np.asarray(per, dtype=float)
    kw = result["kweight"]
    chik = summarize_structures(per * k ** kw, confidence=confidence,
                                n_bootstrap=n_bootstrap, seed=seed)
    if not all(key in result for key in ("ft_kmin", "ft_kmax", "ft_dk")):
        return chik, None
    kstep = float(np.round(k[1] - k[0], 10))
    n_r = len(result["r"])
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, n_r))
    for i in range(n_bootstrap):
        mean = per[rng.integers(0, len(per), len(per))].mean(axis=0)
        draws[i] = chi_fourier_transform(
            k, mean, kweight=kw, kmin=result["ft_kmin"], kmax=result["ft_kmax"],
            dk=result["ft_dk"], kstep=kstep)["chir_mag"][:n_r]
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail], axis=0)
    chir = {"bootstrap_low": low.tolist(), "bootstrap_high": high.tolist(),
            "confidence": float(confidence), "n_bootstrap": int(n_bootstrap),
            "seed": int(seed), "n_structures": int(len(per)),
            "sampling_unit": "structure", "bootstrap_method": "percentile",
            "band_type": "pointwise"}
    return chik, chir


def _feff_csv_header(result):
    """The parameter line written above each CSV header, as plot_tr does."""
    return (f"# absorber={result['absorber']} edge={result['edge']} "
            f"n_sites={result['n_sites']} n_structures={result['n_structures']} "
            f"kweight={result['kweight']} window=hanning "
            f"ft_kmin={result.get('ft_kmin', '')} ft_kmax={result.get('ft_kmax', '')} "
            f"ft_dk={result.get('ft_dk', '')} R=phase-uncorrected")


def _write_feff_csvs(result, k_path, r_path, chir_band=None):
    """chi(k) and chi(R) tables; NaN is written as an empty cell."""
    import csv
    kw = result["kweight"]
    cell = lambda v: f"{v:.8e}" if np.isfinite(v) else ""
    k = np.asarray(result["k"], dtype=float)
    chi = np.asarray(result["chi"], dtype=float)
    with open(k_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([_feff_csv_header(result)])
        w.writerow(["k_invA", "chi", f"k{kw}chi_invA{kw}", "chi_site_std",
                    "chi_structure_sem"])
        for row in zip(k, chi, chi * k ** kw, result["chi_std"], result["chi_sem"]):
            w.writerow([f"{row[0]:.4f}", *map(cell, row[1:])])
    unit = f"invA{kw + 1}"
    columns = [result["r"], result["chir_mag"], result["chir_re"], result["chir_im"]]
    header = ["R_A", f"chir_mag_{unit}", f"chir_re_{unit}", f"chir_im_{unit}"]
    if chir_band is not None:
        columns += [chir_band["bootstrap_low"], chir_band["bootstrap_high"]]
        header += ["chir_mag_bootstrap_low", "chir_mag_bootstrap_high"]
    with open(r_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([_feff_csv_header(result)])
        w.writerow(header)
        for row in zip(*columns):
            w.writerow([f"{row[0]:.4f}", *map(cell, row[1:])])


def write_feff_csv(result, filepath):
    """Write the averaged chi(k) and ``|chi(R)|`` to two CSV files.

    ``<filepath>`` takes chi(k); ``<stem>_chir.csv`` takes chi(R), since
    the two live on different grids. The tables are those
    :func:`plot_feff` writes beside its figures, including the
    whole-structure bootstrap band of ``|chi(R)|`` when at least two
    structures contribute.
    """
    stem, ext = os.path.splitext(filepath)
    r_path = f"{stem}_chir{ext or '.csv'}"
    _write_feff_csvs(result, filepath, r_path, feff_uncertainty(result)[1])
    return [filepath, r_path]


def plot_feff(result, output_dir=".", prefix="analysis", dpi=300,
              save_pdf=False, show_title=False, save_csv=True):
    """Plot the ensemble EXAFS and write the numbers beside the figures.

    Same conventions as the other analysis plots: one standalone figure
    per curve, saved without touching pyplot, with a whole-structure
    bootstrap band (:func:`feff_uncertainty`) and a CSV of the raw numbers.

    Writes ``{prefix}_feff_chik.png`` (``k^w chi(k)``) and
    ``{prefix}_feff_chir.png`` (``|chi(R)|``, phase-uncorrected), plus
    ``.pdf`` with ``save_pdf``. With ``save_csv`` also ``{base}.csv`` for
    each, and ``{prefix}_feff_chik_uncertainty.{csv,json}`` and
    ``_per_structure.csv`` when at least two structures contribute.
    """
    from .plotting import (_PALETTE, _apply_pub_style, _draw_uncertainty,
                           _figure, _save_curve_uncertainty, _save_fig)

    os.makedirs(output_dir, exist_ok=True)
    kw = result["kweight"]
    k = np.asarray(result["k"], dtype=float)
    r = np.asarray(result["r"], dtype=float)
    chik_band, chir_band = feff_uncertainty(result)
    title = (f"{result['absorber']} {result['edge']}-edge EXAFS — "
             f"{result['n_sites']} sites, {result['n_structures']} structures")

    fig, ax = _figure(figsize=(5.4, 4.0))
    ax.plot(k, np.asarray(result["chi"], dtype=float) * k ** kw, lw=1.4,
            color=_PALETTE[0])
    if _draw_uncertainty(ax, k, chik_band, _PALETTE[0]):
        ax.legend(frameon=False, fontsize=8)
    ax.axhline(0.0, ls=":", color="grey", alpha=0.6)
    ax.set_xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(rf"$k^{{{kw}}}\chi(k)$ ($\mathrm{{\AA}}^{{-{kw}}}$)")
    ax.set_xlim(k.min(), k.max())
    _apply_pub_style(ax)
    if show_title:
        ax.set_title(title)
    fig.tight_layout()
    k_base = os.path.join(output_dir, f"{prefix}_feff_chik")
    _save_fig(fig, k_base, dpi, save_pdf)

    fig, ax = _figure(figsize=(5.4, 4.0))
    ax.plot(r, result["chir_mag"], lw=1.4, color=_PALETTE[0])
    if _draw_uncertainty(ax, r, chir_band, _PALETTE[0]):
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(r"$R$ ($\mathrm{\AA}$), phase-uncorrected")
    ax.set_ylabel(rf"$|\chi(R)|$ ($\mathrm{{\AA}}^{{-{kw + 1}}}$)")
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(bottom=0.0)
    _apply_pub_style(ax)
    if show_title:
        window = (f", k = {result['ft_kmin']:g}–{result['ft_kmax']:g} " r"$\mathrm{\AA}^{-1}$"
                  if "ft_kmin" in result else "")
        ax.set_title(title + window)
    fig.tight_layout()
    r_base = os.path.join(output_dir, f"{prefix}_feff_chir")
    _save_fig(fig, r_base, dpi, save_pdf)

    if save_csv:
        _write_feff_csvs(result, f"{k_base}.csv", f"{r_base}.csv", chir_band)
        print(f"  Saved: {k_base}.csv")
        print(f"  Saved: {r_base}.csv")
        _save_curve_uncertainty(k_base, k, {f"k{kw}chi": chik_band})


def feff_pipeline(atoms_list, absorbers, output_dir, *, executable="feff",
                  workers=1, resume=True, timeout=None, n_sites=None,
                  radius=DEFAULT_RADIUS, edge="K", stoichiometry="cluster",
                  cards=None, seed=0, kweight=2, kmin=3.0, kmax=20.0, dk=1.0,
                  plot_dir=None, prefix="analysis", dpi=300, save_pdf=False,
                  show_title=False):
    """From structures to averaged spectra in one call.

    For each absorbing element: write the job tree (:func:`write_feff_jobs`),
    run FEFF locally (:func:`run_feff_jobs`), average and transform the
    output (:func:`collect_feff_results`) and, with ``plot_dir``, save the
    figures and CSV tables (:func:`plot_feff`) as ``{prefix}_{El}_feff_*``.

    One absorber uses ``output_dir`` itself as the job tree; several use
    ``output_dir/<El>``. With ``resume`` (default), jobs that already
    finished with matching fingerprints are not rerun, so an interrupted
    pipeline can simply be started again. For a cluster, write the trees,
    run their ``joblist.txt`` as an array and collect afterwards instead.

    A tree with any failed job is not averaged -- a partial average would
    lean towards whichever sites happened to run -- so its ``result`` is
    None and the reasons are in ``run["failed"]``.

    Parameters
    ----------
    atoms_list : list of ase.Atoms
        The ensemble.
    absorbers : str or list of str
        Absorbing elements, e.g. ``"Ga"``, ``["In", "Ga", "Zn"]`` or
        ``"In,Ga,Zn"``.
    output_dir : str
        Where the job tree(s) go.
    executable, workers, resume, timeout
        Passed to :func:`run_feff_jobs` (``resume`` as ``skip_complete``).
    n_sites, radius, edge, stoichiometry, cards, seed
        Passed to :func:`write_feff_jobs`.
    kweight, kmin, kmax, dk
        Passed to :func:`collect_feff_results`.
    plot_dir, prefix, dpi, save_pdf, show_title
        Passed to :func:`plot_feff`; nothing is plotted without ``plot_dir``.

    Returns
    -------
    dict
        ``{El: {"jobs_dir", "n_jobs", "run", "result", "report"}}`` in the
        order given.
    """
    if isinstance(absorbers, str):
        absorbers = absorbers.split(",")
    absorbers = list(dict.fromkeys(a.strip() for a in absorbers if a.strip()))
    if not absorbers:
        raise ValueError("no absorbing element given")
    out = {}
    for element in absorbers:
        tree = (output_dir if len(absorbers) == 1
                else os.path.join(output_dir, element))
        manifest = write_feff_jobs(atoms_list, element, tree, n_sites=n_sites,
                                   radius=radius, edge=edge, seed=seed,
                                   cards=cards, stoichiometry=stoichiometry)
        run = run_feff_jobs(tree, executable=executable, timeout=timeout,
                            workers=workers, skip_complete=resume)
        entry = {"jobs_dir": tree, "n_jobs": manifest["n_jobs"], "run": run,
                 "result": None, "report": None}
        if not run["failed"]:
            result = collect_feff_results(tree, kweight=kweight, kmin=kmin,
                                          kmax=kmax, dk=dk)
            entry.update(result=result, report=format_feff_report(result))
            if plot_dir is not None:
                plot_feff(result, output_dir=plot_dir,
                          prefix=f"{prefix}_{element}", dpi=dpi,
                          save_pdf=save_pdf, show_title=show_title)
        out[element] = entry
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run provenance-tracked FEFF jobs")
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--job", action="append", help="Job directory name (repeatable)")
    parser.add_argument("--executable", default="feff")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--workers", type=int, default=1,
                        help="Jobs run at once (default 1)")
    parser.add_argument("--skip-complete", action="store_true",
                        help="Leave jobs already completed with matching fingerprints")
    options = parser.parse_args()
    outcome = run_feff_jobs(options.jobs_dir, executable=options.executable,
                            jobs=options.job, timeout=options.timeout,
                            workers=options.workers,
                            skip_complete=options.skip_complete)
    print(json.dumps(outcome, indent=2))
    sys.exit(1 if outcome["failed"] else 0)
