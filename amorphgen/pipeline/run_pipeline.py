"""
amorphgen.pipeline.run_pipeline
--------------------------------
Orchestrates the full melt-and-quench pipeline.

Usage
-----
    from amorphgen import MeltQuenchPipeline

    # Default MACE model
    pipe = MeltQuenchPipeline(
        input_file="POSCAR",
        work_dir="my_run",
        cfg_override={"model": "mace-mpa-0", "device": "cuda"},
    )
    final_atoms = pipe.run()

    )

    # CHGNet
    pipe = MeltQuenchPipeline(
        input_file="POSCAR",
        cfg_override={"model": "chgnet"},
    )

    # Custom fine-tuned MACE model
    pipe = MeltQuenchPipeline(
        input_file="POSCAR",
        cfg_override={"model_path": "/data/models/InO_finetuned.model"},
    )

Resuming from a checkpoint
--------------------------
    pipe.run(stages=[5, 6, 7], input_file="stage4_eq.xyz")
"""

import os
import time
import platform
import sys
import hashlib
from copy import deepcopy
from datetime import datetime

from ase.io import read

from . import opt_cell, melt_cell, equilibrate, quench, final_opt
from ..utils import get_calculator, merge_config
from ..configs import DEFAULT_CONFIG
from .manifest import RunManifest, _json_value
from ..utils.run_lock import run_lock


def _calculator_parameters(calc, seen=None):
    """Capture known calculator and wrapper parameters without runtime state."""
    seen = set() if seen is None else seen
    if id(calc) in seen:
        return None
    seen.add(id(calc))
    parameters = {
        key: _json_value(getattr(calc, key, None))
        for key in ("parameters", "pair_params", "charges", "cutoff",
                    "coulomb_method", "alpha", "coulomb", "repulsive_core_config")
    }
    base = getattr(calc, "base_calculator", None)
    if base is not None:
        parameters["base_calculator"] = _calculator_parameters(base, seen)
    return parameters


def _resume_config_value(value):
    """Include identities of live reference calculators in nested settings."""
    from ase.calculators.calculator import Calculator
    from ..utils.run_provenance import calculator_provenance

    if isinstance(value, dict):
        return {str(k): _resume_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resume_config_value(v) for v in value]
    if isinstance(value, Calculator):
        model = calculator_provenance({}, value, injected=True)["model"]
        model.pop("hash_unavailable_reason", None)
        return {"model": model, "parameters": _calculator_parameters(value)}
    return _json_value(value)


class MeltQuenchPipeline:
    """
    End-to-end melt-and-quench pipeline for amorphous structure generation.

    Parameters
    ----------
    input_file : str
        Path to the starting crystalline structure (any ASE-readable format).
    work_dir : str
        Directory where all output files are written.  Created if absent.
    cfg_override : dict, optional
        Any keys in DEFAULT_CONFIG to override, including:

        * ``"model"``      : foundation model short name (any backend)
        * ``"model_path"`` : path to local .model file (overrides model)
        * ``"device"``     : ``"cuda"`` or ``"cpu"``
        * ``"eq_premelt"`` : ``{"ensemble": "NVT" or "NPT", ...}``
        * ``"melt"``       : ``{"ensemble": "NVT" or "NPT", ...}``
        * ``"quench"``     : ``{"ensemble": "NVT" or "NPT", ...}``
        * ``"eq_high"``    : ``{"ensemble": "NVT" or "NPT", ...}``
        * ``"eq_low"``     : ``{"ensemble": "NVT" or "NPT", ...}``

    share_calc : bool
        If True, one calculator is shared across all stages.
    calc : ASE calculator, optional
        A pre-built ASE calculator to use for every stage, bypassing the
        ``get_calculator()`` backend factory. Lets you drive the full pipeline
        with any ASE calculator (a fine-tuned model, a custom potential, an
        external code). NPT and cell-filter stages still require the
        calculator to provide a stress tensor.
    """

    STAGE_NAMES = {
        1: "Structure optimisation (crystalline)",
        2: "Pre-melt equilibration (300 K)",
        3: "Melt (heat ramp)",
        4: "High-T equilibration",
        5: "Quench (cooling ramp)",
        6: "Low-T equilibration",
        7: "Final optimisation (amorphous)",
    }

    # Default checkpoint files written by each stage.
    STAGE_CHECKPOINTS = {
        1: "stage1_opt.xyz",
        2: "stage2_eq.xyz",
        3: "stage3_melted.xyz",
        4: "stage4_eq.xyz",
        5: "stage5_quenched.xyz",
        6: "stage6_eq.xyz",
        7: "stage7_opt.xyz",
    }

    def __init__(self, input_file: str,
                 work_dir: str = "melt_quench_run",
                 cfg_override: dict | None = None,
                 share_calc: bool = True,
                 calc=None):

        self.input_file = input_file
        self.work_dir   = work_dir
        self.cfg        = merge_config(DEFAULT_CONFIG, cfg_override)
        self.share_calc = share_calc
        # An explicit ASE calculator supplied here is used for every stage,
        # bypassing the get_calculator() factory. This lets callers drive the
        # full pipeline with any ASE calculator (a fine-tuned model, a custom
        # potential, an external code) rather than only the built-in backends.
        # A stress-less calculator will still be rejected by the NPT / cell-
        # filter guards where a stress tensor is required.
        self._injected_calc = calc
        self._calc      = calc
        self._calc_provenance = None
        self._calc_config = None
        os.makedirs(work_dir, exist_ok=True)

        # Handle legacy "mace_model" key → "model"
        if self.cfg.get("mace_model") and not self.cfg.get("model"):
            self.cfg["model"] = self.cfg["mace_model"]

    # ─────────────────────────────────────────────────────────────────────────

    def _get_calc(self):
        """Build or return the shared calculator.

        An injected calculator (passed to the constructor) is always reused
        as-is; otherwise one is built from the config via get_calculator().
        """
        if self._injected_calc is not None:
            return self._injected_calc
        calculator_config = self._calculator_config()
        if self._calc is None or not self.share_calc or self._calc_config != calculator_config:
            from ..utils.common import resolve_device
            device = resolve_device(self.cfg.get("device", "cuda"))

            calc_kwargs = {}
            if self.cfg.get("classical_params"):
                calc_kwargs["classical_params"] = self.cfg["classical_params"]
            self._calc = get_calculator(
                model=self.cfg.get("model", "mace-mpa-0"),
                device=device,
                model_path=self.cfg.get("model_path"),
                default_dtype=self.cfg.get("default_dtype", "auto"),
                **calc_kwargs,
            )
            from ..utils.run_provenance import calculator_provenance
            # Keep the identity of the weights actually loaded, even if the
            # checkpoint file is replaced before this calculator is reused.
            self._calc_provenance = calculator_provenance(self.cfg, self._calc)
            self._calc_config = calculator_config
        return self._calc

    def _calculator_config(self):
        return _json_value({key: self.cfg.get(key) for key in (
            "model", "model_path", "device", "default_dtype", "classical_params",
        )})

    # ─────────────────────────────────────────────────────────────────────────

    def _find_resume_point(self, stages: list[int], previous_stages=None) -> tuple[list[int], str]:
        """
        Scan work_dir for completed stage checkpoints and return the
        remaining stages and the input file to resume from.

        Parameters
        ----------
        stages : list of int
            The originally requested stages.

        Returns
        -------
        remaining_stages : list of int
            Stages that still need to run.
        resume_input : str
            Path to the checkpoint file to resume from, or the original
            input_file if no checkpoints are found.
        """
        resume_input = self.input_file
        remaining_stages = list(stages)
        checkpoints = self._stage_checkpoints()

        # Walk stages in order; if a checkpoint exists, advance past it
        for s in stages:
            if previous_stages is not None and previous_stages.get(s) not in ("completed", "skipped"):
                break
            checkpoint = checkpoints.get(s)
            if checkpoint is None:
                break
            checkpoint_path = os.path.join(self.work_dir, checkpoint)
            if os.path.isfile(checkpoint_path):
                # A truncated output is not a completed checkpoint.
                try:
                    read(checkpoint_path)
                except Exception:
                    break
                resume_input = checkpoint_path
                remaining_stages.remove(s)
            else:
                break  # first missing checkpoint → start here

        return remaining_stages, resume_input

    def _stage_checkpoints(self):
        sections = {1: "opt", 2: "eq_premelt", 3: "melt", 4: "eq_high",
                    5: "quench", 6: "eq_low", 7: "final_opt"}
        checkpoints = {}
        for stage, section in sections.items():
            cfg = dict(self.cfg.get("opt", {})) if stage == 7 else {}
            cfg.update(self.cfg.get(section, {}))
            checkpoints[stage] = cfg.get("output_xyz", self.STAGE_CHECKPOINTS[stage])
        return checkpoints

    def _clear_stale_trajectory(self, stage):
        """Reset fresh-stage frame files before recording the stage as started.

        Stage setup can fail before its output writer truncates a trajectory.
        Clearing here prevents that failure from adopting an earlier run's frames.
        """
        defaults = {2: ("eq_premelt", "stage2_eq_traj.xyz"),
                    3: ("melt", "stage3_melt_traj.xyz"),
                    4: ("eq_high", "stage4_eq_traj.xyz"),
                    5: ("quench", "stage5_quench_traj.xyz"),
                    6: ("eq_low", "stage6_eq_traj.xyz")}
        if stage not in defaults:
            return
        section, default = defaults[stage]
        paths = {self.cfg[section].get("traj_file", default)}
        if stage in (3, 5):
            paths.add("stage3_melt.xyz" if stage == 3 else "stage5_quench.xyz")
        for path in paths:
            if os.path.isfile(path):
                os.unlink(path)

    def _resume_settings(self, stages, input_file):
        """Snapshot the requested protocol before any output can be reused."""
        from ..utils.common import run_index_for
        from ..utils.run_provenance import calculator_provenance

        input_path = os.path.abspath(input_file)
        digest = None
        if os.path.isfile(input_path):
            hasher = hashlib.sha256()
            with open(input_path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        provenance = calculator_provenance(
            self.cfg, self._injected_calc, injected=self._injected_calc is not None,
        )
        model = provenance["model"]
        identity = {key: model[key] for key in (
            "name", "path", "sha256", "hash_source", "calculator_class",
        )}
        # A shared calculator may still hold older weights after its file
        # was replaced. Keep requested and loaded file identities separate.
        requested_model = dict(identity)
        if (self._injected_calc is None and self.share_calc
                and self._calc_config == self._calculator_config()
                and self._calc_provenance is not None):
            loaded = self._calc_provenance["model"]
            if loaded["hash_source"] == "file":
                for key in ("name", "path", "sha256", "hash_source"):
                    identity[key] = loaded[key]
        if self._injected_calc is not None:
            # ASE parameters cover classical/custom calculators that do not
            # expose model weights. AmorphGen pair potentials keep their
            # parameters as attributes rather than ASE's parameters dict.
            identity["parameters"] = _calculator_parameters(self._injected_calc)
        orig_dir = os.getcwd()
        try:
            os.chdir(self.work_dir)
            seed_index = run_index_for(self.cfg)
        finally:
            os.chdir(orig_dir)
        return {
            "version": 1, "config": _resume_config_value(self.cfg),
            "stages": list(stages), "seed_index": seed_index,
            "input": {"path": input_path, "sha256": digest},
            "calculator": identity,
            "requested_model": requested_model,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def run(self,
            stages: list[int] | None = None,
            input_file: str | None = None,
            resume: bool = False) -> object:
        """
        Execute the pipeline.

        Parameters
        ----------
        stages : list of int, optional
            Which stages to run (default: all, i.e. ``[1, 2, 3, 4, 5, 6, 7]``).
        input_file : str, optional
            Override the input structure file (useful for resuming from a
            mid-pipeline checkpoint).
        resume : bool
            If True, verify the saved input and settings, then scan work_dir
            for valid checkpoints of stages recorded as completed. Automatically
            determines which stage to resume from and which input file
            to use.

        Returns
        -------
        ase.Atoms
            The final optimised amorphous structure.

        Notes
        -----
        ``run_manifest.json`` records configuration, calculator provenance,
        and stage outcomes before and during execution. Each accepted invocation adds
        an attempt, retaining earlier attempts when resuming or rerunning.
        """
        with run_lock(self.work_dir):
            return self._run_locked(stages, input_file, resume)

    def _run_locked(self, stages, input_file, resume):
        from ..utils.run_provenance import calculator_provenance

        if stages is None:
            stages = [1, 2, 3, 4, 5, 6, 7]
        stages = list(stages)
        if input_file is None:
            input_file = self.input_file

        settings = self._resume_settings(stages, input_file)
        manifest = RunManifest(self.work_dir, input_file, self.cfg,
                               stages, self.STAGE_NAMES, resume, settings)
        try:
            manifest.attempt.update(calculator_provenance(
                self.cfg, self._injected_calc,
                injected=self._injected_calc is not None,
            ))
            manifest.attempt["seed_index"] = settings["seed_index"]
            manifest.save()
            atoms = self._run(stages, input_file, resume, manifest)
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            try:
                manifest.finish(status, exc)
            except Exception as write_error:
                # Preserve the simulation failure if disk writes also fail.
                print(f"Could not update {manifest.path}: {write_error}", file=sys.stderr)
            raise
        else:
            manifest.finish("completed")
            return atoms

    def _run(self, stages, input_file, resume, manifest):
        from ..utils.run_provenance import calculator_provenance

        if resume:
            remaining, resume_input = self._find_resume_point(stages, manifest.previous_stages)
            skipped = [s for s in stages if s not in remaining]
            checkpoints = self._stage_checkpoints()
            manifest.skip_stages(skipped, checkpoints)
            if not remaining and stages:
                print(f"  All stages already completed in {self.work_dir}/")
                final_checkpoint = os.path.join(
                    self.work_dir,
                    checkpoints[stages[-1]],
                )
                manifest.attempt["input_file"] = os.path.abspath(final_checkpoint)
                manifest.save()
                return read(final_checkpoint)
            if remaining != stages:
                print(f"  Resuming: skipping completed stages {skipped}")
                print(f"  Starting from stage {remaining[0]} "
                      f"(input: {os.path.basename(resume_input)})")
                input_file = resume_input
            stages = remaining

        manifest.attempt["input_file"] = os.path.abspath(input_file)
        manifest.save()
        atoms = read(input_file)
        calc = self._get_calc()
        atoms.calc = calc
        if self._injected_calc is None:
            provenance = deepcopy(self._calc_provenance)
            if provenance is None:
                provenance = calculator_provenance(self.cfg, calc)
            provenance["precision"]["requested"] = self.cfg.get("default_dtype", "auto")
            provenance["device"]["requested"] = self.cfg.get("device", "auto")
            manifest.attempt.update(provenance)
        manifest.save()

        model_name = self.cfg.get("model", "mace-mpa-0")
        model_path = self.cfg.get("model_path")
        model_display = model_path if model_path else model_name
        device = self.cfg.get("device", "cuda")
        n_atoms = len(atoms)
        formula = atoms.get_chemical_formula(mode="hill")

        bar = "=" * 65
        print(f"\n{bar}")
        from .. import __version__
        print(f"  AmorphGen  v{__version__}  -  Melt-and-Quench Pipeline")
        from ..utils.common import compute_density_gcm3
        density = compute_density_gcm3(atoms)
        print(f"  Model:  {model_display}")
        print(f"  Device: {device}")
        print(f"  Input:  {input_file}")
        print(f"  System: {formula} ({n_atoms} atoms)")
        print(f"  Density: {density:.2f} g/cm3")
        print(f"  Stages: {stages}")
        print(f"  Output: {self.work_dir}/")
        print(f"{bar}\n")

        orig_dir = os.getcwd()
        os.chdir(self.work_dir)
        t0 = time.time()
        stage_timings = []

        try:
            for s in stages:
                from ..utils.preemption import stop_if_requested
                stop_if_requested()
                name = self.STAGE_NAMES.get(s, f"Stage {s}")
                print(f"\n{'-' * 65}")
                print(f"  Stage {s}: {name}")
                print(f"{'-' * 65}\n")

                t_stage = time.time()
                stage_resume = resume and s == stages[0] and s in manifest.previous_stages
                if not stage_resume:
                    self._clear_stale_trajectory(s)
                manifest.start_stage(s)

                # MD stages get the resume flag for FRAME-level resume: an
                # interrupted stage picks up from the last frame of its
                # stage trajectory (stages that never started have no
                # trajectory and run fresh). Optimisation stages (1, 7)
                # restart whole — LBFGS/FIRE state is not checkpointed.
                if s == 1:
                    atoms = opt_cell.run(atoms, self.cfg, calc)
                elif s == 2:
                    atoms = equilibrate.run(atoms, self.cfg, calc,
                                            stage="premelt", resume=stage_resume)
                elif s == 3:
                    atoms = melt_cell.run(atoms, self.cfg, calc, resume=stage_resume)
                elif s == 4:
                    atoms = equilibrate.run(atoms, self.cfg, calc,
                                            stage="high", resume=stage_resume)
                elif s == 5:
                    atoms = quench.run(atoms, self.cfg, calc, resume=stage_resume)
                elif s == 6:
                    atoms = equilibrate.run(atoms, self.cfg, calc,
                                            stage="low", resume=stage_resume)
                elif s == 7:
                    atoms = final_opt.run(atoms, self.cfg, calc)
                else:
                    print(f"  WARNING: Unknown stage {s} - skipping.")
                    manifest.finish_stage("skipped")
                    continue

                dt = time.time() - t_stage
                stage_timings.append((s, name, dt))
                manifest.finish_stage()
                d = compute_density_gcm3(atoms)
                print(f"  [Stage {s} completed in {dt:.1f} s ({dt/60:.1f} min) "
                      f"| density={d:.2f} g/cm3]")

        finally:
            os.chdir(orig_dir)

        elapsed = time.time() - t0

        # Print summary
        print(f"\n{bar}")
        print(f"  Pipeline complete  ({elapsed / 60:.1f} min)")
        print(f"  Output directory:  {self.work_dir}/")
        print(f"{bar}\n")

        # Write pipeline summary log
        self._write_summary_log(
            input_file, formula, n_atoms, model_display, device,
            stages, stage_timings, elapsed
        )

        return atoms

    def _write_summary_log(self, input_file, formula, n_atoms,
                           model_display, device, stages,
                           stage_timings, total_elapsed):
        """Write a pipeline_summary.log file with timing and config."""
        logfile = os.path.join(self.work_dir, "pipeline_summary.log")
        bar = "=" * 65

        with open(logfile, "w") as f:
            f.write(f"{bar}\n")
            f.write(f"  AmorphGen — Pipeline Summary\n")
            f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Platform: {platform.platform()}\n")
            f.write(f"  Python: {platform.python_version()}\n")
            f.write(f"{bar}\n\n")

            f.write(f"  System:  {formula} ({n_atoms} atoms)\n")
            f.write(f"  Model:   {model_display}\n")
            f.write(f"  Device:  {device}\n")
            f.write(f"  Input:   {input_file}\n")
            f.write(f"  Output:  {self.work_dir}/\n")
            f.write(f"  Stages:  {stages}\n\n")

            # Stage timings
            f.write(f"  {'Stage':<45} {'Time (s)':>10} {'Time (min)':>12}\n")
            f.write(f"  {'-'*69}\n")
            for s, name, dt in stage_timings:
                f.write(f"  {s}. {name:<42} {dt:>10.1f} {dt/60:>11.1f}\n")
            f.write(f"  {'-'*69}\n")
            f.write(f"  {'Total':<45} {total_elapsed:>10.1f} "
                    f"{total_elapsed/60:>11.1f}\n\n")

            # Per-atom timing
            if n_atoms > 0:
                f.write(f"  Per-atom total: {total_elapsed/n_atoms:.2f} s/atom\n")
                for s, name, dt in stage_timings:
                    f.write(f"  Per-atom stage {s}: {dt/n_atoms:.2f} s/atom\n")
                f.write(f"\n")

            # Key config parameters
            f.write(f"  Configuration:\n")
            for key in ["opt", "eq_premelt", "melt", "eq_high",
                        "quench", "eq_low"]:
                if key in self.cfg:
                    f.write(f"    {key}:\n")
                    for k, v in self.cfg[key].items():
                        f.write(f"      {k}: {v}\n")

            f.write(f"\n{bar}\n")

        print(f"  Summary log: {logfile}")
