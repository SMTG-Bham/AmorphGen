"""
test/test_imports.py
---------------------
Verify all public API imports work correctly.
"""

import subprocess
import sys
from pathlib import Path


class TestPackageImports:
    """Tier 1: verify the package structure is importable."""

    def test_top_level_import(self):
        import amorphgen
        assert hasattr(amorphgen, "__version__")
        assert isinstance(amorphgen.__version__, str)
        assert amorphgen.__version__

    def test_pipeline_import(self):
        from amorphgen import MeltQuenchPipeline
        assert callable(MeltQuenchPipeline)

    def test_top_level_generate_random(self):
        from amorphgen import generate_random
        assert callable(generate_random)

    def test_top_level_batch_random(self):
        from amorphgen import batch_random
        assert callable(batch_random)

    def test_config_import(self):
        from amorphgen import DEFAULT_CONFIG
        assert isinstance(DEFAULT_CONFIG, dict)
        assert "model" in DEFAULT_CONFIG

    def test_calculator_factory_import(self):
        from amorphgen.utils import get_calculator
        assert callable(get_calculator)

    def test_deprecated_alias_import(self):
        from amorphgen.utils import get_mace_calculator
        assert callable(get_mace_calculator)

    def test_list_models_import(self):
        from amorphgen.utils import list_models
        assert callable(list_models)

    def test_stage_modules_import(self):
        from amorphgen.pipeline import (
            opt_cell, melt_cell, equilibrate, quench,
            final_opt, batch_quench, random_gen,
        )
        for mod in [opt_cell, melt_cell, equilibrate, quench,
                    final_opt, batch_quench, random_gen]:
            assert hasattr(mod, "run") or hasattr(mod, "batch_random")

    def test_utility_imports(self):
        from amorphgen.utils import (
            make_cubic, build_md_dynamics, resolve_ramp,
            MDLogger, TrajectoryWriter, TRAJ_FORMATS,
            attach_outputs, merge_config, extract_snapshots,
        )
        for utility in (make_cubic, build_md_dynamics, resolve_ramp, MDLogger,
                        TrajectoryWriter, attach_outputs, merge_config, extract_snapshots):
            assert callable(utility)
        assert {"extxyz", "traj"} <= TRAJ_FORMATS

    def test_model_registries_import(self):
        from amorphgen.utils import (
            MACE_FOUNDATION_MODELS,
            CHGNET_MODELS,
            SEVENNET_MODELS,
            MODEL_DESCRIPTIONS,
        )
        assert "mace-mpa-0" in MACE_FOUNDATION_MODELS
        assert "mace-mpa-0-medium" in MODEL_DESCRIPTIONS
        assert "chgnet" in CHGNET_MODELS
        assert "sevennet" in SEVENNET_MODELS
        assert "7net-mf-ompa" in SEVENNET_MODELS


def test_public_imports_do_not_load_optional_ml_backends():
    """A fresh interpreter catches eager imports hidden by pytest's module cache."""
    script = """
import importlib.abc
import sys

class BlockMLImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "torch_sim", "mace", "chgnet", "sevenn"}:
            raise AssertionError(f"Unexpected optional backend import: {fullname}")

sys.meta_path.insert(0, BlockMLImports())
import amorphgen
import amorphgen.utils
import amorphgen.analysis
for name in amorphgen.__all__:
    getattr(amorphgen, name)
for name in amorphgen.utils.__all__:
    getattr(amorphgen.utils, name)
for name in amorphgen.analysis.__all__:
    getattr(amorphgen.analysis, name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
