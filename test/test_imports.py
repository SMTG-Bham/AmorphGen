"""
tests/test_imports.py
---------------------
Verify all public API imports work correctly.
"""

import pytest


class TestPackageImports:
    """Tier 1: verify the package structure is importable."""

    def test_top_level_import(self):
        import amorphgen
        assert hasattr(amorphgen, "__version__")
        # Accept any 1.0.0 family version (1.0.0, 1.0.0rc2, 1.0.0.post1, ...)
        # so the test doesn't break on every release-candidate bump.
        assert isinstance(amorphgen.__version__, str)
        assert amorphgen.__version__.startswith("1.0.0")

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

    def test_model_registries_import(self):
        from amorphgen.utils import (
            MACE_FOUNDATION_MODELS,
            CHGNET_MODELS,
            SEVENNET_MODELS,
            MODEL_DESCRIPTIONS,
        )
        assert len(MACE_FOUNDATION_MODELS) > 10
        assert "chgnet" in CHGNET_MODELS
        assert "sevennet" in SEVENNET_MODELS
        assert "7net-mf-ompa" in SEVENNET_MODELS
