"""
amorphgen.analysis
-------------------
Structural analysis tools for amorphous structures.

Usage
-----
    from amorphgen.analysis import StructureAnalyser

    sa = StructureAnalyser("output_dir/", cutoff="auto")
    sa.summary()
    sa.plot(output_dir="plots/")

    # Energy ranking from a random-gen log file (no calculator re-evaluation)
    from amorphgen.analysis import rank_from_log, format_log_ranking
    result = rank_from_log("random_structures/random_gen.log")
    print(format_log_ranking(result))

    # Validation against literature reference YAML
    from amorphgen.analysis import validate_against_reference, format_validation_report
    import yaml
    with open("examples/reference_a_Ga2O3.yaml") as f:
        ref = yaml.safe_load(f)
    print(format_validation_report(validate_against_reference(sa, ref)))

    # Multi-ensemble comparison plots (e.g. Random vs Hybrid vs Reference).
    # Each descriptor is saved as its own PNG/PDF/CSV.
    from amorphgen.analysis import EnsembleSpec, compare_ensembles
    compare_ensembles(
        ensembles=[
            EnsembleSpec("Random", "random/*.vasp"),
            EnsembleSpec("Hybrid", "hybrid/*.xyz"),
        ],
        rdf_pairs=[("Si-O", "-"), ("Si-Si", "--"), ("O-O", ":")],
        cn_top_key="Si-O",
        cn_bot_key="O-Si",
        angle_keys=[("O-Si-O", "-"), ("Si-O-Si", "--")],
        exp_density=(2.18, 2.22),
        output_dir="comparison_plots",
        prefix="sio2",
    )
"""

from .analyser import StructureAnalyser
from .energy import rank_from_log, format_log_ranking
from .validate import validate_against_reference, format_validation_report
from .comparison_plots import (
    EnsembleSpec,
    compare_ensembles,
    plot_partial_rdf,
    plot_coordination,
    plot_bond_angles,
    plot_density,
)

__all__ = [
    "StructureAnalyser",
    "rank_from_log",
    "format_log_ranking",
    "validate_against_reference",
    "format_validation_report",
    "EnsembleSpec",
    "compare_ensembles",
    "plot_partial_rdf",
    "plot_coordination",
    "plot_bond_angles",
    "plot_density",
]
