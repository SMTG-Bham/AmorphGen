# Tutorials

Jupyter notebook tutorials demonstrating AmorphGen workflows.

**Start here**:

| Tutorial | Description | Backend | System |
|----------|-------------|---------|--------|
| [Tutorial 1: Quick-start tutorial](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T1_5min_intro/tutorial_1_5min_intro.ipynb) | Orientation: what AmorphGen does, the three workflows | CHGNet | a-SiO₂ |

**Workflow tutorials** — each tutorial reports its own measured wall time on the CPU it was validated on:

| Tutorial | Description | Backend | System |
|----------|-------------|---------|--------|
| [Tutorial 2: Random structure generation](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T2_automated_random_gen/tutorial_2_automated_random_gen.ipynb) | Composition is the only input; auto minsep / density / target CN / oxidation state; CHGNet relax + save each structure | CHGNet | Si, SiO₂, In₂O₃, CdTe, AlN, LiCl, TiO₂, Cu |
| [Tutorial 3: Parameters control + ensemble analysis](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T3_random_gen/tutorial_3_random_generation.ipynb) | Explicit minsep (from crystal-phase bond lengths) +  target density (from amorphous-thin-film references). Structure ensembles per system; quantitative RDF / energy / CN / bond-angle analysis vs the crystalline reference | MACE | In₂O₃, TiO₂, Al₂O₃, Ga₂O₃ |
| [Tutorial 4: Melt-and-quench ](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T4_MQ_via_7_steps/tutorial_4_melt_quench.ipynb) | Full 7-stage MQ pipeline | CHGNet (CPU) / MACE (GPU) | SiO₂ |
| [Tutorial 5: Hybrid approach](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T5_mix_random_MQ/tutorial_5_batch_quench.ipynb) | Random gen → equilibrate → batch quench | CHGNet | TiO₂ |
| [Tutorial 6: Classical potentials](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T6_classical_potential/tutorial_6_classical_potential.ipynb) | Buckingham+Coulomb relaxation, hybrid workflow | Classical | SiO₂, Al₂O₃, TiO₂ |

**Application case studies** (assume familiarity with the workflow tutorials):

| Tutorial | Description | Backend | System |
|----------|-------------|---------|--------|
| [Tutorial 7: Dimer dissociation kinetics](https://github.com/SMTG-Bham/AmorphGen/blob/main/Tutorials/T7_application_dimer_dissociation/tutorial_7_dimer_dissociation.ipynb) | O–O peroxide-defect dissociation in amorphous oxide; Arrhenius temperature scan | MACE | In₂O₃ |
