# Packing-factor calibration

`packing_reference_densities.csv` lists 108 compounds with a sourced density:
the composition, the measured density, whether it is an amorphous or a
crystalline reference, the source with a DOI or URL, and whether the source was
checked (`verified`). `fit_packing_factors.py` assigns each compound to its
material class with the installed AmorphGen, back-solves the packing factor that
would reproduce its target density (the reference for an amorphous compound, the
crystal density times 0.80 otherwise), and reports per class the median factor,
its range and the leave-one-out error of the class median.

```bash
python fit_packing_factors.py            # 100 verified rows, ratio 0.80
python fit_packing_factors.py --all-rows --ratio 0.85 --json fit.json
```

The script only reads; it never changes the factors in
`amorphgen/utils/radii.py` (`PACKING_FACTORS`).
