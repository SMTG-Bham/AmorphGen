"""Names of the scalar observables supported by sequential generation."""

SCALAR_DESCRIPTORS = frozenset({"density", "energy.total", "energy.per_atom"})
_SPECIES_COUNTS = {
    "coordination": 2, "total_coordination": 1,
    "bond_distance": 2, "bond_angle": 3,
}


def supports_sequential_descriptor(name, symbols):
    """Check descriptor syntax and membership in the caller's allowed species.

    The CLI allows chemical elements; a generation run requires those elements
    in its actual composition. Statistical bounds and types are separate from
    this observable-name contract.
    """
    if name in SCALAR_DESCRIPTORS:
        return True
    kind, separator, label = name.partition(".")
    elements = label.split("-")
    return bool(separator and len(elements) == _SPECIES_COUNTS.get(kind)
                and all(element in symbols for element in elements))
