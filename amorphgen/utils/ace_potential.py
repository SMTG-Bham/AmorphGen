"""
amorphgen.utils.ace_potential
-----------------------------
ACE potentials (pacemaker ``.yaml``, ``.yace`` or ``.ace`` files) as an ASE
calculator, through pyace's ``PyACECalculator``.

Imported only by the ACE loader in :mod:`amorphgen.utils.calculators`, so the
optional ``pyace`` package (``pip install "amorphgen[ace]"``) is not needed
anywhere else.
"""

from __future__ import annotations

import logging

# pyace calls logging.basicConfig(level=INFO) on import, which would print
# every library's INFO records for the rest of the run; undo that.
_root = logging.getLogger()
_handlers, _level = list(_root.handlers), _root.level
try:
    from pyace import PyACECalculator
finally:
    _root.handlers[:] = _handlers
    _root.setLevel(_level)

ACE_PARAM_KEYS = frozenset({"recursive_evaluator", "recursive", "fast_nl"})


class ACECalculator(PyACECalculator):
    """``PyACECalculator`` that the pipeline stages can share.

    ``potential_files`` lists the potential file (hashed into the run
    manifest).
    """

    def __init__(self, basis_set, **kwargs):
        super().__init__(basis_set, **kwargs)
        self.potential_files = [basis_set]

    def __deepcopy__(self, memo):
        # The stages deep-copy their input Atoms, and the attached calculator
        # with them; pyace's C++ evaluator cannot be copied. The stages then
        # attach the shared calculator anyway.
        return self
