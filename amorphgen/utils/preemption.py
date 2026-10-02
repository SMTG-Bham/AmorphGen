"""Cooperative Slurm interruption at safe simulation boundaries.

The CLI enables this only with ``AMORPHGEN_CHECKPOINT_ON_SIGNAL=1``.
Signal handlers merely set a flag: trajectory writes, cleanup and manifest
updates happen in normal Python execution. MD stops at a regular output
boundary so frame-count-based resume remains valid. Optimisations restart
their unfinished structure/chunk; optimiser state is not checkpointed.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import signal


PREEMPTION_EXIT_CODE = 75
_requested_signal = None


class PreemptionRequested(SystemExit):
    """Exit without allowing broad ``except Exception`` retry handlers to run."""

    def __init__(self, signum):
        super().__init__(PREEMPTION_EXIT_CODE)
        self.signum = signum

    def __str__(self):
        return (f"{signal.Signals(self.signum).name}: stopped at a safe boundary; "
                "resume with --resume (exit 75)")


def _request_stop(signum, frame):
    global _requested_signal
    if _requested_signal is None:
        _requested_signal = signum


def stop_if_requested():
    """Stop at a caller-owned checkpoint or restartable work boundary."""
    if _requested_signal is not None:
        raise PreemptionRequested(_requested_signal)


@contextmanager
def checkpoint_signals():
    """Temporarily install opt-in handlers and restore the caller's handlers."""
    global _requested_signal
    if os.environ.get("AMORPHGEN_CHECKPOINT_ON_SIGNAL") != "1":
        yield
        return
    previous_request = _requested_signal
    previous_handlers = {}
    _requested_signal = None
    try:
        for name in ("SIGUSR1", "SIGTERM"):
            signum = getattr(signal, name, None)
            if signum is not None:
                previous_handlers[signum] = signal.signal(signum, _request_stop)
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        _requested_signal = previous_request
