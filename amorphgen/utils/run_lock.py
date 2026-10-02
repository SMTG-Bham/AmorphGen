"""Exclusive ownership of an output directory for the duration of a run."""

from contextlib import contextmanager
import errno
import fcntl
from pathlib import Path


@contextmanager
def run_lock(directory):
    """Acquire a nonblocking process lock on ``directory``.

    The lock file is deliberately persistent and its contents are untouched.
    Removing it after a run could allow two processes to lock different inodes
    at the same path. Closing the handle releases the lock, including on an
    exception or process termination, so an existing file is not a stale lock.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".amorphgen.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError(
                    f"Another run is already using this output directory "
                    f"(lock: {lock_path.resolve()})"
                ) from exc
            raise
        yield lock_path
