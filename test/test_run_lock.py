"""Output directory locks exclude other processes and survive interrupted runs."""

from contextlib import contextmanager
import multiprocessing

import pytest

from amorphgen.utils.run_lock import run_lock


def _hold_lock(directory, connection):
    try:
        with run_lock(directory):
            connection.send(("acquired", None))
            connection.recv()
    except RuntimeError as exc:
        connection.send(("rejected", str(exc)))
    finally:
        connection.close()


@contextmanager
def _lock_process(directory):
    # Spawn ensures this process cannot inherit a parent's locked descriptor.
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_hold_lock, args=(directory, child))
    process.start()
    child.close()
    try:
        yield process, parent
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        parent.close()


def _message(connection):
    assert connection.poll(20), "Lock operation blocked instead of completing"
    return connection.recv()


def test_other_process_is_rejected_while_directory_is_locked(tmp_path):
    with run_lock(tmp_path):
        with _lock_process(tmp_path) as (process, connection):
            state, message = _message(connection)
            assert state == "rejected"
            assert "already using this output directory" in message
            assert str(tmp_path / ".amorphgen.lock") in message
            process.join(timeout=10)
            assert process.exitcode == 0


def test_distinct_directories_can_run_concurrently(tmp_path):
    with run_lock(tmp_path / "first"):
        with _lock_process(tmp_path / "second") as (process, connection):
            assert _message(connection) == ("acquired", None)
            connection.send("release")
            process.join(timeout=10)
            assert process.exitcode == 0


def test_normal_process_exit_releases_lock(tmp_path):
    with _lock_process(tmp_path) as (process, connection):
        assert _message(connection) == ("acquired", None)
        connection.send("release")
        process.join(timeout=10)
        assert process.exitcode == 0
        with run_lock(tmp_path):
            assert (tmp_path / ".amorphgen.lock").exists()


def test_abrupt_process_death_releases_lock(tmp_path):
    with _lock_process(tmp_path) as (process, connection):
        assert _message(connection) == ("acquired", None)
        process.kill()
        process.join(timeout=10)
        assert process.exitcode is not None
        assert process.exitcode != 0
        with run_lock(tmp_path):
            assert (tmp_path / ".amorphgen.lock").exists()


def test_exception_releases_lock_without_removing_file(tmp_path):
    with pytest.raises(ValueError, match="run failed"):
        with run_lock(tmp_path):
            raise ValueError("run failed")
    lock_path = tmp_path / ".amorphgen.lock"
    inode = lock_path.stat().st_ino
    with run_lock(tmp_path):
        assert lock_path.stat().st_ino == inode
    assert lock_path.stat().st_ino == inode


def test_existing_lock_contents_and_metadata_are_unchanged(tmp_path):
    lock_path = tmp_path / ".amorphgen.lock"
    lock_path.write_bytes(b"existing contents\n")
    before = lock_path.stat()
    with run_lock(tmp_path):
        assert lock_path.read_bytes() == b"existing contents\n"
    after = lock_path.stat()
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns == before.st_ctime_ns
