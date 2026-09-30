"""Conversion must never destroy its inputs through destination aliases."""

import os
import sys

import pytest
from ase import Atoms
from ase.io import read, write

from amorphgen import convert


def _trajectory(path):
    frames = [Atoms("Cu", positions=[[x, 0, 0]], cell=[5, 5, 5], pbc=True)
              for x in (0, 1)]
    write(path, frames, format="extxyz")
    return path.read_bytes()


def test_cli_default_conversion_refuses_to_replace_trajectory(tmp_path, monkeypatch, capsys):
    from amorphgen.cli import main

    src = tmp_path / "traj.xyz"
    original = _trajectory(src)
    monkeypatch.setattr(sys, "argv", ["amorphgen", "--convert", str(src)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "would overwrite an input file" in capsys.readouterr().out
    assert src.read_bytes() == original
    assert len(read(src, index=":")) == 2


@pytest.mark.parametrize("alias", ["same_directory", "symlink_directory", "symlink_file", "hardlink"])
def test_explicit_output_cannot_alias_input(tmp_path, alias):
    src_dir = tmp_path / "input"
    src_dir.mkdir()
    src = src_dir / "traj.xyz"
    original = _trajectory(src)
    out = tmp_path / "out"
    if alias == "same_directory":
        out = src_dir / ".." / "input"
    elif alias == "symlink_directory":
        out.symlink_to(src_dir, target_is_directory=True)
    else:
        out.mkdir()
        if alias == "symlink_file":
            (out / src.name).symlink_to(src)
        else:
            os.link(src, out / src.name)
    with pytest.raises(ValueError, match="would overwrite an input file"):
        convert(str(src), output_format="xyz", output_dir=str(out))
    assert src.read_bytes() == original


def test_directory_collision_is_checked_before_any_conversion(tmp_path):
    src = tmp_path / "traj.xyz"
    original = _trajectory(src)
    write(tmp_path / "a.cif", Atoms("Cu", cell=[5, 5, 5], pbc=True))
    with pytest.raises(ValueError, match="would overwrite an input file"):
        convert(str(tmp_path), output_format="xyz", output_dir=str(tmp_path))
    assert not (tmp_path / "a.xyz").exists()
    assert src.read_bytes() == original


def test_directory_outputs_remain_distinct_after_disambiguation(tmp_path):
    src = tmp_path / "input"
    src.mkdir()
    for count, name in enumerate(["s.cif", "s.xyz", "s_xyz.cif"], start=1):
        atoms = Atoms("Cu" * count, positions=[[i, 0, 0] for i in range(count)],
                      cell=[5, 5, 5], pbc=True)
        write(src / name, atoms)
    outputs = convert(str(src), output_format="vasp", output_dir=str(tmp_path / "out"))
    assert len(set(outputs)) == 3
    assert [len(read(path)) for path in outputs] == [1, 2, 3]


def test_converting_to_another_directory_preserves_input_trajectory(tmp_path):
    src = tmp_path / "traj.xyz"
    original = _trajectory(src)
    outputs = convert(str(src), output_format="vasp", output_dir=str(tmp_path / "out"))
    assert src.read_bytes() == original
    assert len(outputs) == 1
    assert read(outputs[0]).positions[0, 0] == pytest.approx(1.0)
