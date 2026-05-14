"""Verify compute_fid_adm passes absolute paths to the evaluator subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def smoke_npz_in_fid_dir() -> Path:
    """One minimal ADM-shaped npz under dit_s3cache/fid (repo-relative path valid from DiT root)."""
    fid_dir = _REPO_ROOT / "dit_s3cache" / "fid"
    fid_dir.mkdir(parents=True, exist_ok=True)
    path = fid_dir / "_pytest_smoke_fid.npz"
    arr = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    np.savez(path, arr_0=arr)
    yield path
    path.unlink(missing_ok=True)


def test_compute_fid_adm_passes_absolute_paths_to_subprocess(
    smoke_npz_in_fid_dir: Path,
) -> None:
    """Relative ref/sample paths must become absolute in argv so open() works with cwd=fid/."""
    from dit_s3cache.fid.fid_cache_sensitivity import compute_fid_adm

    evaluator = _REPO_ROOT / "dit_s3cache" / "fid" / "evaluator.py"
    if not evaluator.is_file():
        pytest.skip("evaluator.py not present")

    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        captured["cmd"] = list(cmd)
        captured["cwd"] = kwargs.get("cwd")
        return MagicMock(returncode=0, stdout="FID: 0.0\n", stderr="")

    rel_eval = Path("dit_s3cache/fid/evaluator.py")
    rel_npz = Path("dit_s3cache/fid/_pytest_smoke_fid.npz")

    old_cwd = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)
        with patch.object(subprocess, "run", side_effect=fake_run):
            compute_fid_adm(
                adm_python=sys.executable,
                adm_evaluator=rel_eval,
                ref_batch=rel_npz,
                sample_batch=rel_npz,
            )
    finally:
        os.chdir(old_cwd)

    cmd: list[str] = captured["cmd"]  # type: ignore[assignment]
    assert len(cmd) == 4
    for i in (1, 2, 3):
        p = Path(cmd[i])
        assert p.is_absolute(), f"argv[{i}] must be absolute, got {cmd[i]!r}"
        assert p.is_file(), f"argv[{i}] must exist: {cmd[i]!r}"

    workdir = Path(str(captured["cwd"]))
    assert workdir == (_REPO_ROOT / "dit_s3cache" / "fid").resolve()

    # With cwd set to fid/, a *relative* repo path would wrongly resolve; absolute must still exist.
    os.chdir(workdir)
    try:
        for i in (2, 3):
            assert Path(cmd[i]).is_file()
    finally:
        os.chdir(_REPO_ROOT)


def test_wrong_nested_path_does_not_exist(smoke_npz_in_fid_dir: Path) -> None:
    """Illustrates the bug: repo-relative path from inside fid/ points to a nested non-file."""
    workdir = _REPO_ROOT / "dit_s3cache" / "fid"
    nested = workdir / "dit_s3cache" / "fid" / smoke_npz_in_fid_dir.name
    assert not nested.exists()
