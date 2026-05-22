"""v2.14 Phase 5 — disk-headroom precheck unit tests.

Pins the v2.14 Phase 5 incident-improvement: `_check_disk_headroom`
must abort cleanly when free disk falls below the configured floor
(default 10 GB; overridable via SOAK_DISK_HEADROOM_FLOOR_GB env var).

Incident reference: v2.13 P1 soak 2026-05-22 — disk filled to 100%
mid-judge and crashed the Qdrant Docker container's overlayfs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class _FakeDiskUsage:
    def __init__(self, total: int, used: int, free: int) -> None:
        self.total = total
        self.used = used
        self.free = free


def test_disk_precheck_aborts_when_below_floor(monkeypatch, tmp_path):
    """If free disk is below the floor, raise SystemExit(3)."""
    from synthetic_soak import _check_disk_headroom

    # Simulate 5 GB free (under the default 10 GB floor)
    fake = _FakeDiskUsage(total=900_000_000_000, used=895_000_000_000,
                          free=5 * 1024 ** 3)
    with patch("synthetic_soak.shutil.disk_usage", return_value=fake):
        with pytest.raises(SystemExit) as excinfo:
            _check_disk_headroom(tmp_path / "work.jsonl", floor_gb=10.0)
    assert excinfo.value.code == 3


def test_disk_precheck_passes_when_above_floor(monkeypatch, tmp_path):
    """If free disk is above the floor, the function returns normally."""
    from synthetic_soak import _check_disk_headroom

    # Simulate 50 GB free (well above the default 10 GB floor)
    fake = _FakeDiskUsage(total=900_000_000_000, used=850_000_000_000,
                          free=50 * 1024 ** 3)
    with patch("synthetic_soak.shutil.disk_usage", return_value=fake):
        # No exception expected
        _check_disk_headroom(tmp_path / "work.jsonl", floor_gb=10.0)


def test_disk_precheck_warns_when_tight(monkeypatch, tmp_path, capsys):
    """Free disk between floor and 2× floor should print a NOTE but not abort."""
    from synthetic_soak import _check_disk_headroom

    # 15 GB free → above 10 GB floor but below 2× floor (20 GB) → NOTE
    fake = _FakeDiskUsage(total=900_000_000_000, used=885_000_000_000,
                          free=15 * 1024 ** 3)
    with patch("synthetic_soak.shutil.disk_usage", return_value=fake):
        _check_disk_headroom(tmp_path / "work.jsonl", floor_gb=10.0)
    captured = capsys.readouterr()
    assert "disk headroom tight" in captured.out


def test_disk_precheck_handles_missing_path(monkeypatch, tmp_path, capsys):
    """When work_path's parent doesn't exist, fall back to REPO_ROOT
    rather than raising an OSError mid-check."""
    from synthetic_soak import _check_disk_headroom

    # Simulate disk_usage failing once on the parent dir
    nonexistent = tmp_path / "missing" / "work.jsonl"
    # The function checks work_path.parent.exists() first — if not, uses REPO_ROOT
    fake = _FakeDiskUsage(total=900_000_000_000, used=400_000_000_000,
                          free=500 * 1024 ** 3)
    with patch("synthetic_soak.shutil.disk_usage", return_value=fake):
        _check_disk_headroom(nonexistent, floor_gb=10.0)  # should not raise


def test_disk_precheck_handles_oserror_gracefully(monkeypatch, tmp_path, capsys):
    """If shutil.disk_usage raises OSError (broken mount, permission, etc.),
    log a warning and return rather than aborting the soak."""
    from synthetic_soak import _check_disk_headroom

    def _raise(*args, **kwargs):
        raise OSError("simulated stat failure")

    with patch("synthetic_soak.shutil.disk_usage", side_effect=_raise):
        # Should not raise SystemExit
        _check_disk_headroom(tmp_path / "work.jsonl", floor_gb=10.0)
    captured = capsys.readouterr()
    assert "disk-headroom precheck couldn't stat" in captured.err
