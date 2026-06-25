import json
from pathlib import Path

import pytest

from benchmarks.gdph_v2.wait_for_stage import wait_for_completed


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_wait_for_completed_accepts_success(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    _write(path, {"state": "completed", "returncode": 0})
    assert wait_for_completed(path, 0.001)["state"] == "completed"


@pytest.mark.parametrize(
    "payload",
    [
        {"state": "failed", "returncode": 1},
        {"state": "completed", "returncode": 1},
    ],
)
def test_wait_for_completed_rejects_failure(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "status.json"
    _write(path, payload)
    with pytest.raises(RuntimeError, match="dependency did not complete successfully"):
        wait_for_completed(path, 0.001)
