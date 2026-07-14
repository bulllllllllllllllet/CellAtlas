import json

from benchmarks.gdph_v2.audit_experiment import _json


def test_json_reads_existing_payload(tmp_path) -> None:
    path = tmp_path / "payload.json"
    payload = {"passed": True, "count": 3}
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _json(path) == payload


def test_json_returns_none_for_invalid_json(tmp_path) -> None:
    path = tmp_path / "payload.json"
    path.write_text("{not valid json", encoding="utf-8")

    assert _json(path) is None
