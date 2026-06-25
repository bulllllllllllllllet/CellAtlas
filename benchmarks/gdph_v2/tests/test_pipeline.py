from pathlib import Path

from benchmarks.gdph_v2.run_remaining_pipeline import build_stages


def test_remaining_pipeline_has_ordered_terminal_audit() -> None:
    stages = build_stages(Path("/tmp/experiment"))
    names = [name for name, _ in stages]
    assert names[0] == "validate_setup_latest"
    assert names.index("validate_model_compatibility_latest") < names.index(
        "audit_after_pilot_inference"
    )
    assert names.index("paired_scale_validation") < names.index("fullres_main20")
    assert names.index("linear_probe_5fold") < names.index("generate_region_queries")
    assert names.index("patch_inference_main20") < names.index("generate_final_report")
    assert names[-1] == "final_audit"
    assert "--require_complete" in stages[-1][1]
