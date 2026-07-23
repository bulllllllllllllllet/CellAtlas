# CaRePrompt baseline protocol

This directory implements the auditable two-day protocol in
`doc/BASELINE_2DAY_PLAN_20260722_180142.md`. Source files have stable names;
every run must use a fresh `YYYYMMDD_HHMMSS` artifact directory below
`/nfs-medical3/zyh/v4/baseline/`.

The code deliberately has no dependency, checkpoint, prompt, or model fallback.
SAM-Med2D, WSI-SAM, and CaRePrompt have nonstandard repository APIs, so their
YAML files require an explicit `module:factory` integration. The returned
backend must implement exactly:

```python
predict(image_or_multiscale_inputs, positive_prompts, negative_prompts,
        prompt_type, positive_box)
```

and return the unified prediction fields documented in
`adapters/callable_adapter.py`.

## Required order

All commands run from the repository root in the `aligner` environment.

1. Freeze region-derived boxes without reading GT:

```bash
conda run -n aligner python -m benchmarks.v4.baseline.tools.build_prompt_geometry \
  --occurrence-source <audited_4000_occurrences.parquet> \
  --cache-index <cache_index.parquet> --label-index <label_index.parquet> \
  --patch-index <patch_index.parquet> --eligibility-index <eligibility_index.parquet> \
  --phase2-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
  --phase2-checkpoint <phase2_checkpoint.pth> \
  --phase5-config benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml \
  --split val --timestamp YYYYMMDD_HHMMSS --limit 1
```

Run the one-occurrence command first. After it passes, use a new timestamp and
remove `--limit` for the frozen geometry job.

2. Build and audit an immutable episode manifest:

```bash
conda run -n aligner python -m benchmarks.v4.baseline.tools.build_episode_manifest \
  --occurrence-source <audited_4000_occurrences.parquet> \
  --prompt-geometry <prompt_geometry_TIMESTAMP.parquet> \
  --cache-index <cache_index.parquet> --label-index <label_index.parquet> \
  --patch-index <patch_index.parquet> --eligibility-index <eligibility_index.parquet> \
  --phase5-config benchmarks/v4/phase_5_prompt_encoder/configs/phase5_prompt_encoder.yaml \
  --split val --timestamp YYYYMMDD_HHMMSS

conda run -n aligner python -m benchmarks.v4.baseline.tools.validate_prompt_coordinates \
  --episode-manifest <episode_manifest_TIMESTAMP.parquet> \
  --phase2-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
  --split val --timestamp YYYYMMDD_HHMMSS
```

3. Evaluate 1, 64, 360, then 4,000 occurrences with new timestamps. A rank
process writes deterministic shard-local Parquet files immediately:

```bash
conda run -n aligner python -m benchmarks.v4.baseline.tools.evaluate_baseline \
  --config benchmarks/v4/baseline/configs/fixed_slic.yaml \
  --phase2-config benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml \
  --episode-manifest <episode_manifest_TIMESTAMP.parquet> --split val \
  --timestamp YYYYMMDD_HHMMSS --limit 1
```

For multi-process runs, rank 0 creates the run directory; other ranks use its
exact path via `--output-dir`, with the same timestamp and world size.

4. Strictly merge all shards and build the comparison report:

```bash
conda run -n aligner python -m benchmarks.v4.baseline.tools.merge_baseline_shards \
  --run-dir <method_TIMESTAMP> --episode-manifest <episode_manifest.parquet> \
  --timestamp YYYYMMDD_HHMMSS --expected-world-size 1

conda run -n aligner python -m benchmarks.v4.baseline.tools.build_baseline_report \
  --config <frozen_report_config.yaml> --timestamp YYYYMMDD_HHMMSS
```

Existing immutable J3/J5 audit evidence can be converted without rerunning test:

```bash
conda run -n aligner python -m benchmarks.v4.baseline.tools.import_careprompt_evidence \
  --episode-manifest <episode_manifest.parquet> \
  --evidence-metrics <joint_pixel_audit/episode_metrics.parquet> \
  --model-prefix baseline --method-name careprompt_j3 --split test \
  --timestamp YYYYMMDD_HHMMSS
```

Use `--model-prefix joint` for J5. Imported evidence deliberately reports
efficiency metrics as unavailable because the historical audit did not persist
per-episode latency or peak memory.

Formal validation and test jobs must be launched in named `tmux` sessions with
timestamped persisted logs as required by the v4 workflow. Test is permitted
only after the validation configuration and all hashes are frozen.
