---
name: v4-workflow
description: Implement, validate, or operate the CellAtlas v4 six-module prompt-aware multiscale WSI system. Use for any CellAtlas v4 task involving the six modules defined in benchmarks/v4/doc/first_talk.md, v4 data preparation, training, inference, evaluation, GPU/CPU parallelization, or v4 artifacts.
---

# CellAtlas v4 Workflow

## Fixed constraints

- Run Python with the `aligner` environment, for example `conda run -n aligner python ...`.
- Write all v4 code only under `/home/zhaoyh/CellAtlas/benchmarks/v4/`; place each module's code in its matching phase directory.
- Write all generated data, checkpoints, reports, logs, visualizations, and temporary artifacts under `/nfs-medical3/zyh/v4/`.
- Append a timestamp in `YYYYMMDD_HHMMSS` form to every newly created output file and directory. Include the timestamp in manifests that name output artifacts.
- Treat `/home/zhaoyh/CellAtlas/benchmarks/v4/data_preprocess/gdph_tissue_nuclei_pairs.csv` and every image it references as read-only source data. Never delete, move, overwrite, or mutate them.
- Never execute delete operations for v4 work. Do not use `rm`, recursive cleanup, reset, checkout, or destructive overwrite patterns.
- Do not add, implement, or invoke fallback/recovery logic. Ask the user for explicit approval before adding any fallback behavior.
- Before treating unavailable `nvidia-smi`, GPU devices, or `htop` as absent, request elevated access and re-check.

## Source specification and phase layout

Read `/home/zhaoyh/CellAtlas/benchmarks/v4/doc/first_talk.md` before designing a v4 module. Implement its six modules in these directories:

| Module | Required phase directory |
| --- | --- |
| 1. Multi-scale WSI pyramid input | `benchmarks/v4/phase_1_multiscale/` |
| 2. Deep region encoder | `benchmarks/v4/phase_2_region_encoder/` |
| 3. Cell-to-region injection | `benchmarks/v4/phase_3_cell_region/` |
| 4. Cross-scale region interaction | `benchmarks/v4/phase_4_cross_scale/` |
| 5. Prompt encoder | `benchmarks/v4/phase_5_prompt_encoder/` |
| 6. Context-aware mask decoder | `benchmarks/v4/phase_6_mask_decoder/` |

Create a timestamped artifact directory inside `/nfs-medical3/zyh/v4/` for each run, for example `/nfs-medical3/zyh/v4/phase_2_region_encoder_20260714_135900/`. Keep source code names stable; timestamp outputs only.

## Execution workflow

1. Read the relevant module and its dependencies in `first_talk.md`; inspect existing v4 artifacts before coding.
2. State the exact inputs, output directory, timestamp, phase directory, and validation criterion before running an expensive job.
3. Implement within the assigned phase directory, with explicit CLI arguments for input manifest and timestamped output root.
4. Validate lightweight invariants first: manifest integrity, tensor shapes, coordinate/scale alignment, deterministic output naming, and a small representative run.
5. Run the full workload only after the small run succeeds. Record commands, hardware allocation, configuration, and metrics in the timestamped output directory.

## Long-running job observability

- Launch expensive or long-running v4 jobs in a named `tmux` session with `remain-on-exit` enabled.
- Redirect both stdout and stderr to a timestamped `.log` file under `/nfs-medical3/zyh/v4/`, append the final `EXIT_CODE`, and report the session name and log path before returning control.
- Never rely on an interactive tmux pane as the only record: tmux may exit after completion or failure. Inspect the persisted log and expected artifacts to determine job status.
- Before any sampled or full workload, run one representative source item through the same code path and retain its output. Only scale to the requested sample/full size after that single-item run exits successfully and its invariants are checked.
- For fan-out preprocessing, do not retain all completed results only in the coordinator process or write one final aggregate at the end. Persist deterministic shard-local artifacts (for example, small Parquet shards) as work completes, plus append-only `completed.jsonl` and `failures.jsonl` status records. Merge shards only after all expected shards validate.
- A validation failure discovered late must leave prior successful shards inspectable and reusable. Do not erase partial artifacts, silently exclude a failed sample, or convert unknown labels into a valid class without explicit user approval.

## Experimental-code preflight and anti-rerun gate

Treat an experiment as a complete data-and-systems pipeline, not merely a model that can start. Before any costly run, think through the upstream data contract, downstream artifacts, failure modes, and recovery path; test inexpensive invariants first so an avoidable implementation error never consumes a full rerun.

- Perform an explicit design review before implementation: input shapes/dtypes/coordinate systems, split isolation, random-access cost, tensor/loss ranges, AMP precision, DDP synchronization, initialization and pretrained-weight transfer, evaluation definitions, checkpoint/resume behavior, and disk-growth implications.
- Do not promote a job to full scale after only an import, shape, or one-forward test. Require a representative multi-step smoke run through the same dataloader, augmentation, AMP/DDP, optimizer, validation, checkpoint, and logging paths used in the formal command.
- Verify data artifacts before training: expected row/split counts, shard completeness, readable representative records, label/coordinate equality against the source, and bounded random-read latency and memory. Do not use large compressed `.npz` shards for per-sample random access when each lookup decompresses an entire array; convert once to an mmap/block-friendly artifact and retain the source artifact.
- Make numerical failures loud and actionable: on every DDP rank, check loss components, total loss, gradients, and key activations for finiteness. Stop all ranks together on a non-finite value and log epoch, step, rank, AMP mode, and decomposed losses; never silently skip bad batches and continue a formal run.
- Analyse the loss for degenerate optima before spending compute. For assignment/region/token models, ensure supervision actually distinguishes slots, include an appropriate non-collapse constraint, and monitor active-slot count, assignment entropy/area distribution, purity and boundary metrics. Hold pseudo-label supervision until these indicators are stable before decaying it.
- When fine-tuning, verify that intended pretrained tensors (including any semantic head) are truly loaded rather than silently dropped; record compatible, missing, and unexpected tensor counts. Use conservative differential learning rates for pretrained versus newly initialized heads.
- Validate distributed behavior rather than assuming it: confirm every selected GPU/rank receives batches and gradients, rank-reduced metrics are correct, and validation/checkpoint code runs under DDP. Check resource use after startup and during at least one validation boundary.
- Save immutable `checkpoint_epoch_XXX.pth` checkpoints containing model, optimizer, scheduler/scaler, epoch, config, and history. Maintain explicit `last_checkpoint.json` and `best_checkpoint.json` pointers, and run one resume smoke test into a new timestamped output directory to prove it starts at the next epoch without reprocessing completed epochs.
- Gate the formal run on recorded evidence: finite multi-step losses, non-degenerate task metrics, stable resource use, successful validation/checkpoint, and successful resume. If a gate fails, fix the cause and repeat only the smallest targeted verification before relaunching.

## Performance policy

The host has six NVIDIA RTX 5880 Ada GPUs and 128 CPU cores. Prefer scalable execution:

- Assign independent samples/shards to processes, use a bounded worker pool, and shard data deterministically.
- Use one explicit process per allocated GPU for GPU-bound work; set device assignment and seeds per process.
- Use parallel CPU decoding/preprocessing with a configurable worker count; avoid oversubscribing workers when GPU processes also decode data.
- Batch disk reads, use persistent workers where appropriate, avoid repeated feature extraction, and write shard-local outputs before a deterministic merge.
- Expose `--gpus`, `--num-workers`, `--batch-size`, and `--seed` in expensive scripts. Log their resolved values.
- Benchmark a representative workload and scale concurrency based on measured GPU utilization, CPU utilization, memory, and NFS throughput.

## CellAtlas v4 fixed development cohort

- Unless the user explicitly requests a final full-data run, use the fixed 200-WSI development cohort: 140 train, 30 validation, 30 test; preserve its patient isolation and the shared split across all six modules.
- The cohort artifacts are `/nfs-medical3/zyh/v4/phase1/data/cohort200_20260716_010810/cohort_manifest.parquet`, tiled GT manifest `/nfs-medical3/zyh/v4/phase1/data/tiled_gt_cohort200_20260716_011804/tiled_gt_manifest.parquet`, and tiled patch index `/nfs-medical3/zyh/v4/phase1/data/tiled_patch_index_cohort200_20260716_013248/patch_index_10x_tiled_gt.parquet`.
- Treat the tiled GT as lossless RGB label storage, not an image-model input; do not replace it with ordinary PNG when random patch access is required.
- Default baseline budget: 30 epochs, 20,000 training patches and 4,000 validation patches per epoch, four GPUs with two DataLoader workers per rank. Re-benchmark before changing this concurrency.

## Safety checks

- Use non-mutating inspection commands first (`rg`, `find`, `ls`, metadata reads).
- Never overwrite an existing timestamped artifact path. Create a new timestamped path instead.
- Keep all paths in manifests relative to their declared root where practical; record the root in metadata.
- If a requested action conflicts with these constraints, stop and request user direction before proceeding.

## Visualization and human review

- Do not use model visual capability to approve or reject CellAtlas v4 visualizations. Leave visual correctness review to the user and record it as pending until the user reports a result.
- For segmentation visualization, show the exact target-class ground truth and predicted mask side by side at the same coordinate extent, aspect ratio, and nearest-neighbor resize. Prefer paired `HE + GT overlay | HE + prediction overlay` panels plus paired binary masks.
- Decode ground truth through the configured exact class-ID/RGB contract; never treat all non-background pixels as the prompted target. Mark positive and negative prompts identically on corresponding panels and state their coordinate conversion.
- Before presenting a visualization, programmatically verify source identity, full-resolution dimensions, target class, palette value, binary values, and prompt coordinate bounds. Report numerical metrics separately from the user's visual judgment.
