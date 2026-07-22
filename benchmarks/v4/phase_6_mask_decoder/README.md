# Phase 6: Context-aware mask decoder

The first Phase-6 path freezes the best Phase-5 prompt model and learns a
zero-initialized residual graph decoder over fine regions. This guarantees
that the untrained decoder reproduces Phase-5 logits exactly and makes the
incremental value of context auditable.

Each region attends only to centroid-KNN neighbours and itself. Decoder inputs
combine Phase-5 task-aware tokens, initial logits, prompt task token, centroid,
and area. Training retains balanced BCE, Dice, and prompt ranking losses and
adds an edge-boundary objective. Evaluation uses a fixed probability threshold
of 0.5 and reports both initial and refined region metrics.

`project_region_probabilities` implements the specified soft projection
`M(x,y) = sum_k A(x,y,k) p_k`. Full pixel projection requires the Phase-2 soft
assignment and is intentionally not approximated from centroids. The cached
Phase-4 token corpus does not contain dense assignments, so region-level graph
training remains random-access efficient while pixel evaluation uses an
explicit Phase-2 assignment path.

`tools/evaluate_prompt_masks.py` recomputes the matching Phase-2 assignment on
validation only and reports four distinct quantities: pooled micro region
Dice, per-episode macro region Dice, macro/micro region Dice after excluding
all positive and negative prompt slots, and macro/micro pixel Dice after soft
projection. It also reports a 2-pixel-tolerance boundary F1. Episodes with no
remaining target region after prompt exclusion are counted but never treated
as perfect empty-mask predictions in the unprompted macro Dice.

## Differentiable joint pixel path

`train_joint_pixel.py` is the end-to-end Phase-2→3→5→6 path used before final
joint fine-tuning. It recomputes the Phase-2 DeepLabV3-ResNet50 assignment from
HE, injects routed Phase-3 cell features, gathers positive/negative prompt sets
from the online fused tokens, runs the Phase-5 prompt encoder and Phase-6 graph
decoder, and projects region probabilities through the soft assignment. Pixel
BCE, soft Dice, boundary, region auxiliary, assignment balance, entropy, and
compactness losses therefore backpropagate into the Phase-2 embedding and
assignment heads. The Phase-2 backbone/semantic head and pretrained
Phase-3/Phase-5 modules are frozen by default.

The trainer records initialization agreement against the audited token cache,
per-group gradient norms, pixel/region/unprompted Dice, 2-pixel boundary F1,
class and prompt-size breakdowns, immutable checkpoints, and resumable
optimizer/scheduler/scaler state. `--overfit-episode-index` is a single-GPU
debug gate and must not be used for a formal run. Formal runs use train and
validation routing only; test is not an accepted trainer input.

Default settings are in `configs/phase6_joint_pixel.yaml`. Invoke the trainer
from the repository root with `conda run -n aligner torchrun` and pass all
upstream checkpoints and manifests explicitly.

The J3 partial-prompt configuration keeps Phase2 geometry and Phase3 frozen
and trains only the Phase5 matcher, task projection, and final prompt-set
Transformer layer alongside the decoder. A separate frozen Phase5 snapshot
provides teacher logits/task tokens and is intentionally excluded from the
student state dict. Decoder and prompt parameters use disjoint optimizer
groups and learning rates.

The J4 parent-context configuration starts from the best J3 checkpoint and
freezes every existing module. It gathers cached middle and coarse tokens over
the audited top-4 parent edges and injects them through a scalar zero-gated
residual adapter. The zero gate makes the initial fine-token path exactly
equivalent to the input checkpoint; only the adapter is optimized.

The J5 limited end-to-end configuration also starts from the best J3
checkpoint. It trains only the ResNet50 `backbone.layer4` at `1e-6`, keeps all
BatchNorm running statistics and every downstream module frozen, and retains
assignment regularization plus prompt-teacher anti-forgetting losses. Gradient
clipping remains enabled at norm 5.

The formal full-budget continuation first repeats this frozen J5 recipe for
five epochs with 20,000 train and 4,000 fixed validation episodes per epoch
(`configs/phase6_joint_j5_full_budget.yaml`). A subsequent J10 stage starts
from that run's best checkpoint, adds the zero-gated Phase4 parent-context
adapter, and trains only the adapter at `2.5e-5`. Its final Dice reference is
written only after the full-budget J5 validation result exists.

The J7 control starts from the best J5 checkpoint and jointly trains only
`backbone.layer4` and the Phase-6 decoder at conservative learning rates. All
geometry heads, cell/prompt modules, parent context, and BatchNorm running
statistics remain frozen. Its configuration is
`configs/phase6_joint_j7_layer4_decoder.yaml`.

The J8 loss-weight control keeps the J7 model, optimizer, sampling, and all
other objectives fixed while doubling only the per-episode soft Pixel Dice
weight. Its independent configuration is
`configs/phase6_joint_j8_macro_dice.yaml`.

The J9 coverage control restores the J7 loss and changes only the episode
budget to 2,048 training and 360 fixed validation episodes per epoch. Its
configuration is `configs/phase6_joint_j9_coverage.yaml`.

`tools/evaluate_visualize_joint_pixel.py` supports an explicit
`--baseline-joint-checkpoint` for validation-only checkpoint comparisons. Such
a trained baseline is not the original cache reference: cache-label mismatch,
prompt remapping, and conflicts are persisted as diagnostics but do not veto
the comparison. When no baseline joint checkpoint is supplied, the upstream
Phase2/5 path remains the cache reference and substantial high-purity cache
mismatch is still a hard audit failure. Conflict episodes from the evaluated
joint checkpoint are merged into `stress_set.parquet`; `--skip-panels` runs the
same metric path without regenerating visual panels.

After the candidate is frozen, the same evaluator requires an explicit
`--split test --cell-routing <test routing>` invocation for the one-shot test
run. It never falls back from test to validation inputs. Test outputs use the
`joint_pixel_audit_test_<timestamp>` prefix and record `test_used: true`.

The evaluator also accepts a global `--pixel-thresholds` grid. It persists
macro/micro Pixel Dice plus class and prompt-size breakdowns for every value,
then selects the highest macro Dice whose micro Dice passes
`--threshold-micro-floor`. This is a single global calibration only; it does
not implement classwise thresholds, percentile selection, or area priors.

## Prompt-conflict policy

- The decoder-only main path keeps Phase2 embedding and assignment frozen.
- The filtered control sets `training.exclude_prompt_conflict_episodes: true`;
  filtering applies only to training loss, while input and excluded counts are
  logged separately. Validation is never filtered.
- Every validation conflict occurrence is saved in the run directory as
  `stress_set_epoch_XXX.parquet`, including its episode index, prompt geometry,
  mapped slots, and conflict slots.
- Inference callers must pass a model output through
  `build_inference_response` from `src.conflict_policy`. A conflict returns
  `status="abstain"`, no mask, and an explicit request to adjust or separate
  the positive/negative prompts.

## Checkpoint policy

Joint pixel training uses final mask Dice as its primary checkpoint criterion.
Pixel macro and micro Dice are hard gates. Region Dice, unprompted Dice,
boundary F1, and prompt-conflict/abstention rate remain Pareto objectives and
produce explicit soft warnings, but they do not veto a Dice-eligible model.
`pareto_checkpoints.json` contains the full non-dominated frontier and all
hard/soft checks. Against the Dice reference configured for the current stage,
`best_checkpoint.json` points to the frontier member that maximizes the larger
of its pixel macro and micro Dice gains. Improvement in either Pixel Dice is
enough to promote a candidate as long as both values still pass their hard
gates; soft warnings remain attached. If no checkpoint passes both Dice gates, its status is
`no_eligible_checkpoint`; `last_checkpoint.json` remains available for
resume/debugging.

An optional `checkpoint_selection.noninferiority` block additionally requires
macro/micro gains to stay within configured negative margins and can require
at least one positive Pixel Dice gain. J10 uses margins of 0.001 against the
full-budget J5 reference before its paired validation report.

The final pre-test gates are Pixel macro Dice >= 0.72 and Pixel micro Dice >=
0.7987. The frozen candidate is J5 epoch 1 with a global pixel threshold of
0.5. Validation conflicts remain in the reported rate and stress set, while
inference must abstain instead of returning a mask for a conflicting prompt.
The frozen J5 candidate passed its one-shot 4,000-episode test at threshold
0.5 with Pixel macro/micro Dice 0.739237/0.812773.
