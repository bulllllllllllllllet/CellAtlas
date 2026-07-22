# Phase 5: Prompt encoder

The frozen default input is the Phase-4 `fine_only` path: 10x Phase-3
cell-aware tokens `[64, 256]`. Phase-4 hierarchical cross-scale tokens are not
the default backbone.

Before training, generate and manually review prompt episodes against HE and
GT. The first auditable contract maps raw positive/negative points to fine
region slots, then passes the selected region tokens and their centroids to the
prompt set encoder. Unknown regions (`label=255`) cannot be prompts and remain
ignored by the binary target.

The visualization script deliberately uses validation only and never reads the
test split. Formal training must not start until these panels are approved.

The training path uses an explicit `eligibility_index.parquet`; it never skips
an infeasible patch inside a DataLoader worker. Build this index from train and
validation only with `tools/build_prompt_eligibility.py`, then pass it to
`train_phase5.py --eligibility-index`. The manifest fixes one connected clean
majority-label prompt set for every feasible patch/target/size combination.
Negative slots are selected online as the spatially nearest target-vs-rest
regions.

The cached training labels contain region majority class but not exact region
purity. Consequently, cached training episodes are treated as noisy prompts;
the separately reviewed validation visualization remains the clean
`purity >= 0.90` audit set. Default sampling mixes point/small/large episodes at
40/35/25 percent and keeps Phase-4 middle/coarse parents out of the main path.

`PromptRegionModel` produces fine-region logits, a task token `[B,256]`, and
task-aware tokens `[B,64,256]` for Phase 6. Training monitors balanced BCE,
region Dice, ranking loss, and recall on target regions not covered by the
positive prompt. Validation also records predicted-positive fraction and
class-/prompt-size-stratified Dice, IoU, recall, and specificity so aggregate
metrics cannot hide rare-class or prompt-scale collapse. Checkpoints include
model, optimizer, scheduler, AMP scaler, config, and complete metric history.
