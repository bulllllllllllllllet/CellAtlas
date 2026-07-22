# V4 visualization

Validation-only figures live in this package. Generated images and metadata
must be written below `/nfs-medical3/zyh/v4/`; source data and sealed test
artifacts are never modified.

The prompt-to-mask workflow entry point is:

```bash
conda run -n aligner python -m benchmarks.v4.visualization.visualize_prompt_workflow --help
```
