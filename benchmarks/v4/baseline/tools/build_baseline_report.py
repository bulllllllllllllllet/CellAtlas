#!/usr/bin/env python3
"""Build an aligned comparison table and paired bootstrap confidence intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchmarks.v4.phase_6_mask_decoder.src.evaluation import dice_from_counts
from benchmarks.v4.baseline.common import atomic_json, exact_occurrence_alignment, new_output_directory, summarize_metrics, timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="YAML method descriptors and metric paths")
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/v4/baseline"))
    parser.add_argument("--timestamp")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def paired_bootstrap(candidate: pd.DataFrame, reference: pd.DataFrame, samples: int, seed: int) -> dict:
    n = len(candidate); rng = np.random.default_rng(seed)
    macro = np.empty(samples, dtype=np.float64); pooled = np.empty(samples, dtype=np.float64)
    c_episode = candidate["episode_dice"].to_numpy(float); r_episode = reference["episode_dice"].to_numpy(float)
    c_counts = candidate[["tp", "fp", "fn"]].to_numpy(np.int64); r_counts = reference[["tp", "fp", "fn"]].to_numpy(np.int64)
    for index in range(samples):
        draw = rng.integers(0, n, size=n)
        macro[index] = float((c_episode[draw] - r_episode[draw]).mean())
        c = c_counts[draw].sum(0); r = r_counts[draw].sum(0)
        pooled[index] = float(dice_from_counts(c[0], c[1], c[2]) - dice_from_counts(r[0], r[1], r[2]))
    return {
        "samples": samples,
        "macro_episode_dice_difference": float((c_episode - r_episode).mean()),
        "macro_episode_dice_95ci": np.quantile(macro, [0.025, 0.975]).tolist(),
        "pooled_pixel_dice_difference": float(
            dice_from_counts(*c_counts.sum(0)) - dice_from_counts(*r_counts.sum(0))
        ),
        "pooled_pixel_dice_95ci": np.quantile(pooled, [0.025, 0.975]).tolist(),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    values = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(columns[index]), *(len(row[index]) for row in values)) for index in range(len(columns))]
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(columns)) + " |"
    rule = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in values]
    return "\n".join([header, rule, *body])


def main() -> None:
    args = parse_args(); stamp = timestamp(args.timestamp)
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    methods = config.get("methods", [])
    if not methods or len({method["name"] for method in methods}) != len(methods):
        raise ValueError("report config requires uniquely named methods")
    frames = {method["name"]: pd.read_parquet(method["metrics"]).sort_values("occurrence_order").reset_index(drop=True) for method in methods}
    exact_occurrence_alignment(list(frames.values()))
    if any(len(frame) != int(config.get("expected_episodes", 4000)) for frame in frames.values()):
        raise ValueError("one or more methods do not contain the expected episode count")
    summaries = {name: summarize_metrics(frame) for name, frame in frames.items()}
    rows = []
    for method in methods:
        summary = summaries[method["name"]]
        rows.append({
            "Method": method["name"], "Prompt": method["prompt"], "Multiscale": bool(method["multiscale"]),
            "Cell-aware": bool(method["cell_aware"]), "Dice": summary["pooled_pixel_dice"],
            "Boundary F1": summary["boundary_f1_2px"],
            "Point": summary["by_prompt_size"].get("point", {}).get("pooled_pixel_dice", np.nan),
            "Small": summary["by_prompt_size"].get("small", {}).get("pooled_pixel_dice", np.nan),
            "Large": summary["by_prompt_size"].get("large", {}).get("pooled_pixel_dice", np.nan),
            "Coverage": summary["coverage"], "Main-table eligible": summary["failed"] == 0 and summary["abstained"] == 0,
        })
    reference_name = str(config["reference_method"])
    if reference_name not in frames:
        raise ValueError(f"reference method {reference_name!r} absent")
    bootstrap = {
        name: paired_bootstrap(frame, frames[reference_name], args.bootstrap_samples, args.seed + index)
        for index, (name, frame) in enumerate(frames.items()) if name != reference_name
    }
    output = new_output_directory(args.output_root, "comparison_report", stamp)
    table = pd.DataFrame(rows)
    table.to_csv(output / f"main_table_{stamp}.csv", index=False)
    detail = {"timestamp": stamp, "reference_method": reference_name, "summaries": summaries, "paired_bootstrap": bootstrap}
    atomic_json(output / f"comparison_{stamp}.json", detail)
    markdown = ["# Baseline comparison", "", markdown_table(table), "", "## Paired bootstrap versus " + reference_name, ""]
    for name, result in bootstrap.items():
        markdown.append(f"- {name}: pooled ΔDice={result['pooled_pixel_dice_difference']:.6f}, 95% CI={result['pooled_pixel_dice_95ci']}")
    (output / f"comparison_{stamp}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "methods": list(frames), "reference": reference_name}, indent=2))


if __name__ == "__main__":
    main()
