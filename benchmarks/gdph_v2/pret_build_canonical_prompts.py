from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.gdph_v2.experiment import DEFAULT_OUTPUT_ROOT
from benchmarks.gdph_v2.pret_utils import read_csv, write_csv_atomic, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scale-stable canonical PRET prompt definitions.")
    parser.add_argument("--queries_csv", default=str(DEFAULT_OUTPUT_ROOT / "region_retrieval" / "queries.csv"))
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--image_id", action="append", default=[])
    parser.add_argument("--prompt_sources", nargs="+", default=["oracle_gt_purity", "realistic_box", "scribble_like"])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5])
    args = parser.parse_args()

    queries = read_csv(args.queries_csv)
    if args.image_id:
        requested = set(args.image_id)
        queries = [query for query in queries if query["image_id"] in requested]

    rows = []
    for query in queries:
        for source in args.prompt_sources:
            for shot in args.shots:
                rows.append(
                    {
                        "query_id": query["query_id"],
                        "image_id": query["image_id"],
                        "class_id": int(query["class_id"]),
                        "shot": int(shot),
                        "prompt_source": source,
                        "x0_original": query["x0_original"],
                        "y0_original": query["y0_original"],
                        "x1_original": query["x1_original"],
                        "y1_original": query["y1_original"],
                    }
                )
    if not rows:
        raise RuntimeError("no canonical prompts generated")
    output_csv = Path(args.output_csv)
    write_csv_atomic(output_csv, rows)
    validation = {
        "passed": True,
        "canonical_prompts": len(rows),
        "base_queries": len({row["query_id"] for row in rows}),
        "prompt_sources": sorted({row["prompt_source"] for row in rows}),
        "shots": sorted({int(row["shot"]) for row in rows}),
        "queries_csv": args.queries_csv,
    }
    write_json_atomic(output_csv.with_suffix(".validation.json"), validation)
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
