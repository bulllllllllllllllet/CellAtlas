from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.gdph_v2.pret_aaai_visual_summary import DEFAULT_NEXT_ROOT, _combine_case_panels
from benchmarks.gdph_v2.pret_utils import PRET_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine each AAAI visual-summary case's eight PNG panels into one grid image.")
    parser.add_argument("--next_root", default=DEFAULT_NEXT_ROOT)
    args = parser.parse_args()

    visual_root = Path(args.next_root) / PRET_DIR / "visual_summary"
    if not visual_root.exists():
        raise FileNotFoundError(f"visual summary directory not found: {visual_root}")
    case_dirs = sorted(path.parent for path in visual_root.glob("class_*/*/*/08_patch_id_map.png"))
    outputs = []
    skipped = []
    for case_dir in case_dirs:
        output = _combine_case_panels(case_dir)
        if output is None:
            skipped.append(str(case_dir))
        else:
            outputs.append(str(output))
            print(f"combined {output}", flush=True)
    print(json.dumps({"combined": len(outputs), "skipped": skipped, "visual_root": str(visual_root)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
