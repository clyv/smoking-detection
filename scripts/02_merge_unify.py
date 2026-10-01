#!/usr/bin/env python
"""Step 2: merge all sources into one dataset with the unified class list.

    python scripts/02_merge_unify.py --overwrite

- Remaps every source's class ids to configs/taxonomy.yaml via class_map.
- Ignores the sources' own train/valid/test splits (step 4 re-splits by source).
- Drops images whose boxes were ALL dropped by the mapping (e.g. only person
  boxes): they probably show smoking whose object was never labelled, so they
  would teach the model that a visible cigarette is background.
- Writes data/merged/{images,labels}/ and data/merged/manifest.csv, and prints
  which unified classes each source actually labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from smokedet.hashing import roboflow_stem  # noqa: E402
from smokedet.taxonomy import build_id_map, load_sources, load_taxonomy, read_names  # noqa: E402
from smokedet.utils import link_or_copy  # noqa: E402
from smokedet.yolo_io import iter_images, read_labels, write_labels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=ROOT / "configs/sources.yaml", type=Path)
    ap.add_argument("--taxonomy", default=ROOT / "configs/taxonomy.yaml", type=Path)
    ap.add_argument("--raw", default=ROOT / "data/raw", type=Path)
    ap.add_argument("--out", default=ROOT / "data/merged", type=Path)
    ap.add_argument("--overwrite", action="store_true", help="delete an existing merged folder first")
    ap.add_argument("--allow-unmapped", action="store_true", help="drop unmapped classes instead of stopping")
    ap.add_argument("--keep-emptied", action="store_true", help="keep images whose boxes were all dropped")
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)
    sources, _ = load_sources(args.sources)

    if args.out.exists():
        if not args.overwrite:
            sys.exit(f"{args.out} exists. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True)
    (args.out / "labels").mkdir(parents=True)

    rows = []
    coverage: dict[str, Counter] = {}
    problems = []
    for src in sources.values():
        if src.role == "skip":
            continue
        d = args.raw / src.name
        if not (d / "data.yaml").exists():
            problems.append(f"{src.name}: not downloaded (run step 1)")
            continue
        names = read_names(d / "data.yaml")
        id_map, unmapped = build_id_map(names, src.class_map, tax)
        if unmapped and not args.allow_unmapped:
            problems.append(f"{src.name}: classes with no class_map entry: {unmapped}")
            continue
        if all(v is None for v in id_map.values()):
            print(f"[{src.name}] WARNING: every class maps to null; this source adds only background images")

        cov = coverage.setdefault(src.name, Counter())
        n_in = n_emptied = 0
        used_names: set[str] = set()
        for img, lbl in iter_images(d):
            n_in += 1
            boxes = read_labels(lbl)
            new = [(id_map[c], cx, cy, w, h) for c, cx, cy, w, h in boxes if id_map.get(c) is not None]
            orig_empty = not boxes
            emptied = bool(boxes) and not new
            if emptied and not args.keep_emptied:
                n_emptied += 1
                continue
            out_name = f"{src.name}__{img.name}"
            k = 1
            while out_name in used_names:  # same file name in two of the source's splits
                out_name = f"{src.name}__{img.stem}_{k}{img.suffix}"
                k += 1
            used_names.add(out_name)
            out_img = args.out / "images" / out_name
            link_or_copy(img, out_img)
            write_labels(args.out / "labels" / (Path(out_name).stem + ".txt"), new)
            per_class = Counter(tax.classes[c] for c, *_ in new)
            cov.update(per_class)
            rows.append(
                {
                    "image": str(out_img.resolve()),
                    "source": src.name,
                    "role": src.role,
                    "priority": src.priority,
                    "orig_name": img.name,
                    "orig_stem": roboflow_stem(img.name),
                    "n_boxes": len(new),
                    "boxes_per_class": json.dumps(dict(per_class)),
                    "orig_empty": orig_empty,
                    "emptied": emptied,
                }
            )
        print(f"[{src.name}] {n_in} images in, {n_in - n_emptied} kept, {n_emptied} dropped (all boxes unmapped)")

    if problems:
        print("\nStopped because of:\n  - " + "\n  - ".join(problems))
        print("Fix configs/sources.yaml (see audit/report.md) or pass --allow-unmapped.")
        sys.exit(1)
    if not rows:
        sys.exit("No images were merged. Check that sources are downloaded and not all set to role: skip.")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "manifest.csv", index=False)

    print("\nBoxes per unified class, by source (0 = this source never labels that class):")
    cov_df = pd.DataFrame({s: {c: coverage[s].get(c, 0) for c in tax.classes} for s in coverage}).T
    print(cov_df.to_string())
    train_names = [s for s in cov_df.index if sources[s].role == "train"]
    tr = cov_df.loc[train_names] if train_names else cov_df
    partial = [c for c in tax.classes if 0 < (tr[c] > 0).sum() < len(tr)]
    if partial:
        print(
            f"\nNote: {partial} are labelled by some training sources but not others. Unlabelled instances in the "
            "other sources count as background during training. Consider pre-labelling them with "
            "scripts/06_yoloworld_baseline.py --mode prelabel and reviewing the results."
        )
    print(f"\nWrote {len(df)} images to {args.out}")


if __name__ == "__main__":
    main()
