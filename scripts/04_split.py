#!/usr/bin/env python
"""Step 4: build train / val / test splits and the Ultralytics data file.

    python scripts/04_split.py

- test  = every kept image from sources with role: test (never seen in training)
          + a slice of the hard negatives, to measure false alarms
- val   = ~10% of the train sources, split by image group (no augmented copy
          of a val photo can sit in train)
- train = the rest + most hard negatives (capped, default 10% of train)

Writes data/splits/{train,val,test}.txt and data/smoking.yaml.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from smokedet.splits import group_split  # noqa: E402
from smokedet.taxonomy import load_taxonomy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxonomy", default=ROOT / "configs/taxonomy.yaml", type=Path)
    ap.add_argument("--merged", default=ROOT / "data/merged", type=Path)
    ap.add_argument("--negatives", default=ROOT / "data/negatives", type=Path)
    ap.add_argument("--data-dir", default=ROOT / "data", type=Path)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--neg-test-frac", type=float, default=0.20, help="share of hard negatives held out for test")
    ap.add_argument("--max-neg-frac", type=float, default=0.10, help="cap on negatives as a share of train images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)
    manifest = args.merged / "manifest_dedup.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found. Run step 3 first.")
    df = pd.read_csv(manifest)
    df = df[df["keep"]].copy()

    df["split"] = ""
    df.loc[df["role"] == "test", "split"] = "test"
    for src, sub in df[df["role"] == "train"].groupby("source"):
        df.loc[sub.index, "split"] = group_split(sub["group_id"].tolist(), args.val_frac, seed=args.seed)

    splits = {s: df.loc[df["split"] == s, "image"].tolist() for s in ("train", "val", "test")}

    neg_counts = {"train": 0, "test": 0}
    neg_manifest = args.negatives / "manifest.csv"
    if neg_manifest.exists():
        neg = pd.read_csv(neg_manifest)
        neg = neg[neg["use"]] if "use" in neg.columns else neg
        paths = neg["image"].tolist()
        random.Random(args.seed).shuffle(paths)
        n_test = int(round(args.neg_test_frac * len(paths)))
        cap = int(args.max_neg_frac * len(splits["train"]))
        test_negs, train_negs = paths[:n_test], paths[n_test:][:cap]
        splits["test"] += test_negs
        splits["train"] += train_negs
        neg_counts = {"train": len(train_negs), "test": len(test_negs)}
    else:
        print(f"(No hard negatives found at {neg_manifest}; run step 5 to add them.)")

    out = args.data_dir / "splits"
    out.mkdir(parents=True, exist_ok=True)
    for s, paths in splits.items():
        (out / f"{s}.txt").write_text("\n".join(str(Path(p).resolve()) for p in paths) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(args.data_dir.resolve()),
        "train": "splits/train.txt",
        "val": "splits/val.txt",
        "test": "splits/test.txt",
        "names": {i: c for i, c in enumerate(tax.classes)},
    }
    with open(args.data_dir / "smoking.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print("\nImages per split (hard negatives included):")
    for s in ("train", "val", "test"):
        neg_note = f" ({neg_counts.get(s, 0)} hard negatives)" if s in neg_counts else ""
        print(f"  {s:<5} {len(splits[s]):>6}{neg_note}")
    print("\nBoxes per class:")
    table = {}
    for s in ("train", "val", "test"):
        c: Counter = Counter()
        for v in df.loc[df["split"] == s, "boxes_per_class"]:
            c.update(json.loads(v))
        table[s] = {k: c.get(k, 0) for k in tax.classes}
    print(pd.DataFrame(table).to_string())
    print("\nTest sources: " + ", ".join(sorted(df.loc[df["split"] == "test", "source"].unique())))
    print(f"Wrote {out}/ and {args.data_dir / 'smoking.yaml'}")


if __name__ == "__main__":
    main()
