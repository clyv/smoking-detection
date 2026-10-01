#!/usr/bin/env python
"""Step 3: find duplicate images within and across sources.

    python scripts/03_dedupe.py              # default: up to 6 of 64 hash bits may differ
    python scripts/03_dedupe.py --threshold 4

Groups images that are the same photo:
  - Roboflow augmented copies (same name before '.rf.'), and
  - near-identical images by perceptual hash, including mirrored copies.

Then, per group:
  - spans a test source AND a train source -> drop the train copies (keeps the test set honest)
  - spans several train sources            -> keep only the highest-priority source's copies
  - lives in one source                    -> keep all, but they share a group_id so step 4
                                              never puts them in different splits

Writes data/merged/manifest_dedup.csv (adds phash, group_id, keep, drop_reason)
and data/merged/duplicates_sample.csv for spot checks. Nothing is deleted from disk.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smokedet.hashing import UnionFind, near_duplicate_pairs, phash_pair  # noqa: E402


def _safe_hash(path: str) -> tuple[int, int]:
    try:
        return phash_pair(path)
    except Exception:  # noqa: BLE001  unreadable image: give it a unique, never-matching hash
        h = abs(hash(path)) & 0xFFFFFFFFFFFFFFFF
        return h, h


def resolve_groups(df: pd.DataFrame, pairs: list[tuple[int, int]]) -> pd.DataFrame:
    """Union Roboflow-stem groups and hash pairs, then decide keep/drop per group."""
    n = len(df)
    uf = UnionFind(n)
    first_of: dict[tuple[str, str], int] = {}
    for i, key in enumerate(zip(df["source"], df["orig_stem"])):
        if key in first_of:
            uf.union(first_of[key], i)
        else:
            first_of[key] = i
    for i, j in pairs:
        uf.union(i, j)

    roots = np.array([uf.find(i) for i in range(n)])
    df = df.copy()
    df["group_id"] = [f"g{r}" for r in roots]
    keep = np.ones(n, dtype=bool)
    reason = np.array([""] * n, dtype=object)

    for _, idx in df.groupby("group_id").indices.items():
        sub = df.iloc[idx]
        if sub["source"].nunique() == 1:
            continue
        alive = np.ones(len(idx), dtype=bool)
        roles = set(sub["role"])
        if "test" in roles and "train" in roles:
            leak = (sub["role"] == "train").to_numpy()
            keep[idx[leak]] = False
            reason[idx[leak]] = "test_leak"
            alive &= ~leak
        remaining = sub[alive]
        if remaining["source"].nunique() > 1:
            best = remaining["priority"].min()
            lose = alive & (sub["priority"] > best).to_numpy()
            keep[idx[lose]] = False
            reason[idx[lose]] = "cross_source_dup"

    df["keep"] = keep
    df["drop_reason"] = reason
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", default=ROOT / "data/merged", type=Path)
    ap.add_argument("--threshold", type=int, default=6, help="max differing bits (of 64) to call two images duplicates")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.merged / "manifest.csv")
    print(f"Hashing {len(df)} images with {args.workers} workers ...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        hashes = list(ex.map(_safe_hash, df["image"].tolist(), chunksize=64))
    h = np.array([a for a, _ in hashes], dtype=np.uint64)
    hm = np.array([b for _, b in hashes], dtype=np.uint64)
    df["phash"] = [f"{int(x):016x}" for x in h]

    print("Comparing hashes ...")
    pairs = near_duplicate_pairs(h, hm, threshold=args.threshold)
    df = resolve_groups(df, pairs)
    df.to_csv(args.merged / "manifest_dedup.csv", index=False)

    cross = [(i, j) for i, j in pairs if df.at[i, "source"] != df.at[j, "source"]]
    sample = pd.DataFrame(
        [{"a": df.at[i, "image"], "b": df.at[j, "image"], "source_a": df.at[i, "source"], "source_b": df.at[j, "source"]} for i, j in cross[:300]]
    )
    sample.to_csv(args.merged / "duplicates_sample.csv", index=False)

    print(f"\n{len(pairs)} near-duplicate pairs ({len(cross)} across sources); {df['group_id'].nunique()} groups.")
    dropped = df[~df["keep"]]
    if len(dropped):
        print("Dropped images by reason and source:")
        print(dropped.groupby(["drop_reason", "source"]).size().to_string())
    else:
        print("Nothing dropped.")
    print(f"Kept {int(df['keep'].sum())} of {len(df)}. Spot-check {args.merged / 'duplicates_sample.csv'}.")
    print("If unrelated images show up as pairs, lower --threshold; if obvious copies are missed, raise it.")


if __name__ == "__main__":
    main()
