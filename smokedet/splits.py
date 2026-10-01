"""Group-aware splitting: all copies of one photo land in the same split."""

from __future__ import annotations

import random
from collections import Counter


def group_split(groups: list[str], val_frac: float, seed: int = 0) -> list[str]:
    """Assign 'train' or 'val' to each item so no group is split across both.

    Groups are shuffled with a fixed seed and moved to val until val holds at
    least val_frac of the items.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    sizes = Counter(groups)
    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    target = val_frac * len(groups)
    val_groups: set[str] = set()
    count = 0
    for g in order:
        if count >= target:
            break
        val_groups.add(g)
        count += sizes[g]
    return ["val" if g in val_groups else "train" for g in groups]
