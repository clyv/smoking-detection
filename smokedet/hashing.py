"""Perceptual hashing and near-duplicate grouping."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash_pair(path: str | Path) -> tuple[int, int]:
    """64-bit perceptual hash of an image and of its mirror image.

    The mirror hash catches horizontally flipped copies, which Roboflow
    augmentation produces and which a plain pHash would miss.
    """
    import imagehash
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        h = imagehash.phash(im, hash_size=8)
        hm = imagehash.phash(ImageOps.mirror(im), hash_size=8)
    return int(str(h), 16), int(str(hm), 16)


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    v = x.view(np.uint8).reshape(*x.shape, 8)
    return _LUT[v].sum(-1)


def near_duplicate_pairs(
    hashes: np.ndarray, mirror: np.ndarray, threshold: int = 6, chunk: int = 256
) -> list[tuple[int, int]]:
    """All pairs (i, j), i < j, whose hashes differ in at most `threshold` bits.

    An image's own hash and its mirror hash are both compared against every
    other image's hash. Runs in chunks so memory stays around chunk * N * 8 bytes.
    """
    h = np.asarray(hashes, dtype=np.uint64)
    m = np.asarray(mirror, dtype=np.uint64)
    n = len(h)
    pairs: list[tuple[int, int]] = []
    for start in range(0, n, chunk):
        a = h[start : start + chunk, None]
        am = m[start : start + chunk, None]
        d = np.minimum(_popcount(a ^ h[None, :]), _popcount(am ^ h[None, :]))
        ii, jj = np.nonzero(d <= threshold)
        ii = ii + start
        keep = jj > ii
        pairs.extend(zip(ii[keep].tolist(), jj[keep].tolist()))
    return pairs


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def roboflow_stem(filename: str) -> str:
    """Original image name behind a Roboflow export file.

    Roboflow names files like 'IMG_0042_jpg.rf.8c1f...jpg'; every augmented
    copy of IMG_0042 shares the part before '.rf.'. Grouping on it keeps all
    copies of one photo in the same split.
    """
    name = Path(filename).name
    if ".rf." in name:
        return name.split(".rf.")[0]
    return Path(name).stem
