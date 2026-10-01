"""Reading and writing YOLO-format datasets (Roboflow export layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = ("train", "valid", "val", "test")

Box = tuple[int, float, float, float, float]  # class, cx, cy, w, h (normalised 0-1)


def read_labels(path: str | Path) -> list[Box]:
    """Read a YOLO label file. Polygon (segmentation) lines are converted to boxes."""
    p = Path(path)
    if not p.exists():
        return []
    boxes: list[Box] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:
            if len(vals) % 2:
                vals = vals[:-1]
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        boxes.append((cls, cx, cy, w, h))
    return boxes


def write_labels(path: str | Path, boxes: list[Box]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def label_path_for(img_path: str | Path) -> Path:
    """images/xxx.jpg -> labels/xxx.txt (the convention Ultralytics uses)."""
    p = Path(img_path)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def iter_images(dataset_dir: str | Path) -> Iterator[tuple[Path, Path]]:
    """Yield (image, label) pairs from a Roboflow YOLO export, across all its splits."""
    d = Path(dataset_dir)
    roots = [d / s for s in SPLIT_DIRS if (d / s / "images").is_dir()]
    if not roots and (d / "images").is_dir():
        roots = [d]
    for r in roots:
        for img in sorted((r / "images").iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                yield img, r / "labels" / (img.stem + ".txt")


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> tuple[float, float, float, float]:
    return (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
