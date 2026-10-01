#!/usr/bin/env python
"""Step 1: download the Roboflow sources and audit their class names.

    # Windows (PowerShell):  $env:ROBOFLOW_API_KEY="your_key"
    # WSL / Linux:           export ROBOFLOW_API_KEY=your_key
    python scripts/01_download_sources.py                 # download everything, then audit
    python scripts/01_download_sources.py --only kyunghee # one source
    python scripts/01_download_sources.py --audit-only    # re-run the audit on what is downloaded

The audit writes audit/report.md (class names, box and image counts, and what
each class maps to in configs/sources.yaml) plus sample crops per class in
audit/<source>/<class>/. Look at the crops, then fix class_map before step 2.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from smokedet.taxonomy import load_sources, norm, read_names  # noqa: E402
from smokedet.yolo_io import iter_images, read_labels  # noqa: E402


def _version_number(v) -> int:
    raw = getattr(v, "version", None) or getattr(v, "id", "")
    return int(str(raw).rstrip("/").split("/")[-1])


def download(src, raw_dir: Path, fmt: str, api_key: str) -> int:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(src.workspace).project(src.project)
    if src.version == "latest":
        versions = project.versions()
        if not versions:
            raise RuntimeError(f"{src.name}: project has no generated versions")
        ver = max(versions, key=_version_number)
        number = _version_number(ver)
    else:
        number = int(src.version)
        ver = project.version(number)
    dest = raw_dir / src.name
    print(f"[{src.name}] downloading v{number} ({fmt}) -> {dest}")
    ver.download(fmt, location=str(dest), overwrite=False)
    (dest / "_version.txt").write_text(str(number), encoding="utf-8")
    return number


def _crop(img_path: Path, box, pad: float = 0.3) -> Image.Image:
    _, cx, cy, w, h = box
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        bw, bh = w * W * (1 + 2 * pad), h * H * (1 + 2 * pad)
        x1, y1 = max(0, cx * W - bw / 2), max(0, cy * H - bh / 2)
        x2, y2 = min(W, cx * W + bw / 2), min(H, cy * H + bh / 2)
        return im.crop((int(x1), int(y1), int(max(x2, x1 + 2)), int(max(y2, y1 + 2))))


def audit(src, raw_dir: Path, audit_dir: Path, n_crops: int) -> list[str]:
    d = raw_dir / src.name
    if not (d / "data.yaml").exists():
        return [f"## {src.name}\n\nNot downloaded.\n"]
    names = read_names(d / "data.yaml")
    rng = random.Random(0)
    boxes_per_class: Counter = Counter()
    images_per_class: Counter = Counter()
    samples: dict[int, list] = defaultdict(list)
    seen_per_class: Counter = Counter()
    n_images = n_empty = 0

    for img, lbl in iter_images(d):
        n_images += 1
        boxes = read_labels(lbl)
        if not boxes:
            n_empty += 1
        for c in {b[0] for b in boxes}:
            images_per_class[c] += 1
        for b in boxes:
            c = b[0]
            boxes_per_class[c] += 1
            seen_per_class[c] += 1
            if len(samples[c]) < n_crops:
                samples[c].append((img, b))
            else:  # reservoir sampling keeps a uniform sample
                k = rng.randrange(seen_per_class[c])
                if k < n_crops:
                    samples[c][k] = (img, b)

    for c, items in samples.items():
        cname = names[c] if c < len(names) else f"id{c}"
        out = audit_dir / src.name / f"{c:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', cname)[:40]}"
        out.mkdir(parents=True, exist_ok=True)
        for k, (img, b) in enumerate(items):
            try:
                _crop(img, b).save(out / f"{k:02d}.jpg", quality=90)
            except Exception as e:  # noqa: BLE001
                print(f"  crop failed for {img.name}: {e}")

    version = (d / "_version.txt").read_text().strip() if (d / "_version.txt").exists() else "?"
    lines = [
        f"## {src.name}  (role: {src.role}, version {version}, license: {src.license})",
        "",
        f"{n_images} images, {n_empty} with no boxes.",
        "",
        "| id | class name | boxes | images | maps to |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, n in enumerate(names):
        key = norm(n)
        if key in src.class_map:
            target = src.class_map[key] or "drop"
        else:
            target = "**UNMAPPED - decide in sources.yaml**"
        lines.append(f"| {i} | {n} | {boxes_per_class[i]} | {images_per_class[i]} | {target} |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=ROOT / "configs/sources.yaml", type=Path)
    ap.add_argument("--raw", default=ROOT / "data/raw", type=Path)
    ap.add_argument("--audit-dir", default=ROOT / "audit", type=Path)
    ap.add_argument("--only", nargs="*", help="source names to process (default: all except role: skip)")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--crops", type=int, default=24, help="sample crops saved per class")
    args = ap.parse_args()

    sources, fmt = load_sources(args.sources)
    chosen = [s for s in sources.values() if s.role != "skip" and (not args.only or s.name in args.only)]

    if not args.audit_only:
        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            sys.exit("Set ROBOFLOW_API_KEY first (Roboflow > Settings > API Keys).")
        args.raw.mkdir(parents=True, exist_ok=True)
        for s in chosen:
            try:
                download(s, args.raw, fmt, key)
            except Exception as e:  # noqa: BLE001
                print(f"[{s.name}] download FAILED: {e}")

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    report = ["# Source audit", "", "Check every UNMAPPED row and every REVIEW comment in configs/sources.yaml.", ""]
    for s in chosen:
        print(f"[{s.name}] auditing")
        report += audit(s, args.raw, args.audit_dir, args.crops)
    (args.audit_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote {args.audit_dir / 'report.md'} and sample crops under {args.audit_dir}/")


if __name__ == "__main__":
    main()
