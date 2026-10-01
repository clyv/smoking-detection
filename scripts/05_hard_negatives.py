#!/usr/bin/env python
"""Step 5: collect hard negatives (people doing smoking-like things without smoking).

    pip install fiftyone
    python scripts/05_hard_negatives.py --max 3000
    python scripts/05_hard_negatives.py --max 3000 --extra-dir path/to/my_negatives

Pulls COCO-2017 images that contain a person AND an object people bring to
their face or hold like a cigarette: cups, bottles, phones, toothbrushes,
cutlery, finger food. Your own photos (inhalers, pens, cold-breath vapour,
straws) can be added with --extra-dir.

Every candidate is screened with YOLO-World for smoking objects. Anything it
flags goes to data/negatives/review/ instead of being used, so a real smoker
in a COCO photo never becomes a "negative". Check that folder by hand.

Writes data/negatives/{images,labels}/ (empty label files = background) and
data/negatives/manifest.csv. Step 4 then splits them between train and test.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from smokedet.taxonomy import load_taxonomy  # noqa: E402
from smokedet.utils import pick_device  # noqa: E402
from smokedet.yolo_io import IMG_EXTS, write_labels  # noqa: E402

CONFUSERS = [
    "cup", "bottle", "wine glass", "cell phone", "toothbrush",
    "fork", "spoon", "sandwich", "hot dog", "pizza", "donut", "banana", "remote",
]


def coco_candidates(n: int, seed: int) -> list[tuple[Path, str]]:
    import fiftyone.zoo as foz

    ds = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=CONFUSERS,
        max_samples=n,
        shuffle=True,
        seed=seed,
    )
    out = []
    for sample in ds:
        dets = sample["ground_truth"].detections if sample["ground_truth"] else []
        labels = {d.label for d in dets}
        hits = sorted(labels & set(CONFUSERS))
        if "person" in labels and hits:
            out.append((Path(sample.filepath), "coco:" + "|".join(hits)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taxonomy", default=ROOT / "configs/taxonomy.yaml", type=Path)
    ap.add_argument("--out", default=ROOT / "data/negatives", type=Path)
    ap.add_argument("--max", type=int, default=3000, help="max COCO negatives to keep")
    ap.add_argument("--extra-dir", type=Path, help="folder of your own negative photos")
    ap.add_argument("--no-coco", action="store_true")
    ap.add_argument("--screen-model", default="yolov8l-worldv2.pt")
    ap.add_argument("--screen-conf", type=float, default=0.15, help="YOLO-World confidence that sends an image to review")
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)
    candidates: list[tuple[Path, str]] = []
    if not args.no_coco:
        print("Loading COCO candidates through FiftyOne (first run downloads images) ...")
        # ask for extra samples: some lack a person or get flagged by the screen
        candidates += coco_candidates(int(args.max * 2.5), args.seed)[: int(args.max * 1.3)]
    if args.extra_dir:
        candidates += [(p, "extra") for p in sorted(args.extra_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not candidates:
        sys.exit("No candidates. Install fiftyone or pass --extra-dir.")
    print(f"{len(candidates)} candidates")

    screen = None
    if not args.no_screen:
        from ultralytics import YOLO

        screen = YOLO(args.screen_model)
        prompts, to_class = tax.flat_prompts()
        smoking_prompt_ids = [i for i, c in to_class.items() if c in tax.smoking_ids]
        screen.set_classes(prompts)
        device = pick_device(args.device)

    for sub in ("images", "labels", "review"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    n_coco = 0
    for k, (path, origin) in enumerate(candidates):
        is_coco = origin.startswith("coco")
        if is_coco and n_coco >= args.max:
            continue
        max_conf = 0.0
        if screen is not None:
            r = screen.predict(str(path), conf=0.05, device=device, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                smoking = [s for c, s in zip(cls, cf) if c in smoking_prompt_ids]
                max_conf = float(max(smoking)) if smoking else 0.0
        name = f"{'coco' if is_coco else 'extra'}__{k:06d}_{path.name}"
        if max_conf >= args.screen_conf:
            shutil.copy2(path, args.out / "review" / name)
            rows.append({"image": str((args.out / "review" / name).resolve()), "origin": origin, "screen_conf": max_conf, "use": False})
            continue
        dst = args.out / "images" / name
        shutil.copy2(path, dst)
        write_labels(args.out / "labels" / (dst.stem + ".txt"), [])
        rows.append({"image": str(dst.resolve()), "origin": origin, "screen_conf": max_conf, "use": True})
        n_coco += int(is_coco)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "manifest.csv", index=False)
    print(f"Kept {int(df['use'].sum())} negatives; {int((~df['use']).sum())} sent to {args.out / 'review'} for a manual look.")
    print("If a review image is truly smoke-free, move it to images/, add an empty label file, and set use=True in the manifest.")


if __name__ == "__main__":
    main()
