#!/usr/bin/env python
"""Step 6: YOLO-World zero-shot baseline, false-positive mining and pre-labelling.

YOLO-World is open-vocabulary: you give it text prompts (configs/taxonomy.yaml
-> prompts) and it detects them without training. It was pretrained on
Objects365, which has one combined "Cigar/Cigarette" class, so cigarettes are
a concept it has seen; vapes and pipes rely on text generalisation only.

Modes:
  eval      score it on a split, with the same metrics as the trained detector
            python scripts/06_yoloworld_baseline.py --mode eval --split test
  mine      run it on unlabelled people images and list its most confident
            smoking detections (false positives there = valuable hard negatives)
            python scripts/06_yoloworld_baseline.py --mode mine --images-dir data/negatives/images
  prelabel  write YOLO labels for chosen classes, to review in CVAT/Roboflow
            (e.g. smoke boxes for sources that never labelled smoke)
            python scripts/06_yoloworld_baseline.py --mode prelabel --images-dir data/raw/takoyati/train/images --classes smoke --out data/prelabels/takoyati
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smokedet.eval_utils import load_ground_truth, predict_images, read_image_list  # noqa: E402
from smokedet.metrics import evaluate_detections, format_report  # noqa: E402
from smokedet.taxonomy import load_taxonomy  # noqa: E402
from smokedet.utils import pick_device  # noqa: E402
from smokedet.yolo_io import IMG_EXTS, write_labels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["eval", "mine", "prelabel"], default="eval")
    ap.add_argument("--model", default="yolov8l-worldv2.pt", help="s/m/l/x sizes; x is most accurate, s fastest")
    ap.add_argument("--taxonomy", default=ROOT / "configs/taxonomy.yaml", type=Path)
    ap.add_argument("--data-dir", default=ROOT / "data", type=Path)
    ap.add_argument("--split", default="test", help="eval: train / val / test (reads data/splits/<split>.txt)")
    ap.add_argument("--images-dir", type=Path, help="mine / prelabel: folder of images")
    ap.add_argument("--classes", nargs="*", help="prelabel: unified classes to write (default: all)")
    ap.add_argument("--out", type=Path, help="output folder (mine / prelabel)")
    ap.add_argument("--conf", type=float, default=0.25, help="operating threshold (image-level metrics, mine, prelabel)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--background-prompt", action="store_true", help="append an empty '' prompt (sometimes helps)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from ultralytics import YOLO

    tax = load_taxonomy(args.taxonomy)
    prompts, to_class = tax.flat_prompts()
    if args.background_prompt:
        prompts = prompts + [""]  # not in to_class, so its detections are ignored
    model = YOLO(args.model)
    model.set_classes(prompts)
    device = pick_device(args.device)
    print(f"YOLO-World {args.model} with prompts {prompts}")

    if args.mode == "eval":
        images = read_image_list(args.data_dir / "splits" / f"{args.split}.txt")
        print(f"Evaluating on {len(images)} images from the {args.split} split ...")
        gts = load_ground_truth(images)
        preds = predict_images(model, images, to_class, conf=0.01, imgsz=args.imgsz, batch=args.batch, device=device)
        report = evaluate_detections(gts, preds, tax.classes, tax.smoking_ids, conf_thr=args.conf)
        print("\n" + format_report(report))
        out = ROOT / "runs/baselines"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"yoloworld_{Path(args.model).stem}_{args.split}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved {path}")
        return

    if not args.images_dir:
        sys.exit("--images-dir is required for mine / prelabel")
    images = sorted(p for p in args.images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    print(f"{len(images)} images in {args.images_dir}")
    preds = predict_images(model, images, to_class, conf=args.conf, imgsz=args.imgsz, batch=args.batch, device=device)

    if args.mode == "mine":
        smoking = set(tax.smoking_ids)
        rows = []
        for p, arr in preds.items():
            s = arr[np.isin(arr[:, 0], list(smoking))] if len(arr) else arr
            if len(s):
                best = s[np.argmax(s[:, 1])]
                rows.append({"image": p, "max_conf": float(best[1]), "class": tax.classes[int(best[0])]})
        df = pd.DataFrame(rows, columns=["image", "max_conf", "class"]).sort_values("max_conf", ascending=False)
        out = args.out or ROOT / "runs/mined"
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "false_positive_candidates.csv", index=False)
        for i, (_, row) in enumerate(df.head(200).iterrows()):
            shutil.copy2(row["image"], out / f"{i:03d}_{row['max_conf']:.2f}_{row['class']}_{Path(row['image']).name}")
        print(f"{len(df)} images with a smoking detection >= {args.conf}. Top 200 copied to {out}/ for review.")
        print("Confirmed non-smoking ones make excellent hard negatives (add them via step 5 --extra-dir).")
        return

    # prelabel
    keep = set(tax.index[c] for c in (args.classes or tax.classes))
    out = args.out or ROOT / "data/prelabels"
    (out / "labels").mkdir(parents=True, exist_ok=True)
    from PIL import Image

    n_boxes = 0
    for p, arr in preds.items():
        with Image.open(p) as im:
            W, H = im.size
        boxes = []
        for c, s, x1, y1, x2, y2 in arr:
            if int(c) in keep:
                boxes.append((int(c), (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
        n_boxes += len(boxes)
        write_labels(out / "labels" / (Path(p).stem + ".txt"), boxes)
    print(f"Wrote {n_boxes} boxes for {sorted(tax.classes[k] for k in keep)} to {out / 'labels'}.")
    print("These are suggestions. Review them in an annotation tool before merging them into a source.")


if __name__ == "__main__":
    main()
