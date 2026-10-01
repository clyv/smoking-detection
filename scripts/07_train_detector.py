#!/usr/bin/env python
"""Step 7: train the stage-1 object detector and score it on the external test set.

    python scripts/07_train_detector.py --model yolo11s.pt --name y11s_960
    python scripts/07_train_detector.py --model yolo26s.pt --name y26s_960
    python scripts/07_train_detector.py --model yolov8s-worldv2.pt --name world_s   # fine-tuned YOLO-World

Cigarettes are tiny, so the default image size is 960 (1280 helps further if
VRAM allows). batch=-1 lets Ultralytics pick the largest batch that fits.

After training, the best weights are scored on data/splits/test.txt with the
same metrics as step 6, so the YOLO-World baseline and every trained model are
directly comparable. Reports go to runs/detect/<name>/test_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smokedet.eval_utils import class_map_from_model_names, load_ground_truth, predict_images, read_image_list  # noqa: E402
from smokedet.metrics import evaluate_detections, format_report  # noqa: E402
from smokedet.taxonomy import load_taxonomy  # noqa: E402
from smokedet.utils import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--data", default=ROOT / "data/smoking.yaml", type=Path)
    ap.add_argument("--taxonomy", default=ROOT / "configs/taxonomy.yaml", type=Path)
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=float, default=-1, help="-1 = auto; 0.0-1.0 = fraction of GPU memory")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8, help="lower this on Windows if dataloaders hang")
    ap.add_argument("--cache", default="disk", choices=["ram", "disk", "false"])
    ap.add_argument("--conf", type=float, default=0.25, help="operating threshold for image-level metrics")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-only", type=Path, help="skip training; score these weights on the test split")
    args = ap.parse_args()

    from ultralytics import YOLO

    tax = load_taxonomy(args.taxonomy)
    device = pick_device(args.device)
    batch = int(args.batch) if args.batch == -1 or args.batch >= 1 else args.batch

    if args.eval_only:
        best = args.eval_only
        save_dir = best.parent.parent
    else:
        name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
        if "world" in Path(args.model).stem.lower():
            from ultralytics import YOLOWorld

            model = YOLOWorld(args.model)
        else:
            model = YOLO(args.model)
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            patience=args.patience,
            device=device,
            workers=args.workers,
            cache=False if args.cache == "false" else args.cache,
            project=str(ROOT / "runs/detect"),
            name=name,
            cos_lr=True,
            close_mosaic=10,
            seed=0,
            plots=True,
        )
        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"

    print(f"\nScoring {best} on the external test split ...")
    model = YOLO(str(best))
    images = read_image_list(ROOT / "data/splits/test.txt")
    gts = load_ground_truth(images)
    cls_map = class_map_from_model_names(model.names, tax)
    preds = predict_images(model, images, cls_map, conf=0.01, imgsz=args.imgsz, device=device)
    report = evaluate_detections(gts, preds, tax.classes, tax.smoking_ids, conf_thr=args.conf)
    report["weights"] = str(best)
    print("\n" + format_report(report))
    (save_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {save_dir / 'test_report.json'}")
    print("Compare with runs/baselines/yoloworld_*_test.json from step 6.")


if __name__ == "__main__":
    main()
