"""Run an Ultralytics model over a list of images and score it with smokedet.metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .metrics import nms_per_class
from .taxonomy import Taxonomy, norm
from .yolo_io import label_path_for, read_labels, yolo_to_xyxy


def read_image_list(path: str | Path) -> list[Path]:
    return [Path(l.strip()) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_ground_truth(images: list[Path]) -> dict:
    gts = {}
    for img in images:
        with Image.open(img) as im:
            W, H = im.size
        rows = [[c, *yolo_to_xyxy(cx, cy, w, h, W, H)] for c, cx, cy, w, h in read_labels(label_path_for(img))]
        gts[str(img)] = {"w": W, "h": H, "boxes": np.array(rows, dtype=float).reshape(-1, 5)}
    return gts


def class_map_from_model_names(model_names: dict | list, tax: Taxonomy) -> dict[int, int | None]:
    """Map a trained model's class ids to taxonomy ids by name (unknown names are ignored)."""
    names = dict(enumerate(model_names)) if isinstance(model_names, list) else model_names
    lookup = {norm(c): i for i, c in enumerate(tax.classes)}
    return {int(k): lookup.get(norm(v)) for k, v in names.items()}


def predict_images(
    model,
    images: list[Path],
    cls_map: dict[int, int | None],
    conf: float = 0.01,
    imgsz: int = 960,
    batch: int = 16,
    device: str | None = None,
) -> dict:
    """Predictions as {image path: array (m, 6) [cls, conf, x1, y1, x2, y2]} in taxonomy ids."""
    preds: dict = {}
    for i in range(0, len(images), batch):
        chunk = [str(p) for p in images[i : i + batch]]
        results = model.predict(chunk, conf=conf, imgsz=imgsz, device=device, verbose=False)
        for path, r in zip(chunk, results):
            rows = []
            if r.boxes is not None and len(r.boxes):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                xyxy = r.boxes.xyxy.cpu().numpy()
                for c, s, b in zip(cls, cf, xyxy):
                    t = cls_map.get(int(c))
                    if t is not None:
                        rows.append([t, float(s), *b.tolist()])
            arr = np.array(rows, dtype=float).reshape(-1, 6)
            preds[path] = nms_per_class(arr) if len(arr) else arr
    return preds
