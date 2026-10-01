"""Detection metrics that do not depend on Ultralytics.

Used for the YOLO-World baseline and the trained detector alike, so both are
scored the same way. Reports:
  - per-class AP@0.5 and mAP@0.5
  - image-level "is there a smoking object?" precision / recall / false-positive rate
  - recall by object size (small objects are the main failure mode)

Data formats (pixel coordinates):
  gts[image_id]   = {"w": W, "h": H, "boxes": array (n, 5) [cls, x1, y1, x2, y2]}
  preds[image_id] = array (m, 6) [cls, conf, x1, y1, x2, y2]
"""

from __future__ import annotations

import math

import numpy as np

SIZE_BUCKETS = (("small", 0, 32), ("medium", 32, 96), ("large", 96, math.inf))  # sqrt(area) in px


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (n, 4) and b (m, 4) in xyxy -> (n, m)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms_per_class(preds: np.ndarray, iou_thr: float = 0.6) -> np.ndarray:
    """Class-wise NMS on [cls, conf, x1, y1, x2, y2] rows (merges synonym prompts)."""
    preds = np.asarray(preds, dtype=float).reshape(-1, 6)
    keep_rows = []
    for c in np.unique(preds[:, 0]):
        p = preds[preds[:, 0] == c]
        p = p[np.argsort(-p[:, 1])]
        while len(p):
            keep_rows.append(p[0])
            if len(p) == 1:
                break
            ious = box_iou(p[:1, 2:], p[1:, 2:])[0]
            p = p[1:][ious < iou_thr]
    return np.array(keep_rows).reshape(-1, 6)


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP (area under the precision envelope)."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-np.asarray(conf))
    tp = np.asarray(tp, dtype=float)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _size_bucket(box: np.ndarray) -> str:
    s = math.sqrt(max(box[2] - box[0], 0) * max(box[3] - box[1], 0))
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= s < hi:
            return name
    return "large"


def evaluate_detections(
    gts: dict,
    preds: dict,
    class_names: list[str],
    smoking_ids: list[int],
    iou_thr: float = 0.5,
    conf_thr: float = 0.25,
) -> dict:
    """Score predictions against ground truth. See module docstring for formats."""
    nc = len(class_names)
    per_class_tp: list[list[float]] = [[] for _ in range(nc)]
    per_class_conf: list[list[float]] = [[] for _ in range(nc)]
    n_gt = np.zeros(nc, dtype=int)
    size_hits = {name: [0, 0] for name, _, _ in SIZE_BUCKETS}  # [found, total] at conf_thr
    img = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    smoking = set(smoking_ids)

    for image_id, g in gts.items():
        gboxes = np.asarray(g["boxes"], dtype=float).reshape(-1, 5)
        p = np.asarray(preds.get(image_id, np.zeros((0, 6))), dtype=float).reshape(-1, 6)

        for c in range(nc):
            gc = gboxes[gboxes[:, 0] == c, 1:]
            pc = p[p[:, 0] == c]
            pc = pc[np.argsort(-pc[:, 1])]
            n_gt[c] += len(gc)
            matched = np.zeros(len(gc), dtype=bool)
            ious = box_iou(pc[:, 2:], gc) if len(pc) and len(gc) else np.zeros((len(pc), len(gc)))
            for k in range(len(pc)):
                hit = 0.0
                if len(gc):
                    cand = np.where(~matched, ious[k], -1.0)
                    j = int(np.argmax(cand))
                    if cand[j] >= iou_thr:
                        matched[j] = True
                        hit = 1.0
                per_class_tp[c].append(hit)
                per_class_conf[c].append(pc[k, 1])

            # recall by size at the operating threshold
            confident = pc[pc[:, 1] >= conf_thr]
            if len(gc):
                found = (box_iou(gc, confident[:, 2:]) >= iou_thr).any(1) if len(confident) else np.zeros(len(gc), bool)
                for box, f in zip(gc, found):
                    b = _size_bucket(box)
                    size_hits[b][1] += 1
                    size_hits[b][0] += int(f)

        gt_pos = bool(np.isin(gboxes[:, 0], list(smoking)).any()) if len(gboxes) else False
        pr_pos = bool((np.isin(p[:, 0], list(smoking)) & (p[:, 1] >= conf_thr)).any()) if len(p) else False
        key = ("tp" if pr_pos else "fn") if gt_pos else ("fp" if pr_pos else "tn")
        img[key] += 1

    ap = {class_names[c]: average_precision(np.array(per_class_tp[c]), np.array(per_class_conf[c]), int(n_gt[c])) for c in range(nc)}
    valid = [v for v in ap.values() if not math.isnan(v)]
    tp, fp, fn, tn = img["tp"], img["fp"], img["fn"], img["tn"]
    return {
        "mAP50": float(np.mean(valid)) if valid else float("nan"),
        "AP50": ap,
        "n_gt": {class_names[c]: int(n_gt[c]) for c in range(nc)},
        "image_level": {
            "conf_thr": conf_thr,
            **img,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        },
        "recall_by_size": {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in size_hits.items()},
        "n_by_size": {k: v[1] for k, v in size_hits.items()},
    }


def format_report(r: dict) -> str:
    lines = [f"mAP@0.5: {r['mAP50']:.3f}", "", "class        AP@0.5   boxes"]
    for c, v in r["AP50"].items():
        lines.append(f"{c:<12} {v:6.3f}   {r['n_gt'][c]}")
    il = r["image_level"]
    lines += [
        "",
        f"image level @conf {il['conf_thr']}: precision {il['precision']:.3f}  recall {il['recall']:.3f}  "
        f"false-positive rate {il['false_positive_rate']:.3f}  (tp {il['tp']} fp {il['fp']} fn {il['fn']} tn {il['tn']})",
        "recall by object size: "
        + "  ".join(f"{k} {v:.3f} (n={r['n_by_size'][k]})" for k, v in r["recall_by_size"].items()),
    ]
    return "\n".join(lines)
