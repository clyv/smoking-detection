"""Runs stage 1 (object detector) and stage 2 (pose + tracking) on a frame.

Shared by scripts/08_extract_tracks.py (building training data) and
scripts/10_run_pipeline.py (inference), so features are computed identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureParams, frame_features
from .taxonomy import Taxonomy, norm


@dataclass
class PersonObs:
    track_id: int
    box: np.ndarray            # xyxy
    features: np.ndarray       # frame_features output


class FrameProcessor:
    def __init__(self, cfg: dict, tax: Taxonomy, device: str | None = None):
        from ultralytics import YOLO

        self._YOLO = YOLO
        self.cfg = cfg
        self.tax = tax
        self.device = device
        self.fp = FeatureParams(kp_conf_thr=cfg.get("kp_conf_thr", 0.3), **cfg.get("features", {}))

        det = cfg["detector"]
        self.det_conf = float(det.get("conf", 0.1))
        self.det_imgsz = int(det.get("imgsz", 960))
        self.det = YOLO(det["weights"])
        if det.get("type", "yolo") == "yoloworld":
            if not hasattr(self.det, "set_classes"):
                raise ValueError(f"detector.type is 'yoloworld' but {det['weights']} is not a YOLO-World model. "
                                 "Use e.g. yolov8l-worldv2.pt, or set detector.type to 'yolo' for your own weights.")
            prompts, to_class = tax.flat_prompts()
            self.det.set_classes(prompts)
            self.det_map = {i: tax.classes[c] for i, c in to_class.items()}
        else:
            lookup = {norm(c): c for c in tax.classes}
            self.det_map = {int(k): lookup.get(norm(v)) for k, v in self.det.names.items()}
        self.reset_tracker()

    def reset_tracker(self) -> None:
        """Fresh pose model = fresh tracker state (call between unrelated videos)."""
        self.pose = self._YOLO(self.cfg["pose_model"])

    def detect_objects(self, frame: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        r = self.det.predict(frame, conf=self.det_conf, imgsz=self.det_imgsz, device=self.device, verbose=False)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for c, s, b in zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                name = self.det_map.get(int(c))
                if name is not None:
                    out.append((name, float(s), b))
        return out

    def process(self, frame: np.ndarray) -> tuple[list[PersonObs], list[tuple[str, float, np.ndarray]]]:
        objects = self.detect_objects(frame)
        r = self.pose.track(
            frame,
            persist=True,
            tracker=self.cfg.get("tracker", "bytetrack.yaml"),
            conf=float(self.cfg.get("pose_conf", 0.3)),
            imgsz=int(self.cfg.get("pose_imgsz", 640)),
            device=self.device,
            verbose=False,
        )[0]
        people: list[PersonObs] = []
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            return people, objects
        ids = r.boxes.id.int().cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        kxy = r.keypoints.xy.cpu().numpy()
        kconf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
        for i, tid in enumerate(ids):
            f = frame_features(
                kxy[i], None if kconf is None else kconf[i], boxes[i], objects, self.tax.classes, self.fp
            )
            people.append(PersonObs(track_id=int(tid), box=boxes[i], features=f))
        return people, objects
