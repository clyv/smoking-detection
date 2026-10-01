"""Per-frame, per-person features for stage 2 (pose) and stage 3 (temporal).

Everything is measured in shoulder widths, so the same thresholds work for a
person near the camera and one far away.

COCO-17 keypoint order (used by Ultralytics pose models):
 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear, 5 left shoulder,
 6 right shoulder, 7 left elbow, 8 right elbow, 9 left wrist, 10 right wrist,
 11-16 hips, knees, ankles
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOSE, L_EYE, R_EYE = 0, 1, 2
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10

BASE_FEATURES = [
    "d_left",                 # left wrist to mouth
    "d_right",                # right wrist to mouth
    "d_min",                  # closer wrist to mouth
    "d_min_delta",            # change in d_min since the previous frame
    "l_wrist_above_shoulder",
    "r_wrist_above_shoulder",
    "l_elbow_angle",          # 0..1 (180 degrees = 1 = straight arm)
    "r_elbow_angle",
    "nose_conf",
    "l_wrist_conf",
    "r_wrist_conf",
    "present",                # 1 if the person was detected in this frame
]

# Values used when a keypoint is missing: "hand far away, arm straight".
FILL = {"d_left": 3.0, "d_right": 3.0, "d_min": 3.0, "l_elbow_angle": 1.0, "r_elbow_angle": 1.0}


def feature_names(classes: list[str]) -> list[str]:
    return BASE_FEATURES + [f"obj_{c}" for c in classes]


@dataclass
class FeatureParams:
    kp_conf_thr: float = 0.3
    radius_face: float = 0.8
    radius_hand: float = 0.6
    radius_smoke: float = 1.5
    smoke_class: str = "smoke"


def _ok(conf: np.ndarray, i: int, thr: float) -> bool:
    return conf is not None and conf[i] >= thr


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b (degrees / 180)."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return np.nan
    cos = np.clip(np.dot(v1, v2) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)) / 180.0)


def frame_features(
    kp_xy: np.ndarray,
    kp_conf: np.ndarray | None,
    person_box: np.ndarray,
    objects: list[tuple[str, float, np.ndarray]],
    classes: list[str],
    p: FeatureParams,
) -> np.ndarray:
    """Feature vector for one person in one frame (d_min_delta is filled later).

    kp_xy: (17, 2) pixel keypoints; kp_conf: (17,) or None; person_box: xyxy;
    objects: detections in the frame as (class name, confidence, xyxy).
    """
    kp_xy = np.asarray(kp_xy, dtype=float)
    conf = None if kp_conf is None else np.asarray(kp_conf, dtype=float)
    thr = p.kp_conf_thr
    x1, y1, x2, y2 = [float(v) for v in person_box]

    # scale: shoulder width, else a fraction of the person's height
    scale = np.nan
    if _ok(conf, L_SH, thr) and _ok(conf, R_SH, thr):
        scale = float(np.linalg.norm(kp_xy[L_SH] - kp_xy[R_SH]))
    if not np.isfinite(scale) or scale < 1.0:
        scale = max(0.25 * (y2 - y1), 1.0)

    # mouth: a little below the nose, along the eyes -> nose direction
    mouth = None
    if _ok(conf, NOSE, thr):
        mouth = kp_xy[NOSE].copy()
        if _ok(conf, L_EYE, thr) and _ok(conf, R_EYE, thr):
            mid_eye = (kp_xy[L_EYE] + kp_xy[R_EYE]) / 2
            mouth = kp_xy[NOSE] + 0.6 * (kp_xy[NOSE] - mid_eye)

    def wrist_dist(w: int) -> float:
        if mouth is None or not _ok(conf, w, thr):
            return np.nan
        return float(np.linalg.norm(kp_xy[w] - mouth) / scale)

    d_l, d_r = wrist_dist(L_WR), wrist_dist(R_WR)
    d_min = np.nanmin([d_l, d_r]) if np.isfinite([d_l, d_r]).any() else np.nan

    def above(w: int, s: int) -> float:
        return float(_ok(conf, w, thr) and _ok(conf, s, thr) and kp_xy[w, 1] < kp_xy[s, 1])

    def elbow(s: int, e: int, w: int) -> float:
        if all(_ok(conf, k, thr) for k in (s, e, w)):
            return _angle(kp_xy[s], kp_xy[e], kp_xy[w])
        return np.nan

    # objects near the mouth or a wrist
    wrists = [kp_xy[w] for w in (L_WR, R_WR) if _ok(conf, w, thr)]
    face_top, face_bottom = y1, y1 + 0.35 * (y2 - y1)
    obj_scores = {c: 0.0 for c in classes}
    for name, score, box in objects:
        if name not in obj_scores:
            continue
        b = np.asarray(box, dtype=float)
        centre = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        near = False
        if mouth is not None:
            r = p.radius_smoke if name == p.smoke_class else p.radius_face
            near = np.linalg.norm(centre - mouth) / scale <= r
        else:  # no face keypoints: fall back to the top third of the person box
            near = x1 <= centre[0] <= x2 and face_top <= centre[1] <= face_bottom
        if not near and name != p.smoke_class:
            near = any(np.linalg.norm(centre - w) / scale <= p.radius_hand for w in wrists)
        if near:
            obj_scores[name] = max(obj_scores[name], float(score))

    c = conf if conf is not None else np.ones(17)
    base = [
        d_l, d_r, d_min, 0.0,
        above(L_WR, L_SH), above(R_WR, R_SH),
        elbow(L_SH, L_EL, L_WR), elbow(R_SH, R_EL, R_WR),
        float(c[NOSE]), float(c[L_WR]), float(c[R_WR]),
        1.0,
    ]
    return np.array(base + [obj_scores[k] for k in classes], dtype=np.float32)


def sequence_features(frames: list[np.ndarray | None], n_features: int) -> np.ndarray:
    """Stack per-frame vectors into (T, F), filling gaps and adding d_min_delta.

    A None entry is a frame where the person was not detected: the previous
    values are carried forward with present = 0 and object scores = 0.
    """
    idx = {n: i for i, n in enumerate(BASE_FEATURES)}
    out = np.zeros((len(frames), n_features), dtype=np.float32)
    prev = None
    for t, f in enumerate(frames):
        if f is None:
            row = prev.copy() if prev is not None else np.zeros(n_features, dtype=np.float32)
            row[idx["present"]] = 0.0
            row[len(BASE_FEATURES):] = 0.0
            if prev is None:
                for k, v in FILL.items():
                    row[idx[k]] = v
        else:
            row = np.asarray(f, dtype=np.float32).copy()
            for k, v in FILL.items():
                if not np.isfinite(row[idx[k]]):
                    row[idx[k]] = v
            row[~np.isfinite(row)] = 0.0
        out[t] = row
        prev = row
    d = out[:, idx["d_min"]]
    out[1:, idx["d_min_delta"]] = np.diff(d)
    out[0, idx["d_min_delta"]] = 0.0
    return out
