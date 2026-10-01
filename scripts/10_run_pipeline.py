#!/usr/bin/env python
"""Step 10: run the full 3-stage pipeline on a video.

    python scripts/10_run_pipeline.py --video path/to/clip.mp4
    python scripts/10_run_pipeline.py --video 0            # webcam

Per processed frame: detect smoking objects (stage 1), detect and track people
with keypoints (stage 2). Every hop_s seconds, for each person, score the last
window_s seconds:
  object score = how consistently a smoking object sits at their mouth/hand
  motion score = learned temporal model if temporal_checkpoint is set,
                 otherwise the hand-to-mouth heuristic
then fuse them (stage 3) and smooth decisions into smoking events.

Outputs runs/pipeline/<video>_annotated.mp4 and runs/pipeline/<video>_events.json.
Works before you train anything: with detector.type = yoloworld and no
temporal checkpoint it uses YOLO-World + the motion heuristic.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smokedet.features import BASE_FEATURES, feature_names, sequence_features  # noqa: E402
from smokedet.fusion import EventAggregator, FusionParams, fuse, window_object_score  # noqa: E402
from smokedet.motion import MotionParams, motion_score  # noqa: E402
from smokedet.taxonomy import load_taxonomy  # noqa: E402
from smokedet.utils import load_yaml, pick_device, torch_device  # noqa: E402

COLORS = {"cigarette": (0, 165, 255), "vape": (255, 128, 0), "cigar": (0, 100, 200), "pipe": (180, 0, 180), "smoke": (200, 200, 200)}


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.5) -> None:
    """Text on a filled box just above (x, y), kept inside the frame."""
    import cv2

    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y_top = max(0, y - th - base - 4)
    x = min(max(0, x), max(0, frame.shape[1] - tw - 4))
    cv2.rectangle(frame, (x, y_top), (x + tw + 4, y_top + th + base + 4), color, -1)
    cv2.putText(frame, text, (x + 2, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video file, or a camera index such as 0")
    ap.add_argument("--config", default=ROOT / "configs/pipeline.yaml", type=Path)
    ap.add_argument("--taxonomy", default=ROOT / "configs/taxonomy.yaml", type=Path)
    ap.add_argument("--out", default=ROOT / "runs/pipeline", type=Path)
    ap.add_argument("--show", action="store_true", help="display frames while processing")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import cv2

    from smokedet.frame_processor import FrameProcessor

    cfg = load_yaml(args.config)
    tax = load_taxonomy(args.taxonomy)
    names = feature_names(tax.classes)
    n_feat = len(names)
    d_idx = names.index("d_min")
    obj_cols = [len(BASE_FEATURES) + tax.classes.index(c) for c in tax.smoking_objects]

    proc = FrameProcessor(cfg, tax, device=pick_device(args.device))
    scorer = None
    if cfg.get("temporal_checkpoint"):
        from smokedet.temporal import TemporalScorer

        ck = Path(cfg["temporal_checkpoint"])
        scorer = TemporalScorer(ck if ck.is_absolute() else ROOT / ck, device=torch_device(args.device))
        if scorer.feature_names != names:
            sys.exit("Temporal checkpoint was trained with different features/classes. Retrain step 9.")
    mp = MotionParams(**cfg.get("motion", {}))
    fp = FusionParams(**cfg.get("fusion", {}))
    agg = EventAggregator(**cfg.get("events", {}))

    src = int(args.video) if str(args.video).isdigit() else str(args.video)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / cfg.get("target_fps", 10))))
    eff_fps = fps / stride
    window = int(round(cfg.get("window_s", 6) * eff_fps))
    hop = max(1, int(round(cfg.get("hop_s", 1) * eff_fps)))
    buf_len = max(window, int(round(cfg.get("buffer_s", 20) * eff_fps)))
    max_missing = int(round(cfg.get("max_missing_s", 1.5) * eff_fps))

    stem = "camera" if isinstance(src, int) else Path(src).stem
    args.out.mkdir(parents=True, exist_ok=True)
    writer = None
    buffers: dict[int, deque] = {}
    last_seen: dict[int, int] = {}
    last_scores: dict[int, tuple[float, float]] = {}
    frame_no = step = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % stride:
            frame_no += 1
            continue
        t = frame_no / fps
        people, objects = proc.process(frame)
        seen = set()
        for p in people:
            buffers.setdefault(p.track_id, deque(maxlen=buf_len)).append(p.features)
            last_seen[p.track_id] = step
            seen.add(p.track_id)
        for tid in list(buffers):
            if tid in seen:
                continue
            if step - last_seen[tid] > max_missing:
                agg.close_track(tid, t)
                buffers.pop(tid)
                last_seen.pop(tid)
                last_scores.pop(tid, None)
            else:
                buffers[tid].append(None)

        if step % hop == 0:
            for tid, buf in buffers.items():
                if sum(f is not None for f in buf) < max(3, window // 3):
                    continue
                seq = sequence_features(list(buf), n_feat)
                win = seq[-window:]
                obj = window_object_score(win[:, obj_cols], fp.obj_percentile)
                mot = scorer.score(win) if scorer else motion_score(seq[:, d_idx], eff_fps, window, mp)
                decision, score, reason = fuse(obj, mot, fp)
                agg.update(tid, t, decision, score, obj, mot, reason)
                last_scores[tid] = (obj, mot)

        # drawing
        for name, s, b in objects:
            x1, y1, x2, y2 = map(int, b)
            c = COLORS.get(name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            draw_label(frame, f"{name} {s:.2f}", x1, y1, c, 0.45)
        for p in people:
            x1, y1, x2, y2 = map(int, p.box)
            active = agg.is_active(p.track_id)
            color = (0, 0, 255) if active else (0, 160, 0)
            obj, mot = last_scores.get(p.track_id, (0.0, 0.0))
            label = f"#{p.track_id}{' SMOKING' if active else ''}  obj {obj:.2f}  mot {mot:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if active else 2)
            draw_label(frame, label, x1, y1, color)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(args.out / f"{stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), eff_fps, (w, h))
        writer.write(frame)
        if args.show:
            cv2.imshow("smoking pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_no += 1
        step += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    events = agg.finish(frame_no / fps)
    out_json = args.out / f"{stem}_events.json"
    out_json.write_text(json.dumps({"video": str(args.video), "events": events,
                                    "motion_source": "temporal_model" if scorer else "heuristic",
                                    "detector": cfg["detector"]}, indent=2), encoding="utf-8")
    print(f"{len(events)} smoking event(s). Wrote {out_json} and {args.out / (stem + '_annotated.mp4')}")
    for e in events:
        print(f"  person #{e['track_id']}: {e['start_s']}s -> {e['end_s']}s  (peak object {e['peak_object']}, motion {e['peak_motion']})")


if __name__ == "__main__":
    main()
