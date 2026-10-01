"""Stage 3: combine object and motion evidence, then smooth decisions into events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FusionParams:
    obj_thr: float = 0.35
    mot_thr: float = 0.50
    obj_strong: float = 0.75
    mot_strong: float = 0.95
    obj_percentile: float = 80.0


def window_object_score(obj_matrix: np.ndarray, percentile: float = 80.0) -> float:
    """Robust object score for a window.

    obj_matrix is (T, K): per frame, the confidence of each smoking-object
    class near this person's mouth or hands. Taking a high percentile over
    time, rather than the max, ignores a one-frame false detection while still
    firing when the object is visible in a fair share of frames.
    """
    m = np.asarray(obj_matrix, dtype=float)
    if m.size == 0:
        return 0.0
    per_frame = m.max(axis=1) if m.ndim == 2 else m
    return float(np.percentile(per_frame, percentile))


def fuse(obj: float, mot: float, p: FusionParams) -> tuple[bool, float, str]:
    """Positive when object and motion agree, or when either is very confident.

    Returns (decision, fused score, reason).
    """
    if obj >= p.obj_thr and mot >= p.mot_thr:
        return True, float(np.sqrt(obj * mot)), "object+motion"
    if obj >= p.obj_strong:
        return True, float(obj), "object"
    if mot >= p.mot_strong:
        return True, float(mot), "motion"
    return False, float(np.sqrt(max(obj, 0) * max(mot, 0))), ""


@dataclass
class Event:
    track_id: int
    start_s: float
    end_s: float | None = None
    peak_score: float = 0.0
    peak_obj: float = 0.0
    peak_mot: float = 0.0
    reasons: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "start_s": round(self.start_s, 2),
            "end_s": None if self.end_s is None else round(self.end_s, 2),
            "peak_score": round(self.peak_score, 3),
            "peak_object": round(self.peak_obj, 3),
            "peak_motion": round(self.peak_mot, 3),
            "triggered_by": self.reasons,
        }


class EventAggregator:
    """Per-person hysteresis: k_on of the last n_on windows positive starts an
    event; m_off negative windows in a row end it."""

    def __init__(self, n_on: int = 4, k_on: int = 2, m_off: int = 5):
        self.n_on, self.k_on, self.m_off = n_on, k_on, m_off
        self.history: dict[int, deque] = {}
        self.neg_run: dict[int, int] = {}
        self.open: dict[int, Event] = {}
        self.closed: list[Event] = []

    def update(self, track_id: int, t: float, decision: bool, score: float, obj: float, mot: float, reason: str) -> bool:
        h = self.history.setdefault(track_id, deque(maxlen=self.n_on))
        h.append(decision)
        self.neg_run[track_id] = 0 if decision else self.neg_run.get(track_id, 0) + 1
        ev = self.open.get(track_id)
        if ev is None and sum(h) >= self.k_on:
            ev = self.open[track_id] = Event(track_id=track_id, start_s=t)
        if ev is not None:
            if decision:
                ev.peak_score = max(ev.peak_score, score)
                ev.peak_obj = max(ev.peak_obj, obj)
                ev.peak_mot = max(ev.peak_mot, mot)
                ev.reasons[reason] = ev.reasons.get(reason, 0) + 1
            if self.neg_run[track_id] >= self.m_off:
                ev.end_s = t
                self.closed.append(self.open.pop(track_id))
        return track_id in self.open

    def is_active(self, track_id: int) -> bool:
        return track_id in self.open

    def close_track(self, track_id: int, t: float) -> None:
        ev = self.open.pop(track_id, None)
        if ev is not None:
            ev.end_s = t
            self.closed.append(ev)
        self.history.pop(track_id, None)
        self.neg_run.pop(track_id, None)

    def finish(self, t: float) -> list[dict]:
        for tid in list(self.open):
            self.close_track(tid, t)
        return [e.to_dict() for e in sorted(self.closed, key=lambda e: e.start_s)]
