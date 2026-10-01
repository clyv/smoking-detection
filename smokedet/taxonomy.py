"""Unified taxonomy and per-source class remapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .utils import load_yaml

VALID_ROLES = {"train", "test", "skip"}


def norm(name: str) -> str:
    """Case- and whitespace-insensitive class-name key."""
    return " ".join(str(name).strip().lower().split())


@dataclass
class Taxonomy:
    classes: list[str]
    smoking_objects: list[str]
    prompts: dict[str, list[str]] = field(default_factory=dict)

    @property
    def index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    @property
    def smoking_ids(self) -> list[int]:
        return [self.index[c] for c in self.smoking_objects]

    def flat_prompts(self) -> tuple[list[str], dict[int, int]]:
        """All YOLO-World prompts, plus prompt index -> unified class id."""
        prompts: list[str] = []
        to_class: dict[int, int] = {}
        for c in self.classes:
            for p in self.prompts.get(c) or [c]:
                to_class[len(prompts)] = self.index[c]
                prompts.append(p)
        return prompts, to_class


def load_taxonomy(path: str | Path) -> Taxonomy:
    cfg = load_yaml(path)
    classes = [str(c) for c in cfg["classes"]]
    smoking = [str(c) for c in cfg.get("smoking_objects", classes)]
    unknown = set(smoking) - set(classes)
    if unknown:
        raise ValueError(f"smoking_objects not in classes: {sorted(unknown)}")
    return Taxonomy(classes=classes, smoking_objects=smoking, prompts=cfg.get("prompts", {}) or {})


@dataclass
class Source:
    name: str
    workspace: str
    project: str
    version: str
    role: str
    priority: int
    license: str
    class_map: dict[str, str | None]


def load_sources(path: str | Path) -> tuple[dict[str, Source], str]:
    cfg = load_yaml(path)
    out: dict[str, Source] = {}
    for name, s in (cfg.get("sources") or {}).items():
        role = str(s.get("role", "train"))
        if role not in VALID_ROLES:
            raise ValueError(f"{name}: role must be one of {sorted(VALID_ROLES)}, got {role!r}")
        cmap = {norm(k): (None if v is None else str(v)) for k, v in (s.get("class_map") or {}).items()}
        out[name] = Source(
            name=name,
            workspace=str(s["workspace"]),
            project=str(s["project"]),
            version=str(s.get("version", "latest")),
            role=role,
            priority=int(s.get("priority", 100)),
            license=str(s.get("license", "")),
            class_map=cmap,
        )
    return out, str(cfg.get("roboflow_format", "yolov8"))


def read_names(data_yaml: str | Path) -> list[str]:
    """Class names from a YOLO data.yaml (list or {id: name} form)."""
    names = load_yaml(data_yaml).get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ValueError(f"No 'names' in {data_yaml}")


def build_id_map(
    src_names: list[str], class_map: dict[str, str | None], tax: Taxonomy
) -> tuple[dict[int, int | None], list[str]]:
    """Map source class ids to unified ids.

    Returns (mapping, unmapped) where mapping[src_id] is a unified id or None
    (drop), and unmapped lists source names with no entry in class_map.
    """
    mapping: dict[int, int | None] = {}
    unmapped: list[str] = []
    for i, n in enumerate(src_names):
        key = norm(n)
        if key not in class_map:
            unmapped.append(n)
            mapping[i] = None
            continue
        target = class_map[key]
        if target is None:
            mapping[i] = None
        elif target in tax.index:
            mapping[i] = tax.index[target]
        else:
            raise ValueError(f"class_map sends {n!r} to {target!r}, which is not in the taxonomy {tax.classes}")
    return mapping, unmapped
