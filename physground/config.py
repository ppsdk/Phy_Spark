from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

VALID_HEAD_KINDS = {"classification", "regression", "bce"}
VALID_HEAD_GROUPS = {"state", "property", "relation", "constraint", "delta"}


@dataclass
class HeadSpec:
    name: str
    group: str
    kind: str
    weight: float = 1.0
    num_labels: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("head name must be non-empty")
        if self.group not in VALID_HEAD_GROUPS:
            raise ValueError(
                f"Unsupported head group for {self.name}: {self.group}. "
                f"Expected one of {sorted(VALID_HEAD_GROUPS)}"
            )
        if self.kind not in VALID_HEAD_KINDS:
            raise ValueError(f"Unsupported head kind: {self.kind}")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError(f"head {self.name} weight must be finite and >=0")
        if self.num_labels < 1:
            raise ValueError(f"head {self.name} needs num_labels>=1")
        if self.kind == "classification" and self.num_labels < 2:
            raise ValueError(f"classification head {self.name} needs num_labels>=2")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeadSpec:
        if not isinstance(data, dict):
            raise ValueError(f"head specification must be a mapping, got {type(data).__name__}")
        if "name" not in data:
            raise ValueError("head specification is missing required field 'name'")
        kind = data.get("kind", "regression")
        num_labels = int(data.get("num_labels", 1))
        return cls(
            name=str(data["name"]),
            group=str(data.get("group", "state")),
            kind=kind,
            weight=float(data.get("weight", 1.0)),
            num_labels=num_labels,
        )


@dataclass
class GroundingConfig:
    heads: list[HeadSpec] = field(default_factory=list)
    lm_weight: float = 1.0
    jepa_weight: float = 0.0
    jepa_dim: int = 0
    hidden_dropout: float = 0.0

    def __post_init__(self) -> None:
        names = [head.name for head in self.heads]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate grounding head names: {', '.join(duplicates)}")
        for name, value in (("lm_weight", self.lm_weight), ("jepa_weight", self.jepa_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >=0")
        if not math.isfinite(self.hidden_dropout) or not 0 <= self.hidden_dropout < 1:
            raise ValueError("hidden_dropout must be in [0, 1)")
        if self.jepa_dim < 0:
            raise ValueError("jepa_dim must be >=0")
        if self.jepa_weight > 0 and self.jepa_dim <= 0:
            raise ValueError("jepa_dim must be >0 when jepa_weight>0")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GroundingConfig:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"grounding configuration must be a mapping, got {type(data).__name__}")
        heads = data.get("heads", [])
        if not isinstance(heads, list):
            raise ValueError("grounding heads must be a list")
        return cls(
            heads=[HeadSpec.from_dict(x) for x in heads],
            lm_weight=float(data.get("lm_weight", 1.0)),
            jepa_weight=float(data.get("jepa_weight", 0.0)),
            jepa_dim=int(data.get("jepa_dim", 0)),
            hidden_dropout=float(data.get("hidden_dropout", 0.0)),
        )
