from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HeadSpec:
    name: str
    group: str
    kind: str
    weight: float = 1.0
    num_labels: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadSpec":
        kind = data.get("kind", "regression")
        num_labels = int(data.get("num_labels", 1))
        if kind == "classification" and num_labels < 2:
            raise ValueError(f"classification head {data.get('name')} needs num_labels>=2")
        if kind not in {"classification", "regression", "bce"}:
            raise ValueError(f"Unsupported head kind: {kind}")
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

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GroundingConfig":
        data = data or {}
        return cls(
            heads=[HeadSpec.from_dict(x) for x in data.get("heads", [])],
            lm_weight=float(data.get("lm_weight", 1.0)),
            jepa_weight=float(data.get("jepa_weight", 0.0)),
            jepa_dim=int(data.get("jepa_dim", 0)),
            hidden_dropout=float(data.get("hidden_dropout", 0.0)),
        )
