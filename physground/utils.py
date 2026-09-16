from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML config: {path}")
    return data


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_choice(text: str) -> str | None:
    """Extract A/B/C/D from a model response without accepting arbitrary letters."""
    if text is None:
        return None
    text = text.strip()
    patterns = [
        r"^(?:answer\s*[:\-]?\s*)?([ABCD])(?:\b|[.)])",
        r"\b(?:option|choice|answer)\s*[:\-]?\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]
    upper = text.upper()
    for pat in patterns:
        m = re.search(pat, upper)
        if m:
            return m.group(1)
    return None


def parse_binary(text: str) -> int | None:
    if text is None:
        return None
    norm = text.strip().lower()
    if norm in {"1", "yes", "y", "true", "possible", "plausible", "valid"}:
        return 1
    if norm in {"0", "no", "n", "false", "impossible", "implausible", "invalid"}:
        return 0
    # Prefer explicit terminal answers when the model includes reasoning.
    for pat, value in [
        (r"(?:final\s+answer|answer)\s*[:\-]?\s*(yes|1|possible|plausible)\b", 1),
        (r"(?:final\s+answer|answer)\s*[:\-]?\s*(no|0|impossible|implausible)\b", 0),
        (r"\b(yes|1|possible|plausible)\s*[.!]?$", 1),
        (r"\b(no|0|impossible|implausible)\s*[.!]?$", 0),
    ]:
        if re.search(pat, norm):
            return value
    return None


def nested_get(mapping: Any, candidate_keys: Iterable[str]) -> Any:
    """Depth-first search for the first key in candidate_keys inside nested dict/list data."""
    wanted = set(candidate_keys)
    if isinstance(mapping, dict):
        for key in candidate_keys:
            if key in mapping:
                return mapping[key]
        for value in mapping.values():
            found = nested_get(value, wanted)
            if found is not None:
                return found
    elif isinstance(mapping, (list, tuple)):
        for value in mapping:
            found = nested_get(value, wanted)
            if found is not None:
                return found
    return None
