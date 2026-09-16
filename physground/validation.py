from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import GroundingConfig, HeadSpec
from .utils import load_yaml, read_jsonl


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _content_errors(
    content: Any,
    location: str,
    media_root: Path | None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(content, list) or not content:
        return [f"{location}: content/media must be a non-empty list"]
    for index, item in enumerate(content):
        item_location = f"{location}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_location}: content item must be an object")
            continue
        typ = item.get("type")
        if typ not in {"text", "image", "video"}:
            errors.append(f"{item_location}: invalid content type {typ!r}")
            continue
        if typ == "text":
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                errors.append(f"{item_location}: text item must contain non-empty text")
            continue
        source = item.get("path", item.get("url"))
        if source is None:
            errors.append(f"{item_location}: {typ} item requires path or url")
            continue
        sources = source if isinstance(source, list) else [source]
        if not sources or not all(isinstance(value, str) and value.strip() for value in sources):
            errors.append(f"{item_location}: path/url must be a string or non-empty string list")
            continue
        if media_root is not None:
            for value in sources:
                if "://" in value or value.startswith("data:"):
                    continue
                path = Path(value)
                candidate = path if path.is_absolute() else media_root / path
                if not candidate.exists():
                    errors.append(f"{item_location}: missing media {candidate}")
        for key in ("start_frame", "end_frame", "num_frames"):
            if key in item and (not isinstance(item[key], int) or isinstance(item[key], bool) or item[key] < 0):
                errors.append(f"{item_location}: {key} must be a non-negative integer")
        if item.get("num_frames") == 0:
            errors.append(f"{item_location}: num_frames must be greater than zero")
        start = item.get("start_frame", 0)
        end = item.get("end_frame")
        if isinstance(start, int) and isinstance(end, int) and end <= start:
            errors.append(f"{item_location}: end_frame must be greater than start_frame")
    return errors


def _target_errors(
    targets: Any,
    masks: Any,
    specs: dict[str, HeadSpec] | None,
    location: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(targets, dict):
        return [f"{location}: targets must be an object"]
    if not isinstance(masks, dict):
        return [f"{location}: target_masks must be an object"]
    for name in masks:
        if name not in targets:
            errors.append(f"{location}: target_masks contains {name!r} without a matching target")
    for name, value in targets.items():
        if specs is not None and name not in specs:
            errors.append(f"{location}: target {name!r} has no matching head in the config")
            continue
        mask = masks.get(name, 1.0)
        if not _is_number(mask) or not 0 <= float(mask) <= 1:
            errors.append(f"{location}: mask for {name!r} must be finite and in [0, 1]")
        if specs is None:
            continue
        spec = specs[name]
        if spec.kind == "classification":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{location}: classification target {name!r} must be an integer")
            elif not 0 <= value < spec.num_labels:
                errors.append(
                    f"{location}: classification target {name!r}={value} is outside "
                    f"[0, {spec.num_labels})"
                )
        elif spec.num_labels == 1:
            if not _is_number(value):
                errors.append(f"{location}: scalar target {name!r} must be a finite number")
        elif not isinstance(value, list) or len(value) != spec.num_labels or not all(
            _is_number(element) for element in value
        ):
            errors.append(
                f"{location}: target {name!r} must contain {spec.num_labels} finite numbers"
            )
    return errors


def validate_manifest_rows(
    rows: Iterable[Any],
    grounding: GroundingConfig | None = None,
    media_root: str | Path | None = None,
) -> list[str]:
    root = Path(media_root).resolve() if media_root is not None else None
    specs = {head.name: head for head in grounding.heads} if grounding is not None else None
    errors: list[str] = []
    row_count = 0
    for row_index, row in enumerate(rows):
        row_count += 1
        location = f"row {row_index}"
        if not isinstance(row, dict):
            errors.append(f"{location}: sample must be an object")
            continue
        content = row.get("content", row.get("media"))
        errors.extend(_content_errors(content, f"{location}.content", root))
        targets = row.get("targets", {})
        masks = row.get("target_masks", {})
        errors.extend(_target_errors(targets, masks, specs, location))
        target_values = targets if isinstance(targets, dict) else {}
        target_masks = masks if isinstance(masks, dict) else {}

        delta = row.get("delta")
        if delta is not None:
            if not isinstance(delta, dict):
                errors.append(f"{location}: delta must be an object")
            else:
                for side in ("t", "tp"):
                    if not isinstance(delta.get(side), dict):
                        errors.append(f"{location}: delta must contain object {side!r}")
                        continue
                    side_content = delta[side].get("content", delta[side].get("media"))
                    errors.extend(_content_errors(side_content, f"{location}.delta.{side}", root))
                tau = delta.get("tau", 0.0)
                if not _is_number(tau) or tau < 0:
                    errors.append(f"{location}: delta.tau must be a finite non-negative number")

        teacher = row.get("teacher_delta")
        teacher_path = row.get("teacher_delta_path")
        if "teacher_delta_path" in row and teacher_path is None:
            errors.append(f"{location}: teacher_delta_path cannot be null")
        if teacher is not None and teacher_path is not None:
            errors.append(f"{location}: use only one of teacher_delta and teacher_delta_path")
        if (teacher is not None or teacher_path is not None) and delta is None:
            errors.append(f"{location}: teacher delta requires delta.t and delta.tp observations")
        if (teacher is not None or teacher_path is not None) and grounding is not None and grounding.jepa_weight <= 0:
            errors.append(f"{location}: teacher delta is ignored because jepa_weight is zero")
        if teacher is not None:
            if not isinstance(teacher, list) or not teacher or not all(_is_number(x) for x in teacher):
                errors.append(f"{location}: teacher_delta must be a non-empty finite-number list")
            elif grounding is not None and grounding.jepa_dim > 0 and len(teacher) != grounding.jepa_dim:
                errors.append(
                    f"{location}: teacher_delta length {len(teacher)} does not match jepa_dim "
                    f"{grounding.jepa_dim}"
                )
        if teacher_path is not None:
            if not isinstance(teacher_path, str) or not teacher_path.strip():
                errors.append(f"{location}: teacher_delta_path must be a non-empty string")
            elif root is not None:
                path = Path(teacher_path)
                candidate = path if path.is_absolute() else root / path
                if not candidate.exists():
                    errors.append(f"{location}: missing teacher delta {candidate}")

        active_target_names = {
            name
            for name in target_values
            if _is_number(target_masks.get(name, 1.0)) and float(target_masks.get(name, 1.0)) > 0
        }
        if specs is not None:
            for name in active_target_names & specs.keys():
                if specs[name].group == "delta" and delta is None:
                    errors.append(f"{location}: active delta target {name!r} requires delta.t and delta.tp")
            active_target_names = {
                name
                for name in active_target_names
                if name in specs
                and specs[name].weight > 0
                and (specs[name].group != "delta" or isinstance(delta, dict))
            }
        has_answer = (
            isinstance(row.get("answer"), str)
            and bool(row["answer"].strip())
            and (grounding is None or grounding.lm_weight > 0)
        )
        has_target = bool(active_target_names)
        has_teacher = (
            (teacher is not None or teacher_path is not None)
            and isinstance(delta, dict)
            and (grounding is None or grounding.jepa_weight > 0)
        )
        if not has_answer and not has_target and not has_teacher:
            errors.append(f"{location}: sample has no active supervision")
    if row_count == 0:
        errors.append("manifest contains no samples")
    return errors


def validate_training_config(config: dict[str, Any]) -> GroundingConfig:
    for section in ("model", "data"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"config section {section!r} must be a mapping")
    if not config["model"].get("id"):
        raise ValueError("config model.id is required")
    if not config["data"].get("train_jsonl"):
        raise ValueError("config data.train_jsonl is required")
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("config section 'training' must be a mapping")
    if int(training.get("gradient_accumulation_steps", 1)) < 1:
        raise ValueError("gradient_accumulation_steps must be >=1")
    if float(training.get("learning_rate", 2e-5)) <= 0:
        raise ValueError("learning_rate must be >0")
    num_frames = config["data"].get("num_frames", 16)
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames < 1:
        raise ValueError("data.num_frames must be a positive integer")
    return GroundingConfig.from_dict(config.get("grounding"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a canonical PhysGround JSONL manifest")
    parser.add_argument("manifest")
    parser.add_argument("--config", help="Optional training YAML used to validate target names and shapes")
    parser.add_argument("--media-root", default=None, help="If set, verify local media paths below this root")
    args = parser.parse_args()

    grounding = None
    if args.config:
        grounding = validate_training_config(load_yaml(args.config))
    rows = read_jsonl(args.manifest)
    errors = validate_manifest_rows(rows, grounding=grounding, media_root=args.media_root)
    print(f"rows={len(rows)} errors={len(errors)}")
    for error in errors[:100]:
        print(f"- {error}")
    if len(errors) > 100:
        print(f"- ... {len(errors) - 100} more errors")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
