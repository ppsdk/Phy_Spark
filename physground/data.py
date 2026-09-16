from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .alignment import answer_boundary
from .config import HeadSpec
from .media import resolve_content
from .utils import read_jsonl


class CanonicalPhysicalDataset(Dataset):
    """JSONL dataset used by all training objectives.

    The canonical format is intentionally independent of CLEVRER / Physion++ / PhysInOne
    field names. Dataset-specific converters should only emit labels that are directly
    available or reliably derived from released annotations.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = read_jsonl(path)
        if not self.rows:
            raise ValueError(f"No samples found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class GroundingCollator:
    """Multimodal collator for SFT + representation grounding.

    Per-device batch size 1 is deliberate in v0.1: multimodal processors use
    architecture-specific packing for variable numbers of images/video patches.
    Effective batch size should be increased with gradient accumulation. This keeps
    Qwen3-VL and InternVL HF checkpoints on one shared, reliable code path.
    """

    def __init__(
        self,
        processor: Any,
        head_specs: list[HeadSpec],
        media_root: str | Path | None = None,
        num_frames: int | None = 16,
        system_prompt: str = "You are a vision-language model reasoning about physical scenes.",
        jepa_enabled: bool = False,
    ) -> None:
        self.processor = processor
        self.head_specs = {h.name: h for h in head_specs}
        self.delta_head_names = {
            h.name for h in head_specs if h.group == "delta" and h.weight > 0
        }
        self.media_root = media_root
        self.num_frames = num_frames
        self.system_prompt = system_prompt
        self.jepa_enabled = bool(jepa_enabled)

    def _content(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        if "content" in sample:
            content = sample["content"]
        else:
            content = [dict(x) for x in sample.get("media", [])]
            content.append({"type": "text", "text": sample.get("question", "Describe the physical state.")})
        return resolve_content(content, self.media_root, self.num_frames)

    def _messages(self, content: list[dict[str, Any]], answer: str | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": content})
        if answer is not None and answer != "":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": str(answer)}]})
        return messages

    def _encode(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> dict[str, torch.Tensor]:
        kwargs = dict(
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            encoded = self.processor.apply_chat_template(messages, num_frames=self.num_frames, **kwargs)
        except TypeError:
            encoded = self.processor.apply_chat_template(messages, **kwargs)
        return dict(encoded)

    def _load_teacher_delta(self, sample: dict[str, Any]) -> torch.Tensor | None:
        if sample.get("teacher_delta") is not None:
            return torch.tensor(sample["teacher_delta"], dtype=torch.float32).unsqueeze(0)
        if "teacher_delta_path" in sample:
            path = Path(sample["teacher_delta_path"])
            if not path.is_absolute() and self.media_root is not None:
                path = Path(self.media_root) / path
            arr = np.load(path)
            return torch.tensor(arr, dtype=torch.float32).reshape(1, -1)
        return None

    def _encode_delta_side(self, side: dict[str, Any]) -> dict[str, torch.Tensor]:
        content = side.get("content")
        if content is None:
            media = side.get("media", [])
            query = side.get("query", "Represent the physical state in this observation.")
            content = [dict(x) for x in media] + [{"type": "text", "text": query}]
        resolved = resolve_content(content, self.media_root, self.num_frames)
        messages = self._messages(resolved, answer=None)
        return self._encode(messages, add_generation_prompt=True)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError(
                "PhysGround v0.1 uses per_device_train_batch_size=1 for portable video packing. "
                "Use gradient_accumulation_steps to increase effective batch size."
            )
        sample = features[0]
        content = self._content(sample)
        prompt_messages = self._messages(content, answer=None)
        prompt_inputs = self._encode(prompt_messages, add_generation_prompt=True)
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])

        answer = sample.get("answer")
        if answer is not None and str(answer) != "":
            full_messages = self._messages(content, answer=str(answer))
            model_inputs = self._encode(full_messages, add_generation_prompt=False)
            boundary = answer_boundary(
                prompt_inputs["input_ids"][0].tolist(),
                model_inputs["input_ids"][0].tolist(),
            )
            labels = model_inputs["input_ids"].clone()
            labels[:, :boundary] = -100
            model_inputs["labels"] = labels
            anchor_index = boundary - 1
        else:
            model_inputs = prompt_inputs
            anchor_index = prompt_len - 1

        model_inputs["anchor_index"] = torch.tensor([anchor_index], dtype=torch.long)

        target_values = sample.get("targets", {}) or {}
        target_masks = sample.get("target_masks", {}) or {}
        grounding_targets: dict[str, torch.Tensor] = {}
        grounding_masks: dict[str, torch.Tensor] = {}
        for name, spec in self.head_specs.items():
            if name not in target_values:
                grounding_masks[name] = torch.zeros(1, dtype=torch.float32)
                continue
            value = target_values[name]
            if spec.kind == "classification":
                tensor = torch.tensor([int(value)], dtype=torch.long)
            elif spec.num_labels > 1:
                tensor = torch.tensor(value, dtype=torch.float32).reshape(1, spec.num_labels)
            else:
                tensor = torch.tensor([float(value)], dtype=torch.float32)
            grounding_targets[name] = tensor
            grounding_masks[name] = torch.tensor(
                [float(target_masks.get(name, 1.0))], dtype=torch.float32
            )
        model_inputs["grounding_targets"] = grounding_targets
        model_inputs["grounding_masks"] = grounding_masks

        teacher = self._load_teacher_delta(sample) if self.jepa_enabled else None
        active_delta_target = any(
            name in target_values
            and float(target_masks.get(name, 1.0)) > 0
            for name in self.delta_head_names
        )
        needs_delta = active_delta_target or (self.jepa_enabled and teacher is not None)
        delta = sample.get("delta")
        if needs_delta:
            if not delta:
                raise ValueError("Active delta supervision requires delta observations")
            if "t" not in delta or "tp" not in delta:
                raise ValueError("delta must contain both 't' and 'tp' observations")
            model_inputs["delta_t_inputs"] = self._encode_delta_side(delta["t"])
            model_inputs["delta_tp_inputs"] = self._encode_delta_side(delta["tp"])
            model_inputs["delta_tau"] = torch.tensor([float(delta.get("tau", 0.0))], dtype=torch.float32)

        if self.jepa_enabled and teacher is not None:
            model_inputs["teacher_delta"] = teacher

        return model_inputs
