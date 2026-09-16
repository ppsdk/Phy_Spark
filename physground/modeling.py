from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoProcessor

from .config import GroundingConfig, HeadSpec


def _auto_model_class():
    # Transformers 5.x names the unified class AutoModelForMultimodalLM; older
    # releases expose AutoModelForImageTextToText. Keep both paths working.
    try:
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM
    except ImportError:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText


def _dtype(value: str | None):
    if value is None or value == "auto":
        return "auto"
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in table:
        raise ValueError(f"Unsupported dtype: {value}")
    return table[value]


def load_processor(model_id: str, trust_remote_code: bool = True):
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)


def load_base_model(
    model_id: str,
    dtype: str = "bf16",
    device_map: str | None = None,
    trust_remote_code: bool = True,
    attn_implementation: str | None = None,
):
    cls = _auto_model_class()
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": _dtype(dtype),
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        # Some remote-code checkpoints don't accept attn_implementation.
        kwargs.pop("attn_implementation", None)
        return cls.from_pretrained(model_id, **kwargs)


def attach_lora(base_model: nn.Module, lora_cfg: dict[str, Any]) -> nn.Module:
    target_modules = lora_cfg.get("target_modules", "all-linear")
    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=target_modules,
        task_type=lora_cfg.get("task_type", None),
    )
    return get_peft_model(base_model, config)


def load_for_inference(
    model_id: str,
    adapter: str | None = None,
    dtype: str = "bf16",
    device_map: str | None = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str | None = None,
):
    model = load_base_model(
        model_id,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    processor = load_processor(model_id, trust_remote_code=trust_remote_code)
    return model, processor


def infer_hidden_size(config: Any) -> int:
    candidates = [
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(getattr(config, "llm_config", None), "hidden_size", None),
        getattr(getattr(config, "language_config", None), "hidden_size", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    for attr in ("text_config", "llm_config", "language_config"):
        sub = getattr(config, attr, None)
        if sub is not None:
            try:
                return infer_hidden_size(sub)
            except ValueError:
                pass
    raise ValueError("Could not infer language hidden_size from model config")


def _extract_hidden_states(outputs: Any):
    hs = getattr(outputs, "hidden_states", None)
    if hs is not None:
        return hs
    for attr in ("language_model_output", "model_output", "decoder_output"):
        child = getattr(outputs, attr, None)
        if child is not None:
            hs = getattr(child, "hidden_states", None)
            if hs is not None:
                return hs
    if isinstance(outputs, dict):
        if outputs.get("hidden_states") is not None:
            return outputs["hidden_states"]
        for value in outputs.values():
            if isinstance(value, dict) and value.get("hidden_states") is not None:
                return value["hidden_states"]
    raise RuntimeError(
        "Model did not return language hidden states. Use a HF-format checkpoint that supports "
        "output_hidden_states=True (Qwen3-VL and InternVL3.x-HF do)."
    )


def _gather_anchor(last_hidden: torch.Tensor, anchor_index: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, L, H]
    anchor_index = anchor_index.to(last_hidden.device).long().clamp_min(0)
    batch = torch.arange(last_hidden.size(0), device=last_hidden.device)
    return last_hidden[batch, anchor_index]


class PhysGroundModel(nn.Module):
    def __init__(self, base_model: nn.Module, grounding: GroundingConfig):
        super().__init__()
        self.base_model = base_model
        self.grounding = grounding
        hidden_size = infer_hidden_size(base_model.config)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(grounding.hidden_dropout)
        self.heads = nn.ModuleDict()
        for spec in grounding.heads:
            out_dim = spec.num_labels if spec.kind in {"classification", "regression"} else max(spec.num_labels, 1)
            self.heads[spec.name] = nn.Linear(hidden_size, out_dim)
        self.delta_fuse = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.jepa_projector = None
        if grounding.jepa_weight > 0:
            if grounding.jepa_dim <= 0:
                raise ValueError("grounding.jepa_dim must be >0 when jepa_weight>0")
            self.jepa_projector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, grounding.jepa_dim),
            )

    @property
    def config(self):
        return self.base_model.config

    def _encode_aux(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.base_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        last = _extract_hidden_states(outputs)[-1]
        mask = inputs.get("attention_mask")
        if mask is None:
            anchor = torch.full((last.size(0),), last.size(1) - 1, device=last.device, dtype=torch.long)
        else:
            anchor = mask.to(last.device).sum(dim=-1).long() - 1
        return _gather_anchor(last, anchor)

    def _head_loss(
        self,
        spec: HeadSpec,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.to(logits.device).float().view(-1)
        if mask.sum() <= 0:
            return logits.sum() * 0.0
        if spec.kind == "classification":
            target = target.to(logits.device).long().view(-1)
            raw = F.cross_entropy(logits, target, reduction="none")
        elif spec.kind == "bce":
            target = target.to(logits.device).float().view_as(logits)
            raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            raw = raw.view(raw.size(0), -1).mean(dim=-1)
        else:
            target = target.to(logits.device).float().view_as(logits)
            raw = F.mse_loss(logits, target, reduction="none")
            raw = raw.view(raw.size(0), -1).mean(dim=-1)
        return (raw * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(
        self,
        anchor_index: torch.Tensor | None = None,
        grounding_targets: dict[str, torch.Tensor] | None = None,
        grounding_masks: dict[str, torch.Tensor] | None = None,
        delta_t_inputs: dict[str, torch.Tensor] | None = None,
        delta_tp_inputs: dict[str, torch.Tensor] | None = None,
        delta_tau: torch.Tensor | None = None,
        teacher_delta: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **model_inputs,
    ) -> dict[str, Any]:
        base_outputs = self.base_model(
            **model_inputs,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        last = _extract_hidden_states(base_outputs)[-1]
        if anchor_index is None:
            mask = model_inputs.get("attention_mask")
            anchor_index = mask.sum(dim=-1).long() - 1 if mask is not None else torch.full(
                (last.size(0),), last.size(1) - 1, device=last.device, dtype=torch.long
            )
        h = self.dropout(_gather_anchor(last, anchor_index))

        total_loss = h.sum() * 0.0
        loss_parts: dict[str, torch.Tensor] = {}
        lm_loss = getattr(base_outputs, "loss", None)
        if lm_loss is not None and torch.isfinite(lm_loss):
            total_loss = total_loss + self.grounding.lm_weight * lm_loss
            loss_parts["lm"] = lm_loss.detach()

        delta_repr = None
        if delta_t_inputs is not None and delta_tp_inputs is not None:
            h_t = self._encode_aux(delta_t_inputs)
            h_tp = self._encode_aux(delta_tp_inputs)
            tau = delta_tau
            if tau is None:
                tau = torch.zeros(h_t.size(0), device=h_t.device)
            tau = tau.to(h_t.device).float().reshape(-1, 1)
            delta_repr = self.delta_fuse(torch.cat([h_tp - h_t, tau], dim=-1))

        grounding_targets = grounding_targets or {}
        grounding_masks = grounding_masks or {}
        specs = {x.name: x for x in self.grounding.heads}
        for name, spec in specs.items():
            target = grounding_targets.get(name)
            mask = grounding_masks.get(name)
            if target is None or mask is None:
                continue
            if spec.group == "delta":
                if delta_repr is None:
                    continue
                rep = delta_repr
            else:
                rep = h
            logits = self.heads[name](self.dropout(rep))
            loss = self._head_loss(spec, logits, target, mask)
            total_loss = total_loss + spec.weight * loss
            loss_parts[name] = loss.detach()

        if teacher_delta is not None and self.jepa_projector is not None and delta_repr is not None:
            pred = self.jepa_projector(delta_repr)
            target = teacher_delta.to(pred.device).float().view_as(pred).detach()
            valid = target.norm(dim=-1) > 1e-8
            if valid.any():
                jepa = 1.0 - F.cosine_similarity(pred[valid], target[valid], dim=-1).mean()
                total_loss = total_loss + self.grounding.jepa_weight * jepa
                loss_parts["jepa"] = jepa.detach()

        return {
            "loss": total_loss,
            "logits": getattr(base_outputs, "logits", None),
            "loss_parts": loss_parts,
        }

    def save_physground(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = output_dir / "adapter"
        self.base_model.save_pretrained(adapter_dir)
        aux = {
            "heads": self.heads.state_dict(),
            "delta_fuse": self.delta_fuse.state_dict(),
            "jepa_projector": self.jepa_projector.state_dict() if self.jepa_projector is not None else None,
        }
        torch.save(aux, output_dir / "physground_heads.pt")
        cfg = {
            "lm_weight": self.grounding.lm_weight,
            "jepa_weight": self.grounding.jepa_weight,
            "jepa_dim": self.grounding.jepa_dim,
            "hidden_dropout": self.grounding.hidden_dropout,
            "heads": [vars(x) for x in self.grounding.heads],
        }
        (output_dir / "physground_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
