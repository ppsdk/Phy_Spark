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


def load_adapter_for_training(base_model: nn.Module, checkpoint_dir: str | Path) -> nn.Module:
    checkpoint_dir = Path(checkpoint_dir)
    adapter_dir = checkpoint_dir / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"No PEFT adapter checkpoint found in {adapter_dir}")
    return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)


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


class PromptRepresentationPooler(nn.Module):
    """Pool only the causal prompt region, never assistant answer tokens."""

    def __init__(self, mode: str = "prompt_end") -> None:
        super().__init__()
        self.mode = mode

    def forward(
        self,
        last_hidden: torch.Tensor,
        anchor_index: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        anchor_index = anchor_index.to(last_hidden.device).long().clamp(0, last_hidden.size(1) - 1)
        if self.mode == "prompt_end":
            return _gather_anchor(last_hidden, anchor_index)
        if self.mode != "prompt_mean":
            raise ValueError(f"Unsupported pooling mode: {self.mode}")

        positions = torch.arange(last_hidden.size(1), device=last_hidden.device).unsqueeze(0)
        prompt_mask = positions <= anchor_index.unsqueeze(1)
        if attention_mask is not None:
            prompt_mask = prompt_mask & attention_mask.to(last_hidden.device).bool()
        weights = prompt_mask.unsqueeze(-1).to(last_hidden.dtype)
        return (last_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class DeltaEncoder(nn.Module):
    """Encode a signed state change and its time horizon in the VLM hidden space."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, h_t: torch.Tensor, h_tp: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.to(h_t.device).float().reshape(-1, 1)
        return self.network(torch.cat([h_tp - h_t, tau], dim=-1))


class GroupProjector(nn.Module):
    """Learn a compact, group-specific subspace shared by related physical heads."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.network(representation)


class PhysGroundModel(nn.Module):
    def __init__(self, base_model: nn.Module, grounding: GroundingConfig):
        super().__init__()
        self.base_model = base_model
        self.grounding = grounding
        ignored = set(getattr(base_model.config, "keys_to_ignore_at_inference", []) or [])
        base_model.config.keys_to_ignore_at_inference = sorted(ignored | {"loss_parts"})
        hidden_size = infer_hidden_size(base_model.config)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(grounding.hidden_dropout)
        self.pooler = PromptRepresentationPooler(grounding.pooling)
        active_groups = sorted({spec.group for spec in grounding.heads if spec.weight > 0})
        head_hidden_dim = grounding.head_hidden_dim or hidden_size
        self.group_projectors = nn.ModuleDict()
        if grounding.head_hidden_dim > 0:
            self.group_projectors.update(
                {
                    group: GroupProjector(hidden_size, head_hidden_dim, grounding.hidden_dropout)
                    for group in active_groups
                }
            )
        self.heads = nn.ModuleDict()
        for spec in grounding.heads:
            out_dim = spec.num_labels if spec.kind in {"classification", "regression"} else max(spec.num_labels, 1)
            self.heads[spec.name] = nn.Linear(head_hidden_dim, out_dim)
        needs_delta = any(spec.group == "delta" and spec.weight > 0 for spec in grounding.heads)
        self.delta_encoder = DeltaEncoder(hidden_size) if needs_delta or grounding.jepa_weight > 0 else None
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
        return self.pooler(last, anchor, mask)

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
        h = self.pooler(last, anchor_index, model_inputs.get("attention_mask"))

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
            if self.delta_encoder is not None:
                delta_repr = self.delta_encoder(h_t, h_tp, tau)

        grounding_targets = grounding_targets or {}
        grounding_masks = grounding_masks or {}
        specs = {x.name: x for x in self.grounding.heads}
        for name, spec in specs.items():
            if spec.weight <= 0:
                continue
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
            if spec.group in self.group_projectors:
                rep = self.group_projectors[spec.group](rep)
            else:
                rep = self.dropout(rep)
            logits = self.heads[name](rep)
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
        from safetensors.torch import save_file

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = output_dir / "adapter"
        self.base_model.save_pretrained(adapter_dir)
        aux: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("heads", self.heads),
            ("group_projectors", self.group_projectors),
            ("delta_encoder", self.delta_encoder),
            ("jepa_projector", self.jepa_projector),
        ):
            if module is not None:
                aux.update(
                    {
                        f"{prefix}.{name}": value.detach().cpu().contiguous()
                        for name, value in module.state_dict().items()
                    }
                )
        save_file(aux, output_dir / "physground_aux.safetensors")
        cfg = {
            "format_version": 2,
            "lm_weight": self.grounding.lm_weight,
            "jepa_weight": self.grounding.jepa_weight,
            "jepa_dim": self.grounding.jepa_dim,
            "jepa_source": self.grounding.jepa_source,
            "jepa_feature_layer": self.grounding.jepa_feature_layer,
            "hidden_dropout": self.grounding.hidden_dropout,
            "pooling": self.grounding.pooling,
            "head_hidden_dim": self.grounding.head_hidden_dim,
            "heads": [vars(x) for x in self.grounding.heads],
        }
        (output_dir / "physground_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def load_physground_aux(self, checkpoint_dir: str | Path) -> None:
        """Restore trainable auxiliary modules from a v2 or legacy v1 checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "physground_config.json"
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            saved_version = int(payload.pop("format_version", 1))
            if saved_version not in {1, 2}:
                raise ValueError(f"Unsupported PhysGround checkpoint format version: {saved_version}")
            saved_grounding = GroundingConfig.from_dict(payload)
            if saved_grounding != self.grounding:
                raise ValueError(
                    "Checkpoint grounding configuration does not match the current run. "
                    "Resume training requires identical heads, weights, pooling, dropout, and JEPA settings."
                )
        safe_path = checkpoint_dir / "physground_aux.safetensors"
        legacy_path = checkpoint_dir / "physground_heads.pt"
        if safe_path.exists():
            from safetensors.torch import load_file

            tensors = load_file(safe_path, device="cpu")
            modules: tuple[tuple[str, nn.Module | None], ...] = (
                ("heads", self.heads),
                ("group_projectors", self.group_projectors),
                ("delta_encoder", self.delta_encoder),
                ("jepa_projector", self.jepa_projector),
            )
            for prefix, module in modules:
                state = {
                    name.removeprefix(f"{prefix}."): value
                    for name, value in tensors.items()
                    if name.startswith(f"{prefix}.")
                }
                if module is None:
                    if state:
                        raise ValueError(f"Checkpoint contains disabled module {prefix!r}")
                    continue
                module.load_state_dict(state, strict=True)
            return
        if legacy_path.exists():
            if self.grounding.head_hidden_dim > 0:
                raise ValueError("Legacy v1 checkpoints require head_hidden_dim=0")
            legacy = torch.load(legacy_path, map_location="cpu", weights_only=True)
            self.heads.load_state_dict(legacy["heads"], strict=True)
            if self.delta_encoder is not None:
                self.delta_encoder.network.load_state_dict(legacy["delta_fuse"], strict=True)
            if self.jepa_projector is not None:
                state = legacy.get("jepa_projector")
                if state is None:
                    raise ValueError("Legacy checkpoint is missing the enabled JEPA projector")
                self.jepa_projector.load_state_dict(state, strict=True)
            return
        raise FileNotFoundError(f"No PhysGround auxiliary checkpoint found in {checkpoint_dir}")

    @classmethod
    def from_checkpoint(
        cls,
        base_model: nn.Module,
        checkpoint_dir: str | Path,
    ) -> PhysGroundModel:
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "physground_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing PhysGround config: {config_path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        saved_version = int(payload.pop("format_version", 1))
        if saved_version not in {1, 2}:
            raise ValueError(f"Unsupported PhysGround checkpoint format version: {saved_version}")
        model = cls(base_model, GroundingConfig.from_dict(payload))
        model.load_physground_aux(checkpoint_dir)
        return model
