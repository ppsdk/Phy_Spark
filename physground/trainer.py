from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from transformers import Trainer


class PhysGroundTrainer(Trainer):
    """Trainer that checkpoints only LoRA adapters + lightweight physical heads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._component_loss_sums: dict[str, float] = defaultdict(float)
        self._component_loss_counts: dict[str, int] = defaultdict(int)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs["loss"]
        if model.training:
            for name, value in outputs.get("loss_parts", {}).items():
                self._component_loss_sums[name] += float(value.detach().float().cpu())
                self._component_loss_counts[name] += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._component_loss_counts:
            for name, total in self._component_loss_sums.items():
                logs[f"loss/{name}"] = total / self._component_loss_counts[name]
            self._component_loss_sums.clear()
            self._component_loss_counts.clear()
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            # Transformers <5 does not accept start_time.
            super().log(logs)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None, **kwargs):
        target = model or self.model
        target.load_physground_aux(resume_from_checkpoint)

    def _save(self, output_dir: str | None = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_physground(path)
        processing = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processing is not None:
            processing.save_pretrained(path / "processor")
        (path / "training_args.json").write_text(
            json.dumps(self.args.to_dict(), indent=2, default=str), encoding="utf-8"
        )
