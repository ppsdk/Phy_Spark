from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import Trainer


class PhysGroundTrainer(Trainer):
    """Trainer that checkpoints only LoRA adapters + lightweight physical heads."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

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
