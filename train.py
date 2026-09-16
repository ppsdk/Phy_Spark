from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from physground.data import CanonicalPhysicalDataset, GroundingCollator
from physground.modeling import (
    PhysGroundModel,
    attach_lora,
    load_adapter_for_training,
    load_base_model,
    load_processor,
)
from physground.trainer import PhysGroundTrainer
from physground.utils import load_yaml, set_seed
from physground.validation import validate_manifest_rows, validate_training_config


def parse_args():
    p = argparse.ArgumentParser(description="PhysGround-Tune training")
    p.add_argument("--config", required=True, help="YAML configuration")
    return p.parse_args()


def resolve_resume_checkpoint(value, output_dir: str) -> str | None:
    if not value:
        return None
    if value is True:
        checkpoint = get_last_checkpoint(output_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint-* directory found below {output_dir}")
        return checkpoint
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return str(path)


def build_training_arguments(
    train_cfg: dict,
    output_dir: str,
    seed: int,
    has_eval: bool,
    has_aux_objectives: bool,
) -> TrainingArguments:
    kwargs = {
        "output_dir": output_dir,
        "learning_rate": float(train_cfg.get("learning_rate", 2e-5)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1.0)),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 8)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "save_steps": int(train_cfg.get("save_steps", 500)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "bf16": bool(train_cfg.get("bf16", True)) and torch.cuda.is_available(),
        "fp16": bool(train_cfg.get("fp16", False)) and torch.cuda.is_available(),
        "remove_unused_columns": False,
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "report_to": train_cfg.get("report_to", "none"),
        "optim": str(train_cfg.get("optim", "adamw_torch_fused")),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "seed": seed,
        "ddp_find_unused_parameters": bool(
            train_cfg.get("ddp_find_unused_parameters", has_aux_objectives)
        ),
    }
    eval_strategy = str(train_cfg.get("eval_strategy", "steps" if has_eval else "no"))
    if eval_strategy != "no" and not has_eval:
        raise ValueError("training.eval_strategy requires data.eval_jsonl")
    signature = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    kwargs[strategy_key] = eval_strategy
    if eval_strategy != "no":
        kwargs["eval_steps"] = int(train_cfg.get("eval_steps", train_cfg.get("logging_steps", 10)))
    return TrainingArguments(**kwargs)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    grounding_cfg = validate_training_config(cfg)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg.get("training", {})
    output_dir = str(train_cfg.get("output_dir", "outputs/physground"))
    resume_checkpoint = resolve_resume_checkpoint(train_cfg.get("resume_from_checkpoint"), output_dir)

    train_dataset = CanonicalPhysicalDataset(data_cfg["train_jsonl"])
    manifest_errors = validate_manifest_rows(
        train_dataset.rows,
        grounding=grounding_cfg,
        media_root=data_cfg.get("media_root"),
    )
    if manifest_errors:
        preview = "\n".join(f"- {error}" for error in manifest_errors[:20])
        raise ValueError(f"Invalid training manifest ({len(manifest_errors)} errors):\n{preview}")
    eval_dataset = None
    if data_cfg.get("eval_jsonl"):
        eval_dataset = CanonicalPhysicalDataset(data_cfg["eval_jsonl"])
        eval_errors = validate_manifest_rows(
            eval_dataset.rows,
            grounding=grounding_cfg,
            media_root=data_cfg.get("media_root"),
        )
        if eval_errors:
            preview = "\n".join(f"- {error}" for error in eval_errors[:20])
            raise ValueError(f"Invalid evaluation manifest ({len(eval_errors)} errors):\n{preview}")

    model_id = model_cfg["id"]
    processor = load_processor(model_id, trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
    base = load_base_model(
        model_id,
        dtype=str(model_cfg.get("dtype", "bf16")),
        device_map=None,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        attn_implementation=model_cfg.get("attn_implementation"),
    )
    if bool(train_cfg.get("gradient_checkpointing", True)):
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()

    if resume_checkpoint:
        base = load_adapter_for_training(base, resume_checkpoint)
    else:
        base = attach_lora(base, cfg.get("lora", {}))
    if hasattr(base, "print_trainable_parameters"):
        base.print_trainable_parameters()
    model = PhysGroundModel(base, grounding_cfg)
    if resume_checkpoint:
        model.load_physground_aux(resume_checkpoint)

    collator = GroundingCollator(
        processor=processor,
        head_specs=grounding_cfg.heads,
        media_root=data_cfg.get("media_root"),
        num_frames=int(data_cfg.get("num_frames", 16)),
        system_prompt=str(data_cfg.get("system_prompt", "You are a vision-language model reasoning about physical scenes.")),
        jepa_enabled=grounding_cfg.jepa_weight > 0,
    )

    training_args = build_training_arguments(
        train_cfg,
        output_dir,
        seed,
        eval_dataset is not None,
        has_aux_objectives=bool(grounding_cfg.heads or grounding_cfg.jepa_weight > 0),
    )

    trainer = PhysGroundTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir)
    print(f"Saved adapter + physical heads to: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
