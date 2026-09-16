from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .media import resolve_content
from .modeling import load_for_inference


def normalize_adapter_path(adapter: str | None) -> str | None:
    if not adapter:
        return None
    p = Path(adapter)
    if (p / "adapter_config.json").exists():
        return str(p)
    if (p / "adapter" / "adapter_config.json").exists():
        return str(p / "adapter")
    return str(p)


def load_runtime(
    model_id: str,
    adapter: str | None = None,
    dtype: str = "bf16",
    device_map: str | None = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str | None = None,
):
    return load_for_inference(
        model_id=model_id,
        adapter=normalize_adapter_path(adapter),
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )


def model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_messages(content: list[dict[str, Any]], system_prompt: str | None = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": content})
    return messages


@torch.inference_mode()
def generate_response(
    model,
    processor,
    content: list[dict[str, Any]],
    media_root: str | Path | None = None,
    system_prompt: str | None = None,
    num_frames: int = 16,
    max_new_tokens: int = 16,
) -> str:
    resolved = resolve_content(content, media_root=media_root, default_num_frames=num_frames)
    messages = build_messages(resolved, system_prompt)
    kwargs = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    try:
        inputs = processor.apply_chat_template(messages, num_frames=num_frames, **kwargs)
    except TypeError:
        inputs = processor.apply_chat_template(messages, **kwargs)
    device = model_input_device(model)
    inputs = inputs.to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    generated = output[:, prompt_len:]
    return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


@torch.inference_mode()
def extract_prompt_representation(
    model,
    processor,
    content: list[dict[str, Any]],
    media_root: str | Path | None = None,
    system_prompt: str | None = None,
    num_frames: int = 16,
) -> torch.Tensor:
    resolved = resolve_content(content, media_root=media_root, default_num_frames=num_frames)
    messages = build_messages(resolved, system_prompt)
    kwargs = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    try:
        inputs = processor.apply_chat_template(messages, num_frames=num_frames, **kwargs)
    except TypeError:
        inputs = processor.apply_chat_template(messages, **kwargs)
    device = model_input_device(model)
    inputs = inputs.to(device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        for attr in ("language_model_output", "model_output", "decoder_output"):
            child = getattr(outputs, attr, None)
            if child is not None and getattr(child, "hidden_states", None) is not None:
                hidden_states = child.hidden_states
                break
    if hidden_states is None:
        raise RuntimeError("Checkpoint does not expose hidden_states with output_hidden_states=True")
    last = hidden_states[-1]
    idx = inputs["attention_mask"].sum(dim=-1).long() - 1
    batch = torch.arange(last.size(0), device=last.device)
    return last[batch, idx].detach().float().cpu()
