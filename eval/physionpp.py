from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from physground.inference import generate_response, load_runtime
from physground.utils import parse_binary, read_jsonl, write_jsonl


PROMPT = (
    "You are shown only the observable prefix of a Physion++ trial. Based on the motion and interaction history, "
    "predict whether the red target object will contact the yellow target object after the prediction point. "
    "Answer only Yes or No."
)


def content(row: dict) -> list[dict]:
    return [
        {
            "type": "video",
            "path": row["video"],
            "end_frame": int(row["start_frame_for_prediction"]),
        },
        {"type": "text", "text": PROMPT},
    ]


def summarize(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r.get("label") in {0, 1}]
    out = {"n": len(rows), "n_labeled": len(labeled)}
    if labeled:
        out["accuracy"] = sum(r["prediction"] == r["label"] for r in labeled) / len(labeled)
    groups = defaultdict(list)
    for r in labeled:
        groups[str(r.get("property") or "unknown")].append(r)
    if groups:
        out["by_property"] = {
            k: {"n": len(v), "accuracy": sum(x["prediction"] == x["label"] for x in v) / len(v)}
            for k, v in sorted(groups.items())
        }
    return out


def main():
    p = argparse.ArgumentParser(description="Direct VLM QA diagnostic on Physion++")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--output", default="results/physionpp_qa.jsonl")
    args = p.parse_args()

    rows_in = read_jsonl(args.manifest)
    if args.limit:
        rows_in = rows_in[: args.limit]
    model, processor = load_runtime(args.model, args.adapter, dtype=args.dtype)

    rows = []
    for row in tqdm(rows_in, desc="Physion++ QA"):
        raw = generate_response(
            model,
            processor,
            content(row),
            media_root=args.data_root,
            system_prompt="Make a future contact prediction from the visible prefix only.",
            num_frames=args.num_frames,
            max_new_tokens=8,
        )
        pred = parse_binary(raw)
        rows.append({**row, "prediction": pred, "raw_response": raw})

    write_jsonl(args.output, rows)
    summary = summarize(rows)
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
