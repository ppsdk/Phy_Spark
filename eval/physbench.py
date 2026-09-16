from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from physground.inference import generate_response, load_runtime
from physground.utils import parse_choice, write_jsonl


def build_content(item: dict, data_root: Path) -> list[dict]:
    question = item["question"]
    files = list(item.get("file_name", []))
    chunks = re.split(r"(<video>|<image>)", question)
    content: list[dict] = []
    media_idx = 0
    for chunk in chunks:
        if chunk not in {"<video>", "<image>"}:
            if chunk:
                content.append({"type": "text", "text": chunk})
            continue
        if media_idx >= len(files):
            raise ValueError(f"Not enough file_name entries for item idx={item.get('idx')}")
        fname = files[media_idx]
        media_idx += 1
        typ = "video" if chunk == "<video>" else "image"
        folder = "video" if typ == "video" else "image"
        content.append({"type": typ, "path": str(data_root / folder / fname)})
    if media_idx != len(files):
        # Some records can carry media not explicitly referenced in text; preserve order up front.
        extras = []
        for fname in files[media_idx:]:
            suffix = Path(fname).suffix.lower()
            typ = "video" if suffix in {".mp4", ".avi", ".mov", ".webm"} else "image"
            folder = "video" if typ == "video" else "image"
            extras.append({"type": typ, "path": str(data_root / folder / fname)})
        content = extras + content
    return content


def summarize(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r.get("answer") is not None]
    summary = {"n": len(rows), "n_labeled": len(labeled)}
    if labeled:
        summary["accuracy"] = sum(bool(r["correct"]) for r in labeled) / len(labeled)
    for key in ("task_type", "sub_type", "ability_type", "source", "split"):
        groups = defaultdict(list)
        for r in labeled:
            if r.get(key) is not None:
                groups[str(r[key])].append(r)
        if groups:
            summary[f"by_{key}"] = {
                name: {"n": len(vals), "accuracy": sum(bool(x["correct"]) for x in vals) / len(vals)}
                for name, vals in sorted(groups.items())
            }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--data-root", required=True, help="PhysBench root containing test.json, image/, video/")
    p.add_argument("--json", default=None, help="Defaults to <data-root>/test.json")
    p.add_argument("--split", default=None, help="Optional filter using record['split']")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--output", default="results/physbench_predictions.jsonl")
    args = p.parse_args()

    root = Path(args.data_root)
    items = json.loads(Path(args.json or root / "test.json").read_text(encoding="utf-8"))
    if args.split:
        items = [x for x in items if str(x.get("split", "")).lower() == args.split.lower()]
    if args.limit:
        items = items[: args.limit]

    model, processor = load_runtime(args.model, args.adapter, dtype=args.dtype)
    rows = []
    system = "Answer the multiple-choice physics question. End with exactly one option letter: A, B, C, or D."
    for item in tqdm(items, desc="PhysBench"):
        raw = generate_response(
            model,
            processor,
            build_content(item, root),
            system_prompt=system,
            num_frames=args.num_frames,
            max_new_tokens=16,
        )
        pred = parse_choice(raw)
        answer = item.get("answer")
        row = {
            "idx": item.get("idx"),
            "prediction": pred,
            "raw_response": raw,
            "answer": answer,
            "correct": (pred == str(answer).strip().upper()) if answer is not None and pred is not None else False,
            "task_type": item.get("task_type"),
            "sub_type": item.get("sub_type"),
            "ability_type": item.get("ability_type"),
            "source": item.get("source"),
            "split": item.get("split"),
        }
        rows.append(row)

    write_jsonl(args.output, rows)
    summary = summarize(rows)
    summary_path = str(Path(args.output).with_suffix(".summary.json"))
    Path(summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
