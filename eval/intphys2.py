from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from physground.inference import generate_response, load_runtime
from physground.utils import parse_binary, write_jsonl


PROMPT = (
    "Analyze the video as a physical event. Ignore rendering quality and playback speed. "
    "Is the object behavior physically plausible under ordinary Earth physics? Answer only Yes or No."
)


def infer_gt(type_value: str) -> int | None:
    s = str(type_value).lower()
    if "impossible" in s or "implausible" in s:
        return 0
    if "possible" in s or "plausible" in s:
        return 1
    return None


def summarize(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r.get("ground_truth") in {0, 1}]
    out = {"n": len(rows), "n_labeled": len(labeled)}
    if labeled:
        out["video_accuracy"] = sum(r["prediction"] == r["ground_truth"] for r in labeled) / len(labeled)

    by_type = defaultdict(list)
    for r in labeled:
        by_type[str(r.get("type"))].append(r)
    if by_type:
        out["by_type"] = {
            k: {"n": len(v), "accuracy": sum(x["prediction"] == x["ground_truth"] for x in v) / len(v)}
            for k, v in sorted(by_type.items())
        }

    scenes = defaultdict(list)
    for r in labeled:
        scenes[str(r.get("scene_idx"))].append(r)
    if scenes:
        pairable = [vals for vals in scenes.values() if len(vals) >= 2]
        if pairable:
            out["scene_all_correct_accuracy"] = sum(
                all(x["prediction"] == x["ground_truth"] for x in vals) for vals in pairable
            ) / len(pairable)
            out["n_scenes"] = len(pairable)

    for field in ("principle", "difficulty", "category", "condition"):
        groups = defaultdict(list)
        for r in labeled:
            if r.get(field) not in (None, "", float("nan")):
                groups[str(r.get(field))].append(r)
        if groups:
            out[f"by_{field}"] = {
                k: {"n": len(v), "accuracy": sum(x["prediction"] == x["ground_truth"] for x in v) / len(v)}
                for k, v in sorted(groups.items())
            }
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate VLMs on IntPhys2 Debug/Main metadata")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--data-root", required=True, help="IntPhys2 root containing Debug/ or Test/")
    p.add_argument("--split", default="Test", choices=["Debug", "Test"])
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--output", default="results/intphys2_predictions.jsonl")
    args = p.parse_args()

    split_root = Path(args.data_root) / args.split
    metadata = pd.read_csv(split_root / "metadata.csv")
    if args.limit:
        metadata = metadata.iloc[: args.limit]

    model, processor = load_runtime(args.model, args.adapter, dtype=args.dtype)
    rows = []
    for _, rec in tqdm(metadata.iterrows(), total=len(metadata), desc=f"IntPhys2 {args.split}"):
        filename = str(rec["filename"])
        video = split_root / "Videos" / (filename if filename.endswith(".mp4") else filename + ".mp4")
        raw = generate_response(
            model,
            processor,
            [{"type": "video", "path": str(video)}, {"type": "text", "text": PROMPT}],
            system_prompt="Judge physical plausibility from the video itself.",
            num_frames=args.num_frames,
            max_new_tokens=8,
        )
        pred = parse_binary(raw)
        row = {
            "scene_idx": rec.get("SceneIndex"),
            "filename": filename,
            "type": rec.get("type"),
            "ground_truth": infer_gt(rec.get("type")),
            "prediction": pred,
            "raw_response": raw,
        }
        for col in ("principle", "Principle", "difficulty", "Difficulty", "category", "condition"):
            if col in rec.index and pd.notna(rec[col]):
                row[col.lower()] = rec[col]
        rows.append(row)

    write_jsonl(args.output, rows)
    summary = summarize(rows)
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
