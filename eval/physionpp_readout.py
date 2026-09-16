from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from physground.inference import extract_prompt_representation, load_runtime
from physground.utils import read_jsonl, write_jsonl

QUERY = "Encode the observed physical interaction history for future object-contact prediction."


def content(row: dict):
    return [
        {
            "type": "video",
            "path": row["video"],
            "end_frame": int(row["start_frame_for_prediction"]),
        },
        {"type": "text", "text": QUERY},
    ]


def extract(rows, model, processor, root, num_frames):
    feats, labels, props, ids = [], [], [], []
    kept_rows = []
    for row in tqdm(rows, desc="Extract Physion++ features"):
        if row.get("label") not in {0, 1}:
            continue
        h = extract_prompt_representation(
            model,
            processor,
            content(row),
            media_root=root,
            system_prompt="Represent the observed physical scene without looking beyond the prediction point.",
            num_frames=num_frames,
        )
        feats.append(h.numpy()[0])
        labels.append(int(row["label"]))
        props.append(str(row.get("property") or "unknown"))
        ids.append(row.get("row_id"))
        kept_rows.append(row)
    if not feats:
        raise ValueError("No labeled Physion++ samples were available for representation extraction")
    return np.stack(feats), np.asarray(labels), np.asarray(props), ids, kept_rows


def main():
    p = argparse.ArgumentParser(description="Physion++ frozen-representation readout protocol")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--readout-manifest", required=True)
    p.add_argument("--readout-root", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--test-root", required=True)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--output", default="results/physionpp_readout.jsonl")
    args = p.parse_args()

    model, processor = load_runtime(args.model, args.adapter, dtype=args.dtype)
    tr = read_jsonl(args.readout_manifest)
    te = read_jsonl(args.test_manifest)
    x_tr, y_tr, p_tr, _, _ = extract(tr, model, processor, args.readout_root, args.num_frames)
    x_te, y_te, p_te, _, rows_te = extract(te, model, processor, args.test_root, args.num_frames)

    predictions = np.full(len(y_te), -1, dtype=int)
    metrics = {}
    for prop in sorted(set(p_te.tolist())):
        train_mask = p_tr == prop
        test_mask = p_te == prop
        if train_mask.sum() < 2 or len(np.unique(y_tr[train_mask])) < 2:
            metrics[prop] = {"error": "insufficient readout labels"}
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(x_tr[train_mask], y_tr[train_mask])
        predictions[test_mask] = clf.predict(x_te[test_mask])
        metrics[prop] = {
            "n": int(test_mask.sum()),
            "accuracy": float(accuracy_score(y_te[test_mask], predictions[test_mask])),
        }

    valid = predictions >= 0
    summary = {
        "n": int(valid.sum()),
        "accuracy": float(accuracy_score(y_te[valid], predictions[valid])) if valid.any() else None,
        "by_property": metrics,
    }
    out_rows = []
    for row, pred in zip(rows_te, predictions.tolist()):
        out_rows.append({**row, "prediction": None if pred < 0 else pred})
    write_jsonl(args.output, out_rows)
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
