from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from physground.utils import nested_get, write_jsonl


def normalize_label(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    s = str(value).strip().lower()
    if s in {"1", "yes", "true", "contact", "hit", "positive"}:
        return 1
    if s in {"0", "no", "false", "no_contact", "negative"}:
        return 0
    return None


def infer_property(path: Path) -> str | None:
    lower = str(path).lower()
    for name in ("mass", "friction", "elasticity", "deformability"):
        if name in lower:
            return name
    return None


def main():
    p = argparse.ArgumentParser(description="Build a canonical Physion++ manifest from official released files")
    p.add_argument("--root", required=True, help="Training_data, Readout_data, or Testing_data root")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    root = Path(args.root).resolve()
    rows = []
    skipped = []
    for meta_path in sorted(root.rglob("*.pkl")):
        try:
            # Only load pickle files from the trusted official Physion++ release.
            with meta_path.open("rb") as f:
                meta = pickle.load(f)
        except Exception as exc:
            skipped.append((str(meta_path), f"pickle error: {exc}"))
            continue

        stem = meta_path.stem
        video = meta_path.with_name(stem + "_image.mp4")
        if not video.exists():
            alt = meta_path.with_suffix(".mp4")
            video = alt if alt.exists() else video
        if not video.exists():
            skipped.append((str(meta_path), "missing RGB video"))
            continue

        start_frame = nested_get(meta, ["start_frame_for_prediction"])
        label = nested_get(meta, ["label", "contact_label", "target_contact", "is_contact"])
        label = normalize_label(label)
        if start_frame is None:
            skipped.append((str(meta_path), "missing start_frame_for_prediction"))
            continue

        rows.append(
            {
                "row_id": stem,
                "video": str(video.relative_to(root)),
                "meta": str(meta_path.relative_to(root)),
                "start_frame_for_prediction": int(start_frame),
                "label": label,
                "property": infer_property(meta_path),
            }
        )

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} records -> {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} files. First 20 reasons:")
        for path, reason in skipped[:20]:
            print(f"  - {path}: {reason}")
    unresolved = sum(r["label"] is None for r in rows)
    if unresolved:
        print(f"WARNING: {unresolved} records have no recognized contact label; usable for representation extraction but not accuracy.")


if __name__ == "__main__":
    main()
