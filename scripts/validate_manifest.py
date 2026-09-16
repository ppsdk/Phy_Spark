from __future__ import annotations

import argparse
from pathlib import Path

from physground.utils import read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--media-root", default=None)
    args = p.parse_args()
    root = Path(args.media_root).resolve() if args.media_root else None
    rows = read_jsonl(args.manifest)
    errors = []
    for i, row in enumerate(rows):
        if not any(k in row for k in ("content", "media")):
            errors.append(f"row {i}: missing content/media")
        if "targets" in row and not isinstance(row["targets"], dict):
            errors.append(f"row {i}: targets must be an object")
        items = row.get("content") or row.get("media") or []
        for item in items:
            if item.get("type") not in {"text", "image", "video"}:
                errors.append(f"row {i}: invalid content type {item.get('type')}")
            path = item.get("path")
            if root is not None and isinstance(path, str):
                pth = Path(path)
                if not pth.is_absolute() and not (root / pth).exists():
                    errors.append(f"row {i}: missing media {root / pth}")
    print(f"rows={len(rows)} errors={len(errors)}")
    for e in errors[:100]:
        print("-", e)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
