from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import numpy as np


def _resolve_path(path: str | Path, root: str | Path | None) -> str:
    p = Path(path)
    if not p.is_absolute() and root is not None:
        p = Path(root) / p
    return str(p)


def decode_video_segment(
    path: str | Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    num_frames: int | None = None,
) -> np.ndarray:
    """Decode a video segment to uint8 [T,H,W,3]. end_frame is exclusive."""
    path = str(path)
    start = max(int(start_frame or 0), 0)
    frames: list[np.ndarray] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx < start:
                continue
            if end_frame is not None and idx >= int(end_frame):
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"No frames decoded from {path} in [{start}, {end_frame})")
    if num_frames is not None and len(frames) > int(num_frames):
        ids = np.linspace(0, len(frames) - 1, int(num_frames), dtype=int)
        frames = [frames[i] for i in ids]
    return np.stack(frames, axis=0)


def resolve_content(
    content: list[dict[str, Any]],
    media_root: str | Path | None = None,
    default_num_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Convert canonical content items into Transformers multimodal chat items."""
    out: list[dict[str, Any]] = []
    for item in content:
        item = dict(item)
        typ = item.get("type")
        if typ == "text":
            out.append({"type": "text", "text": str(item.get("text", ""))})
            continue
        if typ not in {"image", "video"}:
            raise ValueError(f"Unsupported content type: {typ}")

        raw_path = item.get("path") or item.get("url")
        if raw_path is None:
            raise ValueError(f"Media item must have path/url: {item}")

        # A list of image paths can be passed as a decoded-frame video source.
        if isinstance(raw_path, list):
            paths = [_resolve_path(p, media_root) for p in raw_path]
            out.append({"type": typ, "path": paths})
            continue

        path = _resolve_path(raw_path, media_root)
        needs_decode = typ == "video" and any(
            key in item for key in ("start_frame", "end_frame", "num_frames")
        )
        if needs_decode:
            video = decode_video_segment(
                path,
                start_frame=item.get("start_frame"),
                end_frame=item.get("end_frame"),
                num_frames=item.get("num_frames", default_num_frames),
            )
            out.append({"type": "video", "video": video})
        else:
            media_item: dict[str, Any] = {"type": typ, "path": path}
            out.append(media_item)
    return out
