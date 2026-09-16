from __future__ import annotations

import unittest
from pathlib import Path

from physground.media import resolve_content


class ResolveContentTests(unittest.TestCase):
    def test_preserves_remote_url(self) -> None:
        resolved = resolve_content([{"type": "image", "url": "https://example.org/image.png"}])
        self.assertEqual(
            resolved,
            [{"type": "image", "url": "https://example.org/image.png"}],
        )

    def test_resolves_local_path_against_root(self) -> None:
        resolved = resolve_content([{"type": "image", "path": "images/example.png"}], "dataset")
        self.assertEqual(Path(resolved[0]["path"]), Path("dataset/images/example.png"))

    def test_remote_video_cannot_request_frame_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "local video path"):
            resolve_content(
                [{"type": "video", "url": "https://example.org/video.mp4", "end_frame": 10}]
            )


if __name__ == "__main__":
    unittest.main()
