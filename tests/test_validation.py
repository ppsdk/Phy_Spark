from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from physground.config import GroundingConfig, HeadSpec
from physground.validation import validate_manifest_rows


def grounding_config() -> GroundingConfig:
    return GroundingConfig(
        heads=[
            HeadSpec(name="contact", group="state", kind="classification", num_labels=2),
            HeadSpec(name="velocity", group="delta", kind="regression", num_labels=3),
        ],
        jepa_weight=0.2,
        jepa_dim=2,
        jepa_source="facebook/vjepa2-test",
        jepa_feature_layer="encoder.final.mean_pool",
    )


class ManifestValidationTests(unittest.TestCase):
    def test_accepts_valid_sample(self) -> None:
        row = {
            "content": [
                {"type": "video", "url": "https://example.org/sample.mp4"},
                {"type": "text", "text": "What happens?"},
            ],
            "answer": "It collides.",
            "targets": {"contact": 1, "velocity": [0.1, 0.2, 0.3]},
            "target_masks": {"contact": 1, "velocity": 1},
            "delta": {
                "tau": 0.5,
                "t": {"media": [{"type": "image", "url": "https://example.org/t.jpg"}]},
                "tp": {"media": [{"type": "image", "url": "https://example.org/tp.jpg"}]},
            },
            "teacher_delta": [0.1, -0.2],
        }
        self.assertEqual(validate_manifest_rows([row], grounding_config()), [])

    def test_reports_target_and_teacher_shape_errors(self) -> None:
        row = {
            "content": [{"type": "text", "text": "Question"}],
            "targets": {"contact": 2, "velocity": [0.1, 0.2]},
            "delta": {
                "t": {"media": [{"type": "image", "url": "https://example.org/t.jpg"}]},
                "tp": {"media": [{"type": "image", "url": "https://example.org/tp.jpg"}]},
            },
            "teacher_delta": [0.1],
        }
        errors = validate_manifest_rows([row], grounding_config())
        self.assertTrue(any("outside" in error for error in errors))
        self.assertTrue(any("3 finite numbers" in error for error in errors))
        self.assertTrue(any("does not match jepa_dim" in error for error in errors))

    def test_checks_local_media_when_root_is_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = {
                "content": [
                    {"type": "image", "path": "missing.png"},
                    {"type": "text", "text": "Question"},
                ],
                "answer": "Answer",
            }
            errors = validate_manifest_rows([row], media_root=Path(directory))
        self.assertTrue(any("missing media" in error for error in errors))

    def test_rejects_unsupervised_sample(self) -> None:
        row = {"content": [{"type": "text", "text": "Question"}]}
        errors = validate_manifest_rows([row])
        self.assertIn("row 0: sample has no active supervision", errors)

    def test_rejects_null_teacher_path(self) -> None:
        row = {
            "content": [{"type": "text", "text": "Question"}],
            "answer": "Answer",
            "teacher_delta_path": None,
        }
        errors = validate_manifest_rows([row])
        self.assertIn("row 0: teacher_delta_path cannot be null", errors)

    def test_rejects_delta_target_without_delta_observations(self) -> None:
        row = {
            "content": [{"type": "text", "text": "Question"}],
            "targets": {"velocity": [0.1, 0.2, 0.3]},
        }
        errors = validate_manifest_rows([row], grounding_config())
        self.assertTrue(any("active delta target" in error for error in errors))
        self.assertTrue(any("no active supervision" in error for error in errors))

    def test_teacher_is_not_supervision_when_jepa_is_disabled(self) -> None:
        row = {
            "content": [{"type": "text", "text": "Question"}],
            "delta": {
                "t": {"media": [{"type": "image", "url": "https://example.org/t.jpg"}]},
                "tp": {"media": [{"type": "image", "url": "https://example.org/tp.jpg"}]},
            },
            "teacher_delta": [0.1, -0.2],
        }
        config = GroundingConfig(jepa_weight=0, jepa_dim=0)
        errors = validate_manifest_rows([row], config)
        self.assertTrue(any("jepa_weight is zero" in error for error in errors))
        self.assertTrue(any("no active supervision" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
