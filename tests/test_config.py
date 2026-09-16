from __future__ import annotations

import unittest

from physground.config import GroundingConfig, HeadSpec
from physground.validation import validate_training_config


class HeadSpecTests(unittest.TestCase):
    def test_valid_classification_head(self) -> None:
        head = HeadSpec.from_dict(
            {"name": "contact", "group": "state", "kind": "classification", "num_labels": 2}
        )
        self.assertEqual(head.num_labels, 2)

    def test_rejects_invalid_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported head group"):
            HeadSpec(name="contact", group="future", kind="classification", num_labels=2)

    def test_rejects_duplicate_names(self) -> None:
        heads = [
            HeadSpec(name="contact", group="state", kind="classification", num_labels=2),
            HeadSpec(name="contact", group="delta", kind="classification", num_labels=2),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate grounding head"):
            GroundingConfig(heads=heads)

    def test_jepa_dimension_is_required_when_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "jepa_dim"):
            GroundingConfig(jepa_weight=0.1, jepa_dim=0)

    def test_jepa_provenance_is_required_when_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "jepa_source"):
            GroundingConfig(jepa_weight=0.1, jepa_dim=4)
        with self.assertRaisesRegex(ValueError, "jepa_feature_layer"):
            GroundingConfig(jepa_weight=0.1, jepa_dim=4, jepa_source="checkpoint")

    def test_rejects_non_list_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "heads must be a list"):
            GroundingConfig.from_dict({"heads": "contact"})

    def test_rejects_unknown_pooling(self) -> None:
        with self.assertRaisesRegex(ValueError, "pooling mode"):
            GroundingConfig(pooling="cls")

    def test_rejects_negative_head_hidden_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "head_hidden_dim"):
            GroundingConfig(head_hidden_dim=-1)


class TrainingConfigTests(unittest.TestCase):
    def test_minimal_training_config(self) -> None:
        config = {
            "model": {"id": "org/model"},
            "data": {"train_jsonl": "train.jsonl", "num_frames": 8},
            "training": {"learning_rate": 1e-4, "gradient_accumulation_steps": 2},
            "grounding": {"heads": []},
        }
        self.assertIsInstance(validate_training_config(config), GroundingConfig)

    def test_rejects_zero_frames(self) -> None:
        config = {
            "model": {"id": "org/model"},
            "data": {"train_jsonl": "train.jsonl", "num_frames": 0},
        }
        with self.assertRaisesRegex(ValueError, "num_frames"):
            validate_training_config(config)

    def test_rejects_all_zero_objective_weights(self) -> None:
        config = {
            "model": {"id": "org/model"},
            "data": {"train_jsonl": "train.jsonl"},
            "grounding": {"lm_weight": 0, "jepa_weight": 0, "heads": []},
        }
        with self.assertRaisesRegex(ValueError, "no positive-weight objective"):
            validate_training_config(config)


if __name__ == "__main__":
    unittest.main()
