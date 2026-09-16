from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from torch import nn

    from physground.config import GroundingConfig, HeadSpec
    from physground.modeling import DeltaEncoder, PhysGroundModel, PromptRepresentationPooler

    TORCH_RUNTIME_AVAILABLE = True
except ImportError:
    TORCH_RUNTIME_AVAILABLE = False


@unittest.skipUnless(TORCH_RUNTIME_AVAILABLE, "torch/transformers/peft runtime is not installed")
class RepresentationModuleTests(unittest.TestCase):
    def test_prompt_end_and_mean_pooling(self) -> None:
        hidden = torch.tensor([[[1.0], [2.0], [9.0], [20.0]]])
        anchor = torch.tensor([1])
        attention = torch.tensor([[1, 1, 1, 0]])
        end = PromptRepresentationPooler("prompt_end")(hidden, anchor, attention)
        mean = PromptRepresentationPooler("prompt_mean")(hidden, anchor, attention)
        torch.testing.assert_close(end, torch.tensor([[2.0]]))
        torch.testing.assert_close(mean, torch.tensor([[1.5]]))

    def test_delta_encoder_shape_and_gradient(self) -> None:
        encoder = DeltaEncoder(4)
        left = torch.randn(2, 4, requires_grad=True)
        right = torch.randn(2, 4, requires_grad=True)
        result = encoder(left, right, torch.tensor([0.5, 1.0]))
        self.assertEqual(tuple(result.shape), (2, 4))
        result.sum().backward()
        self.assertIsNotNone(left.grad)
        self.assertIsNotNone(right.grad)


if TORCH_RUNTIME_AVAILABLE:

    class DummyBackbone(nn.Module):
        def __init__(self, hidden_size: int = 4) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.embedding = nn.Embedding(32, hidden_size)
            self.lm_head = nn.Linear(hidden_size, 32)

        def forward(self, input_ids, labels=None, **kwargs):
            hidden = self.embedding(input_ids)
            logits = self.lm_head(hidden)
            loss = logits.mean() if labels is not None else None
            return SimpleNamespace(hidden_states=(hidden,), logits=logits, loss=loss)

        def save_pretrained(self, path) -> None:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")


@unittest.skipUnless(TORCH_RUNTIME_AVAILABLE, "torch/transformers/peft runtime is not installed")
class PhysGroundModelTests(unittest.TestCase):
    def _config(self) -> GroundingConfig:
        return GroundingConfig(
            heads=[
                HeadSpec("contact", "state", "classification", num_labels=2),
                HeadSpec("velocity", "delta", "regression", num_labels=3),
            ],
            lm_weight=1,
            pooling="prompt_end",
            head_hidden_dim=3,
        )

    def test_forward_combines_lm_state_and_delta_losses(self) -> None:
        model = PhysGroundModel(DummyBackbone(), self._config())
        common = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        output = model(
            **common,
            labels=torch.tensor([[-100, -100, 3]]),
            anchor_index=torch.tensor([1]),
            grounding_targets={
                "contact": torch.tensor([1]),
                "velocity": torch.tensor([[0.1, 0.2, 0.3]]),
            },
            grounding_masks={"contact": torch.tensor([1.0]), "velocity": torch.tensor([1.0])},
            delta_t_inputs=common,
            delta_tp_inputs={**common, "input_ids": torch.tensor([[1, 2, 4]])},
            delta_tau=torch.tensor([0.5]),
        )
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(set(output["loss_parts"]), {"lm", "contact", "velocity"})
        output["loss"].backward()
        self.assertIsNotNone(model.heads["contact"].weight.grad)
        self.assertIsNotNone(model.group_projectors["state"].network[1].weight.grad)
        self.assertIsNotNone(model.delta_encoder)
        self.assertIsNotNone(model.delta_encoder.network[0].weight.grad)

    def test_sft_only_model_does_not_allocate_delta_encoder(self) -> None:
        config = GroundingConfig(heads=[], lm_weight=1)
        model = PhysGroundModel(DummyBackbone(), config)
        self.assertIsNone(model.delta_encoder)

    def test_sft_only_checkpoint_round_trip(self) -> None:
        config = GroundingConfig(heads=[], lm_weight=1)
        source = PhysGroundModel(DummyBackbone(), config)
        with tempfile.TemporaryDirectory() as directory:
            source.save_physground(directory)
            restored = PhysGroundModel.from_checkpoint(DummyBackbone(), directory)
            self.assertIsNone(restored.delta_encoder)

    def test_safe_auxiliary_checkpoint_round_trip(self) -> None:
        source = PhysGroundModel(DummyBackbone(), self._config())
        with tempfile.TemporaryDirectory() as directory:
            source.save_physground(directory)
            config = json.loads((Path(directory) / "physground_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["format_version"], 2)
            self.assertTrue((Path(directory) / "physground_aux.safetensors").exists())

            restored = PhysGroundModel.from_checkpoint(DummyBackbone(), directory)
            for source_value, restored_value in zip(source.heads.parameters(), restored.heads.parameters()):
                torch.testing.assert_close(source_value, restored_value)


if __name__ == "__main__":
    unittest.main()
