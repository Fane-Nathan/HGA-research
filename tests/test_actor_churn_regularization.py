import unittest

import torch

from tmrl.custom.custom_algorithms import (
    _compute_actor_churn_loss,
    _compute_critic_churn_loss,
    _ema_update_module,
)


class TestActorChurnRegularization(unittest.TestCase):
    def test_disabled_churn_loss_is_zero_even_when_policy_drift_exists(self):
        current_mu = torch.tensor([[1.0, -1.0, 0.5]])
        anchor_mu = torch.zeros_like(current_mu)

        raw_loss, weighted_loss = _compute_actor_churn_loss(
            current_mu,
            anchor_mu,
            enabled=False,
            weight=0.01,
        )

        self.assertEqual(raw_loss.item(), 0.0)
        self.assertEqual(weighted_loss.item(), 0.0)

    def test_enabled_churn_loss_penalizes_pretanh_mean_drift(self):
        current_mu = torch.tensor([[1.0, -1.0, 0.5]])
        anchor_mu = torch.zeros_like(current_mu)

        raw_loss, weighted_loss = _compute_actor_churn_loss(
            current_mu,
            anchor_mu,
            enabled=True,
            weight=0.02,
        )

        expected_raw = ((current_mu - anchor_mu) ** 2).mean().item()
        self.assertAlmostEqual(raw_loss.item(), expected_raw)
        self.assertAlmostEqual(weighted_loss.item(), expected_raw * 0.02)

    def test_disabled_critic_churn_loss_is_zero_even_when_q_drift_exists(self):
        current_q = torch.tensor([[[1.0, -0.5]], [[0.25, 0.75]]])
        anchor_q = torch.zeros_like(current_q)

        raw_loss, weighted_loss = _compute_critic_churn_loss(
            current_q,
            anchor_q,
            enabled=False,
            weight=0.01,
        )

        self.assertEqual(raw_loss.item(), 0.0)
        self.assertEqual(weighted_loss.item(), 0.0)

    def test_enabled_critic_churn_loss_penalizes_q_output_drift(self):
        current_q = torch.tensor([[[1.0, -0.5]], [[0.25, 0.75]]])
        anchor_q = torch.zeros_like(current_q)

        raw_loss, weighted_loss = _compute_critic_churn_loss(
            current_q,
            anchor_q,
            enabled=True,
            weight=0.03,
        )

        expected_raw = ((current_q - anchor_q) ** 2).mean().item()
        self.assertAlmostEqual(raw_loss.item(), expected_raw)
        self.assertAlmostEqual(weighted_loss.item(), expected_raw * 0.03)

    def test_ema_update_module_moves_anchor_toward_source(self):
        anchor = torch.nn.Linear(2, 1)
        source = torch.nn.Linear(2, 1)
        with torch.no_grad():
            anchor.weight.fill_(0.0)
            anchor.bias.fill_(0.0)
            source.weight.fill_(1.0)
            source.bias.fill_(1.0)

        _ema_update_module(anchor, source, polyak=0.75)

        self.assertTrue(torch.allclose(anchor.weight, torch.full_like(anchor.weight, 0.25)))
        self.assertTrue(torch.allclose(anchor.bias, torch.full_like(anchor.bias, 0.25)))


if __name__ == "__main__":
    unittest.main()
