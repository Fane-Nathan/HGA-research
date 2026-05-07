import unittest

from tmrl.custom.custom_algorithms import (
    HealthGatedFloorController,
    _gate_awdb_min_weight,
    _compute_actor_guard_decision,
    _compute_det_skill_transfer_feedback,
    _compute_q_action_sensitivity_regularizer,
    _compute_hgi_imagination_gate,
    _compute_hgi_trust_metrics,
    _compute_hgi_warmup_gate,
)
import torch


class TestHgiActorGuard(unittest.TestCase):
    def test_grad_health_only_does_not_hard_throttle_healthy_actor_signal(self):
        decision = _compute_actor_guard_decision(
            guard_enabled=True,
            grad_health_mean=0.02,
            dqda_norm_value=1.8e-4,
            q_pi_std_value=0.041,
            q_action_sensitivity_value=0.005,
        )

        self.assertTrue(decision["grad_health_only"])
        self.assertTrue(decision["healthy_signal_override"])
        self.assertEqual(decision["guard_tier"], 0.0)
        self.assertEqual(decision["lr_scale"], 1.0)
        self.assertEqual(decision["throttle_reason"], 0.0)

    def test_critical_grad_health_only_is_overridden_when_critic_signal_is_healthy(self):
        decision = _compute_actor_guard_decision(
            guard_enabled=True,
            grad_health_mean=0.005,
            dqda_norm_value=1.8e-4,
            q_pi_std_value=0.041,
            q_action_sensitivity_value=0.005,
        )

        self.assertTrue(decision["grad_health_critical"])
        self.assertFalse(decision["grad_health_severe"])
        self.assertTrue(decision["healthy_signal_override"])
        self.assertEqual(decision["guard_tier"], 0.0)
        self.assertEqual(decision["lr_scale"], 1.0)

    def test_severe_grad_health_still_hard_blocks(self):
        decision = _compute_actor_guard_decision(
            guard_enabled=True,
            grad_health_mean=0.0005,
            dqda_norm_value=1.8e-4,
            q_pi_std_value=0.041,
            q_action_sensitivity_value=0.005,
        )

        self.assertTrue(decision["grad_health_severe"])
        self.assertFalse(decision["healthy_signal_override"])
        self.assertEqual(decision["guard_tier"], 3.0)
        self.assertTrue(decision["tier3_hard_block_active"])
        self.assertEqual(decision["throttle_reason"], 4.0)

    def test_flat_critic_with_low_grad_health_still_triggers_tier1(self):
        decision = _compute_actor_guard_decision(
            guard_enabled=True,
            grad_health_mean=0.02,
            dqda_norm_value=1.8e-4,
            q_pi_std_value=0.01,
            q_action_sensitivity_value=0.005,
        )

        self.assertFalse(decision["healthy_signal_override"])
        self.assertEqual(decision["guard_tier"], 1.0)
        self.assertEqual(decision["lr_scale"], 0.1)
        self.assertEqual(decision["throttle_reason"], 3.0)

    def test_dqda_starvation_uses_starvation_recovery_scale(self):
        decision = _compute_actor_guard_decision(
            guard_enabled=True,
            grad_health_mean=0.02,
            dqda_norm_value=5e-5,
            q_pi_std_value=0.041,
            q_action_sensitivity_value=0.005,
        )

        self.assertTrue(decision["dqda_starving"])
        self.assertEqual(decision["guard_tier"], 1.0)
        self.assertEqual(decision["lr_scale"], 0.5)
        self.assertEqual(decision["throttle_reason"], 1.0)

    def test_q_overconfidence_soft_throttles_actor_without_env_specific_return_scale(self):
        decision = _compute_actor_guard_decision(
            guard_enabled=True,
            grad_health_mean=0.08,
            dqda_norm_value=2.0e-4,
            q_pi_std_value=0.08,
            q_action_sensitivity_value=0.006,
            q_overconfidence_norm=1.25,
            churn_norm=1.0,
            data_liveness_health=1.0,
            q_overconfidence_soft=0.5,
            q_overconfidence_hard=1.25,
            adaptive_lr_floor=0.05,
        )

        self.assertEqual(decision["guard_tier"], 1.0)
        self.assertEqual(decision["throttle_reason"], 7.0)
        self.assertAlmostEqual(decision["q_overconfidence_risk"], 1.0)
        self.assertAlmostEqual(decision["lr_scale"], 0.05)

    def test_relative_churn_spike_throttles_actor(self):
        decision = _compute_actor_guard_decision(
            guard_enabled=True,
            grad_health_mean=0.08,
            dqda_norm_value=2.0e-4,
            q_pi_std_value=0.08,
            q_action_sensitivity_value=0.006,
            q_overconfidence_norm=0.0,
            churn_norm=8.0,
            data_liveness_health=1.0,
            churn_soft_ratio=3.0,
            churn_hard_ratio=8.0,
            adaptive_lr_floor=0.05,
        )

        self.assertEqual(decision["guard_tier"], 1.0)
        self.assertEqual(decision["throttle_reason"], 8.0)
        self.assertAlmostEqual(decision["churn_risk"], 1.0)
        self.assertAlmostEqual(decision["lr_scale"], 0.05)

    def test_relative_reward_liveness_drop_throttles_actor(self):
        decision = _compute_actor_guard_decision(
            guard_enabled=True,
            grad_health_mean=0.08,
            dqda_norm_value=2.0e-4,
            q_pi_std_value=0.08,
            q_action_sensitivity_value=0.006,
            q_overconfidence_norm=0.0,
            churn_norm=1.0,
            data_liveness_health=0.0,
            adaptive_lr_floor=0.05,
        )

        self.assertEqual(decision["guard_tier"], 1.0)
        self.assertEqual(decision["throttle_reason"], 9.0)
        self.assertAlmostEqual(decision["data_liveness_risk"], 1.0)
        self.assertAlmostEqual(decision["lr_scale"], 0.05)


class TestHealthGatedEntropy(unittest.TestCase):
    def test_disabled_entropy_controller_keeps_base_floor(self):
        base_floor = torch.tensor([0.02, 0.05, 0.05])

        controller = HealthGatedFloorController(
            base_floor,
            {"HEALTH_GATED_ENTROPY_ENABLED": False},
        )
        floor, diag = controller.step(
            {"churn/anchor_mu_abs_diff": 2.0},
            {"HEALTH_GATED_ENTROPY_ENABLED": False},
        )

        self.assertTrue(torch.allclose(floor, base_floor))
        self.assertEqual(diag["entropy_health/enabled"], 0.0)
        self.assertEqual(diag["entropy_health/floor_actual_steer"], base_floor[0].item())

    def test_high_actor_churn_with_healthy_critic_consolidates_entropy_floor(self):
        base_floor = torch.tensor([0.02, 0.05, 0.05])
        controller = HealthGatedFloorController(
            base_floor,
            {
                "HEALTH_ENTROPY_ETA": 1.0,
                "HEALTH_ENTROPY_CHURN_SIGMA": 0.01,
                "HEALTH_ENTROPY_HYSTERESIS_ROUNDS": 0,
                "HEALTH_ENTROPY_SLEW_FRAC": 10.0,
            },
        )

        floor, diag = controller.step(
            {
                "churn/anchor_mu_abs_diff": 2.0,
                "bridge/dqda_norm": 1.0e-3,
                "bridge/q_pi_std": 0.2,
                "bridge/q_action_sensitivity": 0.005,
                "grad_health_steer": 1.0,
                "grad_health_gas": 1.0,
                "grad_health_brake": 1.0,
                "guard/tier": 0.0,
                "hgi/critic_health": 1.0,
                "hgi/model_trust": 1.0,
            },
            {
                "HEALTH_ENTROPY_ETA": 1.0,
                "HEALTH_ENTROPY_CHURN_SIGMA": 0.01,
                "HEALTH_ENTROPY_HYSTERESIS_ROUNDS": 0,
                "HEALTH_ENTROPY_SLEW_FRAC": 10.0,
            },
        )

        self.assertTrue(torch.allclose(floor, base_floor * 0.25, atol=1e-5))
        self.assertLess(diag["entropy_health/g_consolidate"], 0.26)
        self.assertEqual(diag["entropy_health/critic_ok"], 1.0)

    def test_starving_critic_signal_restores_loosened_floor(self):
        base_floor = torch.tensor([0.02, 0.05, 0.05])
        cfg = {
            "HEALTH_ENTROPY_ETA": 1.0,
            "HEALTH_ENTROPY_CHURN_SIGMA": 0.01,
            "HEALTH_ENTROPY_HYSTERESIS_ROUNDS": 0,
            "HEALTH_ENTROPY_SLEW_FRAC": 10.0,
        }
        controller = HealthGatedFloorController(base_floor, cfg)
        lowered, _ = controller.step(
            {
                "churn/anchor_mu_abs_diff": 2.0,
                "bridge/dqda_norm": 1.0e-3,
                "bridge/q_pi_std": 0.2,
                "bridge/q_action_sensitivity": 0.005,
                "grad_health_steer": 1.0,
                "grad_health_gas": 1.0,
                "grad_health_brake": 1.0,
                "guard/tier": 0.0,
                "hgi/critic_health": 1.0,
                "hgi/model_trust": 1.0,
            },
            cfg,
        )

        floor, diag = controller.step(
            {
                "churn/anchor_mu_abs_diff": 0.0,
                "bridge/dqda_norm": 1e-6,
                "bridge/q_pi_std": 0.2,
                "bridge/q_action_sensitivity": 0.005,
                "grad_health_steer": 1.0,
                "grad_health_gas": 1.0,
                "grad_health_brake": 1.0,
                "guard/tier": 0.0,
                "hgi/critic_health": 0.0,
                "hgi/model_trust": 1.0,
            },
            cfg,
        )

        self.assertTrue(torch.all(lowered < base_floor))
        self.assertTrue(torch.allclose(floor, base_floor, atol=1e-6))
        self.assertGreater(diag["entropy_health/g_search"], 3.0)

    def test_starving_signal_never_exceeds_baseline_floor(self):
        base_floor = torch.tensor([0.02, 0.05, 0.05])
        controller = HealthGatedFloorController(
            base_floor,
            {"HEALTH_ENTROPY_ETA": 1.0, "HEALTH_ENTROPY_HYSTERESIS_ROUNDS": 0},
        )

        floor, diag = controller.step(
            {
                "bridge/dqda_norm": 1e-6,
                "bridge/q_pi_std": 0.2,
                "bridge/q_action_sensitivity": 0.005,
                "grad_health_steer": 1.0,
                "grad_health_gas": 1.0,
                "grad_health_brake": 1.0,
                "hgi/model_trust": 1.0,
            },
            {"HEALTH_ENTROPY_ETA": 1.0, "HEALTH_ENTROPY_HYSTERESIS_ROUNDS": 0},
        )

        self.assertTrue(torch.allclose(floor, base_floor, atol=1e-6))
        self.assertGreater(diag["entropy_health/g_search"], 3.0)

    def test_preserve_gate_is_neutral_when_deterministic_is_not_winning(self):
        base_floor = torch.tensor([0.02, 0.05, 0.05])
        cfg = {
            "HEALTH_ENTROPY_ETA": 1.0,
            "HEALTH_ENTROPY_HYSTERESIS_ROUNDS": 0,
            "HEALTH_ENTROPY_SLEW_FRAC": 10.0,
            "HEALTH_ENTROPY_DET_ADVANTAGE": 5.0,
        }
        controller = HealthGatedFloorController(base_floor, cfg)

        floor, diag = controller.step(
            {
                "bridge/dqda_norm": 1.0e-3,
                "bridge/q_pi_std": 0.2,
                "bridge/q_action_sensitivity": 0.005,
                "grad_health_steer": 1.0,
                "grad_health_gas": 1.0,
                "grad_health_brake": 1.0,
                "hgi/critic_health": 1.0,
                "hgi/model_trust": 1.0,
                "return_test_det": 1.0,
                "return_test_stoch": 3.0,
            },
            cfg,
        )

        self.assertTrue(torch.allclose(floor, base_floor, atol=1e-6))
        self.assertEqual(diag["entropy_health/g_consolidate"], 1.0)
        self.assertEqual(diag["entropy_health/g_preserve"], 1.0)
        self.assertEqual(diag["entropy_health/preserve_active"], 0.0)

    def test_preserve_gate_lowers_floor_only_after_deterministic_wins(self):
        base_floor = torch.tensor([0.02, 0.05, 0.05])
        cfg = {
            "HEALTH_ENTROPY_ETA": 1.0,
            "HEALTH_ENTROPY_HYSTERESIS_ROUNDS": 0,
            "HEALTH_ENTROPY_SLEW_FRAC": 10.0,
            "HEALTH_ENTROPY_DET_ADVANTAGE": 5.0,
            "HEALTH_ENTROPY_DET_SIGMA": 1.0,
        }
        controller = HealthGatedFloorController(base_floor, cfg)

        floor, diag = controller.step(
            {
                "bridge/dqda_norm": 1.0e-3,
                "bridge/q_pi_std": 0.2,
                "bridge/q_action_sensitivity": 0.005,
                "grad_health_steer": 1.0,
                "grad_health_gas": 1.0,
                "grad_health_brake": 1.0,
                "hgi/critic_health": 1.0,
                "hgi/model_trust": 1.0,
                "return_test_det": 12.0,
                "return_test_stoch": 3.0,
            },
            cfg,
        )

        self.assertTrue(torch.all(floor < base_floor))
        self.assertLess(diag["entropy_health/g_preserve"], 1.0)
        self.assertEqual(diag["entropy_health/preserve_active"], 1.0)

    def test_positive_log_std_blocks_stagnation_recovery_floor_reset(self):
        base_floor = torch.tensor([0.02, 0.05, 0.05])
        cfg = {
            "HEALTH_ENTROPY_ETA": 1.0,
            "HEALTH_ENTROPY_CHURN_SIGMA": 0.01,
            "HEALTH_ENTROPY_HYSTERESIS_ROUNDS": 0,
            "HEALTH_ENTROPY_SLEW_FRAC": 10.0,
            "HEALTH_ENTROPY_STAGNATION_ROUNDS": 1,
            "HEALTH_ENTROPY_DET_ADVANTAGE": 5.0,
        }
        controller = HealthGatedFloorController(base_floor, cfg)
        common = {
            "bridge/dqda_norm": 1e-3,
            "bridge/q_pi_std": 0.08,
            "bridge/q_action_sensitivity": 0.04,
            "hgi/critic_health_pre_eval": 1.0,
            "hgi/model_trust": 1.0,
            "churn/anchor_mu_abs_diff": 2.0,
            "guard/grad_health_mean": 0.002,
            "debug_log_std_mean": 0.1,
        }

        controller.step(
            dict(common, return_test_det=82.0, return_test_stoch=8.0),
            cfg,
        )
        floor, diag = controller.step(
            dict(common, return_test_det=1.0, return_test_stoch=1.0),
            cfg,
        )

        self.assertEqual(diag["entropy_health/stagnation_active"], 1.0)
        self.assertAlmostEqual(diag["entropy_health/g_recovery_steer"], 1.0, places=5)
        self.assertTrue(torch.all(floor < base_floor))

    def test_awdb_min_weight_requires_positive_mean_q_advantage(self):
        self.assertEqual(
            _gate_awdb_min_weight(
                advantage_mean=-0.05,
                positive_advantage_mean=0.01,
                target_weight=0.35,
            ),
            0.0,
        )
        self.assertEqual(
            _gate_awdb_min_weight(
                advantage_mean=0.02,
                positive_advantage_mean=0.01,
                target_weight=0.35,
            ),
            0.35,
        )


class TestDeterministicSkillTransfer(unittest.TestCase):
    def test_stochastic_advantage_raises_deterministic_distillation_drive(self):
        feedback = _compute_det_skill_transfer_feedback(
            {
                "return_test_det": 9.0,
                "return_test_stoch": 61.0,
            },
            {
                "DET_REG_LAMBDA": 0.03,
                "DET_SKILL_TRANSFER_LAMBDA_MAX": 0.08,
                "DET_SKILL_TRANSFER_GAP_THRESHOLD": 0.25,
                "DET_SKILL_TRANSFER_GAP_SIGMA": 0.15,
                "DET_SKILL_GAP_SCALE_FLOOR": 1.0,
            },
        )

        self.assertEqual(feedback["skill_transfer/eval_present"], 1.0)
        self.assertGreater(feedback["skill_transfer/stoch_adv_norm"], 1.0)
        self.assertGreater(feedback["skill_transfer/stoch_adv_drive_ema"], 0.99)
        self.assertAlmostEqual(feedback["skill_transfer/det_distill_multiplier"], 0.08 / 0.03, places=2)
        self.assertLess(feedback["skill_transfer/det_stoch_gap_health"], 0.2)

    def test_deterministic_advantage_preserves_without_extra_distillation(self):
        feedback = _compute_det_skill_transfer_feedback(
            {
                "return_test_det": 82.0,
                "return_test_stoch": 9.0,
            },
            {
                "DET_REG_LAMBDA": 0.03,
                "DET_SKILL_TRANSFER_LAMBDA_MAX": 0.08,
            },
        )

        self.assertGreater(feedback["skill_transfer/det_adv_drive_ema"], 0.99)
        self.assertLess(feedback["skill_transfer/stoch_adv_drive_ema"], 0.2)
        self.assertAlmostEqual(feedback["skill_transfer/det_distill_multiplier"], 1.0, places=3)
        self.assertEqual(feedback["skill_transfer/det_stoch_gap_health"], 1.0)

    def test_eval_feedback_ema_smooths_single_round_noise(self):
        first = _compute_det_skill_transfer_feedback(
            {"return_test_det": 9.0, "return_test_stoch": 61.0},
            {"DET_SKILL_TRANSFER_EMA_BETA": 0.8},
        )
        second = _compute_det_skill_transfer_feedback(
            {"return_test_det": 82.0, "return_test_stoch": 9.0},
            {"DET_SKILL_TRANSFER_EMA_BETA": 0.8},
            previous=first,
        )

        self.assertGreater(second["skill_transfer/stoch_adv_drive_ema"], 0.5)
        self.assertLess(second["skill_transfer/stoch_adv_drive_ema"], first["skill_transfer/stoch_adv_drive_ema"])
        self.assertGreater(second["skill_transfer/det_adv_drive_ema"], 0.1)


class TestQActionSensitivityRegularizer(unittest.TestCase):
    def test_squared_mode_preserves_original_loss_shape(self):
        sensitivity = torch.tensor(0.008)

        reg = _compute_q_action_sensitivity_regularizer(
            sensitivity,
            floor=0.05,
            base_weight=1.0,
            max_weight=1.0,
            loss_type="squared",
        )

        self.assertAlmostEqual(reg["gap"].item(), 0.042, places=6)
        self.assertAlmostEqual(reg["raw_loss"].item(), 0.042 ** 2, places=6)
        self.assertAlmostEqual(reg["weighted_loss"].item(), 0.042 ** 2, places=6)
        self.assertEqual(reg["effective_weight"], 1.0)
        self.assertEqual(reg["loss_type_code"], 0.0)

    def test_huber_mode_uses_health_gated_lambda_ramp(self):
        sensitivity = torch.tensor(0.008)

        reg = _compute_q_action_sensitivity_regularizer(
            sensitivity,
            floor=0.05,
            base_weight=1.0,
            max_weight=2.0,
            loss_type="huber",
            huber_beta=0.05,
        )

        expected_drive = 0.042 / 0.05
        expected_lambda = 1.0 + expected_drive
        expected_raw = 0.5 * (0.042 ** 2) / 0.05
        self.assertAlmostEqual(reg["drive"].item(), expected_drive, places=6)
        self.assertAlmostEqual(reg["effective_weight"], expected_lambda, places=6)
        self.assertAlmostEqual(reg["raw_loss"].item(), expected_raw, places=6)
        self.assertAlmostEqual(
            reg["weighted_loss"].item(),
            expected_raw * expected_lambda,
            places=6,
        )
        self.assertEqual(reg["loss_type_code"], 2.0)

    def test_regularizer_is_zero_when_sensitivity_is_healthy(self):
        sensitivity = torch.tensor(0.06)

        reg = _compute_q_action_sensitivity_regularizer(
            sensitivity,
            floor=0.05,
            base_weight=1.0,
            max_weight=2.0,
            loss_type="linear",
        )

        self.assertEqual(reg["gap"].item(), 0.0)
        self.assertEqual(reg["drive"].item(), 0.0)
        self.assertEqual(reg["raw_loss"].item(), 0.0)
        self.assertEqual(reg["weighted_loss"].item(), 0.0)
        self.assertEqual(reg["effective_weight"], 1.0)
        self.assertEqual(reg["loss_type_code"], 1.0)


class TestHgiImaginationGate(unittest.TestCase):
    def _healthy_metrics(self, **overrides):
        metrics = {
            "wm/kl_mean": 0.6,
            "wm_kl_clamped": 1.0,
            "wm/reward_error_abs_mean": 0.001,
            "wm/reward_target_std": 0.1,
            "wm_val/recon_prior_post_ratio": 1.0,
            "wm/verifier_trust_mean": 1.0,
            "bridge/dqda_norm": 1.8e-4,
            "bridge/q_pi_std": 0.041,
            "bridge/q_action_sensitivity": 0.005,
            "guard/grad_health_mean": 0.08,
            "guard/tier": 0.0,
            "hgi/det_stoch_gap_health": 1.0,
        }
        metrics.update(overrides)
        return metrics

    def test_high_trust_uses_full_horizon_after_post_warmup_ramp(self):
        gate = _compute_hgi_imagination_gate(
            {"hgi/model_trust": 0.9, "hgi/critic_health": 0.9, "hgi/imag_trust": 0.81},
            base_horizon=15,
            post_warmup_steps=5000,
        )

        self.assertEqual(gate["hgi/effective_horizon"], 15.0)
        self.assertEqual(gate["hgi/skipped_ratio"], 0.0)
        self.assertEqual(gate["hgi/gate_state"], 3.0)
        self.assertEqual(gate["hgi/ramp_active"], 0.0)
        self.assertEqual(gate["hgi/post_warmup_steps"], 5000.0)
        self.assertEqual(gate["hgi/ramp_remaining_steps"], 0.0)

    def test_high_trust_uses_short_horizon_during_post_warmup_ramp(self):
        gate = _compute_hgi_imagination_gate(
            {"hgi/model_trust": 0.9, "hgi/critic_health": 0.9, "hgi/imag_trust": 0.81},
            {"HGI_POST_WARMUP_SHORT_STEPS": 5000},
            base_horizon=15,
            post_warmup_steps=0,
        )

        self.assertEqual(gate["hgi/effective_horizon"], 3.0)
        self.assertEqual(gate["hgi/skipped_ratio"], 0.0)
        self.assertEqual(gate["hgi/gate_state"], 2.0)
        self.assertEqual(gate["hgi/ramp_active"], 1.0)
        self.assertEqual(gate["hgi/post_warmup_steps"], 0.0)
        self.assertEqual(gate["hgi/ramp_remaining_steps"], 5000.0)
        self.assertEqual(gate["hgi/post_warmup_short_steps"], 5000.0)

    def test_warmup_gate_logs_stable_skip_columns_before_imagination_is_allowed(self):
        gate = _compute_hgi_warmup_gate(
            {"HGI_SHORT_HORIZON": 3, "HGI_FULL_HORIZON": 15, "HGI_POST_WARMUP_SHORT_STEPS": 5000},
            base_horizon=15,
            warmup_remaining_steps=12345,
        )

        self.assertEqual(gate["hgi/effective_horizon"], 0.0)
        self.assertEqual(gate["hgi/skipped_ratio"], 1.0)
        self.assertEqual(gate["hgi/skip_reason_warmup"], 1.0)
        self.assertEqual(gate["hgi/warmup_active"], 1.0)
        self.assertEqual(gate["hgi/warmup_remaining_steps"], 12345.0)
        self.assertEqual(gate["hgi/ramp_active"], 0.0)
        self.assertEqual(gate["hgi/post_warmup_steps"], 0.0)
        self.assertEqual(gate["hgi/ramp_remaining_steps"], 5000.0)
        self.assertEqual(gate["hgi/post_warmup_short_steps"], 5000.0)

    def test_computed_high_model_trust_and_healthy_critic_use_full_horizon(self):
        trust = _compute_hgi_trust_metrics(self._healthy_metrics())
        gate = _compute_hgi_imagination_gate(trust, base_horizon=15, post_warmup_steps=5000)

        self.assertGreater(trust["hgi/model_trust"], 0.9)
        self.assertEqual(trust["hgi/critic_health"], 1.0)
        self.assertEqual(trust["hgi/actor_health_cache_used"], 0.0)
        self.assertEqual(trust["hgi/actor_health_fresh_present"], 1.0)
        self.assertEqual(gate["hgi/effective_horizon"], 15.0)
        self.assertEqual(gate["hgi/gate_state"], 3.0)

    def test_kl_band_trust_treats_free_nats_latent_as_healthy(self):
        trust = _compute_hgi_trust_metrics(
            self._healthy_metrics(
                **{
                    "wm/kl_mean": 0.6,
                    "wm_kl_clamped": 1.0,
                    "wm/post_prior_mu_abs_diff": 0.05,
                    "wm_val/post_advantage_ratio": 0.06,
                }
            )
        )

        self.assertEqual(trust["hgi/model_trust_kl"], 1.0)
        self.assertGreater(trust["hgi/model_trust"], 0.9)

    def test_latent_ablation_failure_drops_model_trust(self):
        trust = _compute_hgi_trust_metrics(
            self._healthy_metrics(
                **{
                    "wm/post_prior_mu_abs_diff": 0.03,
                    "wm_val/post_advantage_ratio": -0.01,
                }
            )
        )

        self.assertEqual(trust["hgi/latent_advantage_health"], 0.0)
        self.assertEqual(trust["hgi/model_trust_latent_alive"], 0.0)
        self.assertEqual(trust["hgi/model_trust"], 0.0)

    def test_cached_actor_health_prevents_critic_only_hgi_collapse(self):
        fresh_metrics = self._healthy_metrics()
        actor_health_cache = {
            key: fresh_metrics[key]
            for key in (
                "bridge/dqda_norm",
                "bridge/q_pi_std",
                "guard/grad_health_mean",
                "guard/tier",
            )
        }
        critic_only_metrics = dict(fresh_metrics)
        for key in actor_health_cache:
            critic_only_metrics.pop(key)

        trust = _compute_hgi_trust_metrics(
            critic_only_metrics,
            actor_health_cache=actor_health_cache,
        )

        self.assertEqual(trust["hgi/actor_health_cache_used"], 1.0)
        self.assertEqual(trust["hgi/actor_health_fresh_present"], 0.0)
        self.assertEqual(trust["hgi/critic_health_dqda"], 1.0)
        self.assertEqual(trust["hgi/critic_health_q_pi_std"], 1.0)
        self.assertEqual(trust["hgi/critic_health"], 1.0)

    def test_q_overconfidence_gap_reduces_hgi_critic_health(self):
        trust = _compute_hgi_trust_metrics(
            self._healthy_metrics(
                **{
                    "bridge/q_real_mean": 0.0,
                    "bridge/q_real_std": 0.05,
                    "bridge/q_pi_mean": 0.2,
                    "bridge/q_pi_std": 0.05,
                }
            ),
            {"ADAPTIVE_Q_GAP_SOFT": 0.5, "ADAPTIVE_Q_GAP_HARD": 1.25},
        )

        self.assertEqual(trust["hgi/critic_health_q_overconfidence"], 0.0)
        self.assertEqual(trust["hgi/critic_health"], 0.0)

    def test_zero_length_data_reduces_hgi_liveness_health(self):
        trust = _compute_hgi_trust_metrics(
            self._healthy_metrics(
                **{
                    "episode_length_train": 0.0,
                    "episode_length_test_det": 0.0,
                    "episode_length_test_stoch": 0.0,
                }
            )
        )

        self.assertEqual(trust["hgi/critic_health_data_liveness"], 0.0)
        self.assertEqual(trust["hgi/critic_health"], 0.0)

    def test_medium_trust_uses_short_horizon(self):
        gate = _compute_hgi_imagination_gate(
            {"hgi/model_trust": 0.7, "hgi/critic_health": 0.7, "hgi/imag_trust": 0.49},
            base_horizon=15,
        )

        self.assertEqual(gate["hgi/effective_horizon"], 3.0)
        self.assertEqual(gate["hgi/skipped_ratio"], 0.0)
        self.assertEqual(gate["hgi/gate_state"], 2.0)

    def test_low_model_trust_skips_imagination(self):
        gate = _compute_hgi_imagination_gate(
            {"hgi/model_trust": 0.2, "hgi/critic_health": 0.9, "hgi/imag_trust": 0.18},
            base_horizon=15,
        )

        self.assertEqual(gate["hgi/effective_horizon"], 0.0)
        self.assertEqual(gate["hgi/skipped_ratio"], 1.0)
        self.assertEqual(gate["hgi/skip_reason_model"], 1.0)

    def test_computed_low_model_trust_skips_imagination(self):
        trust = _compute_hgi_trust_metrics(
            self._healthy_metrics(
                **{
                    "wm/kl_mean": 0.5,
                    "wm/reward_error_abs_mean": 2.0,
                    "wm/reward_target_std": 0.05,
                }
            )
        )
        gate = _compute_hgi_imagination_gate(trust, base_horizon=15)

        self.assertLess(trust["hgi/model_trust"], 0.35)
        self.assertEqual(gate["hgi/effective_horizon"], 0.0)
        self.assertEqual(gate["hgi/skip_reason_model"], 1.0)
        self.assertEqual(gate["hgi/ramp_active"], 0.0)

    def test_low_critic_health_skips_imagination(self):
        gate = _compute_hgi_imagination_gate(
            {"hgi/model_trust": 0.9, "hgi/critic_health": 0.2, "hgi/imag_trust": 0.18},
            base_horizon=15,
        )

        self.assertEqual(gate["hgi/effective_horizon"], 0.0)
        self.assertEqual(gate["hgi/skipped_ratio"], 1.0)
        self.assertEqual(gate["hgi/skip_reason_critic"], 1.0)
        self.assertEqual(gate["hgi/ramp_active"], 0.0)

    def test_computed_flat_critic_skips_imagination(self):
        trust = _compute_hgi_trust_metrics(
            self._healthy_metrics(
                **{
                    "bridge/q_pi_std": 0.005,
                    "bridge/q_action_sensitivity": 0.001,
                }
            )
        )
        gate = _compute_hgi_imagination_gate(trust, base_horizon=15)

        self.assertLess(trust["hgi/critic_health"], 0.35)
        self.assertEqual(gate["hgi/effective_horizon"], 0.0)
        self.assertEqual(gate["hgi/skip_reason_critic"], 1.0)


if __name__ == "__main__":
    unittest.main()
