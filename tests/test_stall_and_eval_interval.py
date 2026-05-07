import unittest
from pathlib import Path

import tmrl.config.config_constants as cfg
from tmrl.training_offline import (
    _update_hgi_eval_gap_metrics,
    compute_best_checkpoint_decision,
    compute_hgi_det_stoch_gap_health,
    compute_stall_dashboard,
    compute_stall_state,
)


class TestStallMetrics(unittest.TestCase):
    def test_stall_detector_flags_flat_eval_series(self):
        eval_returns = [10.0] * 25
        train_returns = [12.0] * 25
        state = compute_stall_state(eval_returns, train_returns, best_eval_ma10=None, stall_epochs=0)
        self.assertTrue(state["stalled"])
        self.assertGreaterEqual(state["stall_epochs"], 10)

    def test_stall_detector_keeps_improving_series_unstalled(self):
        eval_returns = [1.0 * (1.2 ** i) for i in range(20)]
        train_returns = [2.0 * (1.2 ** i) for i in range(20)]
        best = None
        stall_epochs = 0
        state = None
        for i in range(len(eval_returns)):
            state = compute_stall_state(
                eval_returns[: i + 1],
                train_returns[: i + 1],
                best_eval_ma10=best,
                stall_epochs=stall_epochs,
            )
            best = state["best_eval_ma10"]
            stall_epochs = state["stall_epochs"]
        self.assertIsNotNone(state)
        self.assertFalse(state["stalled"])

    def test_dashboard_status_levels(self):
        healthy = compute_stall_dashboard({"stall_epochs": 2, "patience": 10}, warning_epochs=5)
        warning = compute_stall_dashboard({"stall_epochs": 6, "patience": 10}, warning_epochs=5)
        stalled = compute_stall_dashboard({"stall_epochs": 10, "patience": 10}, warning_epochs=5)
        self.assertEqual(healthy["status"], "healthy")
        self.assertEqual(warning["status"], "warning")
        self.assertEqual(stalled["status"], "stalled")

    def test_hgi_gap_health_flags_dead_deterministic_policy(self):
        health = compute_hgi_det_stoch_gap_health(return_det=0.05, return_stoch=16.0, gap_warn=10.0)
        self.assertLess(health, 0.01)

    def test_hgi_gap_health_reduces_critic_health_and_imag_trust(self):
        metrics = {
            "return_test_det": 0.05,
            "return_test_stoch": 16.0,
            "hgi/model_trust": 0.9,
            "hgi/critic_health": 0.9,
            "hgi/imag_trust": 0.81,
        }

        _update_hgi_eval_gap_metrics(metrics)

        self.assertEqual(metrics["hgi/critic_health_pre_eval"], 0.9)
        self.assertLess(metrics["hgi/det_stoch_gap_health"], 0.01)
        self.assertEqual(metrics["hgi/critic_health"], metrics["hgi/det_stoch_gap_health"])
        self.assertLess(metrics["hgi/imag_trust"], 0.01)

    def test_hgi_gap_health_accepts_tracking_deterministic_policy(self):
        health = compute_hgi_det_stoch_gap_health(return_det=9.5, return_stoch=9.7, gap_warn=10.0)
        self.assertGreater(health, 0.9)


class TestEvalIntervalWiring(unittest.TestCase):
    def test_test_episode_interval_constant_exists(self):
        self.assertTrue(hasattr(cfg, "TEST_EPISODE_INTERVAL"))
        self.assertGreaterEqual(int(cfg.TEST_EPISODE_INTERVAL), 0)

    def test_worker_run_uses_configured_test_episode_interval(self):
        source = Path("tmrl/__main__.py").read_text(encoding="utf-8")
        self.assertIn("rw.run(test_episode_interval=cfg.TEST_EPISODE_INTERVAL)", source)


class TestBestCheckpointDecision(unittest.TestCase):
    def test_below_threshold_does_not_save(self):
        decision = compute_best_checkpoint_decision(
            metrics={"return_test_det": 49.9, "episode_length_test_det": 1000},
            best_state={"best_metric_value": float("-inf"), "best_tie_breaker_value": float("-inf")},
            alg_cfg={"BEST_CHECKPOINT_MIN_RETURN": 50.0},
        )

        self.assertFalse(decision["triggered"])

    def test_first_high_deterministic_return_saves(self):
        decision = compute_best_checkpoint_decision(
            metrics={"return_test_det": 82.6, "episode_length_test_det": 960},
            best_state={"best_metric_value": float("-inf"), "best_tie_breaker_value": float("-inf")},
            alg_cfg={"BEST_CHECKPOINT_MIN_RETURN": 50.0},
        )

        self.assertTrue(decision["triggered"])

    def test_lower_later_return_does_not_save(self):
        decision = compute_best_checkpoint_decision(
            metrics={"return_test_det": 71.0, "episode_length_test_det": 1000},
            best_state={"best_metric_value": 82.6, "best_tie_breaker_value": 960},
            alg_cfg={"BEST_CHECKPOINT_MIN_RETURN": 50.0},
        )

        self.assertFalse(decision["triggered"])

    def test_equal_return_with_longer_deterministic_episode_saves(self):
        decision = compute_best_checkpoint_decision(
            metrics={"return_test_det": 82.6, "episode_length_test_det": 1000},
            best_state={"best_metric_value": 82.6, "best_tie_breaker_value": 960},
            alg_cfg={"BEST_CHECKPOINT_MIN_RETURN": 50.0},
        )

        self.assertTrue(decision["triggered"])

    def test_disabled_config_never_saves(self):
        decision = compute_best_checkpoint_decision(
            metrics={"return_test_det": 100.0, "episode_length_test_det": 1000},
            best_state={"best_metric_value": float("-inf"), "best_tie_breaker_value": float("-inf")},
            alg_cfg={"BEST_CHECKPOINT_ENABLED": False, "BEST_CHECKPOINT_MIN_RETURN": 50.0},
        )

        self.assertFalse(decision["triggered"])
