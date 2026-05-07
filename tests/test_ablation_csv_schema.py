import csv
import json
import pandas as pd
import numpy as np
import os
import tempfile
import unittest
from gymnasium.spaces import Box

from tmrl.training_offline import STABLE_CSV_COLUMNS, MetricsWriter, TrainingOffline


class DummyEnv:
    def __enter__(self):
        self.observation_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class DummyMemory:
    def __init__(self, nb_steps, device):
        self.nb_steps = nb_steps
        self.device = device

    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())


class DummyAgent:
    def __init__(self, observation_space, action_space, device):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device


def _append_metrics_row(training_obj, epoch, rnd, return_train, loss_critic):
    row_data = pd.Series({"return_train": return_train, "loss_critic": loss_critic})
    training_obj._metrics_writer.write(row_data, epoch=epoch, rnd=rnd, wall_clock_seconds=1.0)


class TestAblationCsvSchema(unittest.TestCase):
    def test_ablation_csv_parseable_after_resume(self):
        previous_ablation_dir = os.environ.get("TMRL_ABLATION_DIR")
        previous_run_name = os.environ.get("TMRL_RUN_NAME")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                os.environ["TMRL_ABLATION_DIR"] = tmp_dir
                os.environ["TMRL_RUN_NAME"] = "csv_resume_test"

                first = TrainingOffline(env_cls=DummyEnv, memory_cls=DummyMemory, training_agent_cls=DummyAgent)
                _append_metrics_row(first, epoch=0, rnd=0, return_train=1.0, loss_critic=0.5)
                first._metrics_writer.close()

                second = TrainingOffline(env_cls=DummyEnv, memory_cls=DummyMemory, training_agent_cls=DummyAgent)
                self.assertTrue(second._csv_header_written)
                _append_metrics_row(second, epoch=0, rnd=1, return_train=1.2, loss_critic=0.4)
                second._metrics_writer.close()

                csv_path = os.path.join(tmp_dir, "csv_resume_test.stable.csv")
                df = pd.read_csv(csv_path)
                self.assertEqual(len(df), 2)
                self.assertTrue({"wall_clock_seconds", "epoch", "round", "return_train", "loss_critic"}.issubset(df.columns))
                pd.to_numeric(df["wall_clock_seconds"], errors="raise")
                pd.to_numeric(df["return_train"], errors="raise")
                pd.to_numeric(df["loss_critic"], errors="raise")
        finally:
            if previous_ablation_dir is None:
                os.environ.pop("TMRL_ABLATION_DIR", None)
            else:
                os.environ["TMRL_ABLATION_DIR"] = previous_ablation_dir

            if previous_run_name is None:
                os.environ.pop("TMRL_RUN_NAME", None)
            else:
                os.environ["TMRL_RUN_NAME"] = previous_run_name

    def test_stable_csv_ignores_dynamic_metrics_but_jsonl_preserves_them(self):
        def fake_metrics(width, dynamic_key, return_train):
            metrics = {f"metric_{idx}": float(idx) for idx in range(width - 5)}
            metrics.update(
                {
                    "memory_len": width,
                    "return_train": return_train,
                    "loss_critic": 0.1,
                    "wm_imagined_steps": 15,
                    dynamic_key: 123.0,
                }
            )
            self.assertEqual(len(metrics), width)
            return metrics

        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = MetricsWriter(ablation_dir=tmp_dir, run_name="dynamic_schema_test", start_time=0.0)
            writer.write(
                fake_metrics(298, "wm_imag/h0_state_mean", 1.0),
                epoch=0,
                rnd=0,
                wall_clock_seconds=1.0,
            )
            writer.write(
                fake_metrics(509, "wm_imag/h7_reward_std", 2.0),
                epoch=0,
                rnd=1,
                wall_clock_seconds=2.0,
            )
            writer.write(
                fake_metrics(941, "wm_imag/h14_uncertainty_max", 3.0),
                epoch=0,
                rnd=2,
                wall_clock_seconds=3.0,
            )
            writer.close()

            with open(os.path.join(tmp_dir, "dynamic_schema_test.stable.csv"), newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], list(STABLE_CSV_COLUMNS))
            self.assertTrue(all(len(row) == len(rows[0]) for row in rows))
            self.assertEqual(len(rows), 4)
            self.assertNotIn("wm_imag/h14_uncertainty_max", rows[0])

            with open(os.path.join(tmp_dir, "dynamic_schema_test.metrics.jsonl"), encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
            self.assertEqual([record["metrics/schema_key_count"] for record in records], [298, 509, 941])
            self.assertIn("wm_imag/h0_state_mean", records[0])
            self.assertIn("wm_imag/h7_reward_std", records[1])
            self.assertIn("wm_imag/h14_uncertainty_max", records[2])

