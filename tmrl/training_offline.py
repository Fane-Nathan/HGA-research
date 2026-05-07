# standard library imports
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

# third-party imports
import torch
from pandas import DataFrame

# local imports
import tmrl.config.config_constants as cfg
from tmrl.util import dump, pandas_dict

import logging


__docformat__ = "google"


def _ma_last(values, window=10):
    if not values:
        return 0.0
    tail = values[-window:]
    return float(sum(tail) / len(tail))


def compute_stall_state(eval_returns,
                        train_returns,
                        best_eval_ma10=None,
                        stall_epochs=0,
                        improvement_threshold=0.05,
                        patience=10):
    """
    Computes rolling anti-stall diagnostics from epoch-level eval/train returns.

    Args:
        eval_returns (List[float]): epoch-level evaluation returns.
        train_returns (List[float]): epoch-level training returns.
        best_eval_ma10 (float or None): best rolling eval MA10 observed so far.
        stall_epochs (int): number of consecutive epochs without meaningful eval MA10 improvement.
        improvement_threshold (float): relative improvement threshold (e.g. 0.05 = 5%).
        patience (int): epochs without improvement before flagged as stalled.
    """
    eval_hist = []
    train_hist = []
    eval_ma10 = 0.0
    train_ma10 = 0.0
    best = best_eval_ma10
    stall = int(stall_epochs)

    for eval_ret, train_ret in zip(eval_returns, train_returns):
        eval_hist.append(float(eval_ret))
        train_hist.append(float(train_ret))
        eval_ma10 = _ma_last(eval_hist, window=10)
        train_ma10 = _ma_last(train_hist, window=10)

        if best is None:
            best = eval_ma10
            stall = 0
            continue

        if best > 0.0:
            improved = eval_ma10 >= best * (1.0 + improvement_threshold)
        else:
            improved = eval_ma10 > best + 1e-9

        if improved:
            best = eval_ma10
            stall = 0
        else:
            stall += 1

    return {
        "eval_return_ma10": float(eval_ma10),
        "train_return_ma10": float(train_ma10),
        "best_eval_ma10": float(best if best is not None else 0.0),
        "stall_epochs": int(stall),
        "stalled": bool(stall >= patience),
        "improvement_threshold": float(improvement_threshold),
        "patience": int(patience),
    }


def compute_stall_dashboard(stall_state, warning_epochs=5):
    """
    Returns a compact dashboard state from stall metrics.
    """
    stall_epochs = int(stall_state.get("stall_epochs", 0))
    patience = int(stall_state.get("patience", 10))
    if stall_epochs >= patience:
        status = "stalled"
    elif stall_epochs >= warning_epochs:
        status = "warning"
    else:
        status = "healthy"
    return {
        "status": status,
        "warning_epochs": int(warning_epochs),
    }


def compute_hgi_det_stoch_gap_health(return_det, return_stoch, gap_warn=10.0):
    """
    Scores whether deterministic eval is tracking stochastic eval.
    """
    det = float(return_det)
    stoch = float(return_stoch)
    if stoch <= 0.0:
        return 1.0
    positive_gap = max(0.0, stoch - det)
    abs_gap_health = max(0.0, min(1.0, 1.0 - positive_gap / max(float(gap_warn), 1e-12)))
    ratio_health = max(0.0, min(1.0, det / max(stoch, 1e-12)))
    return min(abs_gap_health, ratio_health) if positive_gap > 0.0 else 1.0


def _safe_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(result):
        return float(default)
    return result


def _safe_bool(value, default=True):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def compute_best_checkpoint_decision(metrics, best_state=None, alg_cfg=None):
    """
    Decides whether current eval metrics should preserve a best-eval checkpoint.
    """
    alg_cfg = alg_cfg or {}
    best_state = best_state or {}

    enabled = _safe_bool(alg_cfg.get("BEST_CHECKPOINT_ENABLED", True), True)
    metric_name = str(alg_cfg.get("BEST_CHECKPOINT_METRIC", "return_test_det"))
    min_return = _safe_float(alg_cfg.get("BEST_CHECKPOINT_MIN_RETURN", 50.0), 50.0)
    tie_breaker_name = str(alg_cfg.get("BEST_CHECKPOINT_TIE_BREAKER", "episode_length_test_det"))

    metric_value = _safe_float(metrics.get(metric_name), 0.0)
    tie_breaker_value = _safe_float(metrics.get(tie_breaker_name), 0.0)
    best_metric_value = _safe_float(best_state.get("best_metric_value"), float("-inf"))
    best_tie_breaker_value = _safe_float(best_state.get("best_tie_breaker_value"), float("-inf"))

    passes_threshold = metric_value >= min_return
    improves_metric = metric_value > best_metric_value
    improves_tie_breaker = metric_value == best_metric_value and tie_breaker_value > best_tie_breaker_value
    triggered = bool(enabled and passes_threshold and (improves_metric or improves_tie_breaker))

    return {
        "triggered": triggered,
        "enabled": enabled,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "min_return": min_return,
        "tie_breaker_name": tie_breaker_name,
        "tie_breaker_value": tie_breaker_value,
        "best_metric_value": best_metric_value,
        "best_tie_breaker_value": best_tie_breaker_value,
    }


def _update_hgi_eval_gap_metrics(metrics):
    alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
    gap_warn = float(alg_cfg.get("HGI_DET_STOCH_GAP_WARN", 10.0))
    gap_health = compute_hgi_det_stoch_gap_health(
        metrics.get("return_test_det", 0.0),
        metrics.get("return_test_stoch", 0.0),
        gap_warn=gap_warn,
    )
    metrics["hgi/det_stoch_gap_health"] = gap_health
    if "hgi/critic_health" in metrics:
        critic_health_pre_eval = float(metrics.get("hgi/critic_health", 1.0))
        metrics["hgi/critic_health_pre_eval"] = critic_health_pre_eval
        metrics["hgi/critic_health"] = min(critic_health_pre_eval, gap_health)
        if "hgi/model_trust" in metrics:
            model_trust = float(metrics.get("hgi/model_trust", 0.0))
            metrics["hgi/imag_trust"] = max(0.0, min(1.0, model_trust * metrics["hgi/critic_health"]))


def _format_metric_value(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)

    if math.isnan(val) or math.isinf(val):
        return str(val)
    if abs(val) >= 10000.0 or (0.0 < abs(val) < 0.0001):
        return f"{val:.4e}"
    return f"{val:.6f}".rstrip("0").rstrip(".")


STABLE_CSV_COLUMNS = (
    "wall_clock_seconds",
    "epoch",
    "round",
    "memory_len",
    "round_time",
    "return_test",
    "return_test_det",
    "return_test_stoch",
    "return_test_det_stoch_gap",
    "return_train",
    "episode_length_test",
    "episode_length_test_det",
    "episode_length_test_stoch",
    "episode_length_train",
    "best_checkpoint/triggered",
    "best_checkpoint/best_return_test_det",
    "best_checkpoint/best_epoch",
    "best_checkpoint/best_round",
    "best_checkpoint/save_failed",
    "debug_alpha_steer",
    "debug_alpha_gas",
    "debug_alpha_brake",
    "debug_alpha_floor_steer",
    "debug_alpha_floor_gas",
    "debug_alpha_floor_brake",
    "debug_log_std_mean",
    "debug_log_std_min",
    "loss_entropy_coef",
    "entropy_coef",
    "skill_transfer/stoch_adv_drive_ema",
    "skill_transfer/det_adv_drive_ema",
    "skill_transfer/det_lambda_eff",
    "skill_transfer/min_awdb_weight_eff",
    "skill_transfer/min_awdb_weight_target",
    "skill_transfer/min_awdb_weight_applied",
    "skill_transfer/loss_log_std_ceiling_weighted",
    "skill_transfer/log_std_above_ceiling",
    "loss_critic",
    "bridge/dqda_norm",
    "bridge/q_pi_std",
    "bridge/q_action_sensitivity",
    "hgi/model_trust",
    "hgi/critic_health",
    "hgi/imag_trust",
    "wm_kl",
    "wm/reward_error_abs_mean",
    "wm_imagined_steps",
)


def _json_safe_metric(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            value = value.item()
        else:
            return _json_safe_metric(value.tolist())

    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, dict):
        return {str(k): _json_safe_metric(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metric(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


def _csv_safe_metric(value):
    if value is None:
        return ""
    value = _json_safe_metric(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    return str(value)


class MetricsWriter:
    """
    Writes full-fidelity metrics to JSONL and a fixed-width dashboard CSV.
    """

    def __init__(self, ablation_dir, run_name, start_time=None):
        self.ablation_dir = Path(ablation_dir)
        self.ablation_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = str(run_name)
        self.stable_path = self.ablation_dir / f"{self.run_name}.stable.csv"
        self.jsonl_path = self.ablation_dir / f"{self.run_name}.metrics.jsonl"
        self.start_time = time.time() if start_time is None else float(start_time)
        self._seen_keys = set()

        self.stable_header_written = self.stable_path.exists() and self.stable_path.stat().st_size > 0
        self.stable_file = open(self.stable_path, "a", newline="", encoding="utf-8")
        self.stable_writer = csv.writer(self.stable_file)
        if not self.stable_header_written:
            self.stable_writer.writerow(STABLE_CSV_COLUMNS)
            self.stable_file.flush()
            self.stable_header_written = True

        self.jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

    @staticmethod
    def _metrics_to_dict(metrics):
        if hasattr(metrics, "to_dict"):
            metrics = metrics.to_dict()
        return {str(key): value for key, value in dict(metrics).items()}

    def write(self, metrics, epoch, rnd, wall_clock_seconds=None):
        raw_metrics = self._metrics_to_dict(metrics)
        metric_keys = set(raw_metrics)
        new_keys = sorted(metric_keys - self._seen_keys)
        self._seen_keys.update(metric_keys)

        elapsed = time.time() - self.start_time if wall_clock_seconds is None else float(wall_clock_seconds)
        base_metrics = {
            "wall_clock_seconds": elapsed,
            "epoch": int(epoch),
            "round": int(rnd),
        }

        record = {key: _json_safe_metric(value) for key, value in raw_metrics.items()}
        record.update(base_metrics)
        record["metrics/schema_key_count"] = len(metric_keys)
        record["metrics/new_key_count"] = len(new_keys)
        record["metrics/new_keys_sample"] = new_keys[:20]
        self.jsonl_file.write(json.dumps(record, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
        self.jsonl_file.flush()

        stable_source = dict(raw_metrics)
        stable_source.update(base_metrics)
        self.stable_writer.writerow([_csv_safe_metric(stable_source.get(column, "")) for column in STABLE_CSV_COLUMNS])
        self.stable_file.flush()

    def close(self):
        for handle_name in ("stable_file", "jsonl_file"):
            handle = getattr(self, handle_name, None)
            if handle is not None and not handle.closed:
                handle.close()


def _metric_matches(key, exact=(), prefixes=()):
    return key in exact or any(key.startswith(prefix) for prefix in prefixes)


def _format_sectioned_stats(stats_series):
    sections = [
        (
            "RUN / RETURNS / TIMING",
            {
                "exact": ("memory_len", "round_time", "idle_time", "update_buf_time", "train_time"),
                "prefixes": ("return_", "episode_length_", "sampling_duration", "training_step_duration"),
            },
        ),
        (
            "ACTOR / ENTROPY",
            {
                "exact": ("loss_actor", "loss_entropy_coef", "entropy_coef", "loss_awdb", "loss_sbr"),
                "prefixes": ("debug_alpha_", "debug_log_std_", "grad_norm_actor", "grad_health_", "ghae_boost_"),
            },
        ),
        (
            "CRITIC",
            {
                "exact": ("loss_critic", "grad_norm_critic", "kl_div_loss", "ewc_loss"),
                "prefixes": (),
            },
        ),
        (
            "TRUST / UNCERTAINTY",
            {
                "exact": (),
                "prefixes": ("hgi/", "verifier_", "wm/verifier_", "bridge/trust"),
            },
        ),
        (
            "IMAGINATION",
            {
                "exact": ("imag_actor_loss", "wm_imagined_steps", "wm_imagined_q_loss"),
                "prefixes": ("wm_imag/",),
            },
        ),
        (
            "WORLD MODEL",
            {
                "exact": ("dynamics_loss", "wm_train_steps", "wm_kl", "wm_recon_state"),
                "prefixes": ("wm/", "wm_input/", "wm_grad/", "curiosity_"),
            },
        ),
        (
            "BRIDGE",
            {
                "exact": (),
                "prefixes": ("bridge/",),
            },
        ),
        (
            "STABILITY GUARDS",
            {
                "exact": (),
                "prefixes": ("guard/",),
            },
        ),
    ]

    items = [(str(key), value) for key, value in stats_series.items()]
    used = set()
    output = []

    for title, matcher in sections:
        rows = []
        for key, value in items:
            if key in used:
                continue
            if _metric_matches(key, exact=matcher["exact"], prefixes=matcher["prefixes"]):
                rows.append((key, value))
                used.add(key)
        if rows:
            output.append(f"========== {title} ==========")
            width = min(max(len(key) for key, _ in rows), 64)
            for key, value in rows:
                output.append(f"  {key:<{width}} : {_format_metric_value(value)}")

    other_rows = [(key, value) for key, value in items if key not in used]
    if other_rows:
        output.append("========== OTHER ==========")
        width = min(max(len(key) for key, _ in other_rows), 64)
        for key, value in other_rows:
            output.append(f"  {key:<{width}} : {_format_metric_value(value)}")

    return "\n".join(output) + "\n"


@dataclass(eq=0)
class TrainingOffline:
    """
    Training wrapper for off-policy algorithms.

    Args:
        env_cls (type): class of a dummy environment, used only to retrieve observation and action spaces if needed. Alternatively, this can be a tuple of the form (observation_space, action_space).
        memory_cls (type): class of the replay memory
        training_agent_cls (type): class of the training agent
        epochs (int): total number of epochs, we save the agent every epoch
        rounds (int): number of rounds per epoch, we generate statistics every round
        steps (int): number of training steps per round
        update_model_interval (int): number of training steps between model broadcasts
        update_buffer_interval (int): number of training steps between retrieving buffered samples
        max_training_steps_per_env_step (float): training will pause when above this ratio
        sleep_between_buffer_retrieval_attempts (float): algorithm will sleep for this amount of time when waiting for needed incoming samples
        profiling (bool): if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
        agent_scheduler (callable): if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
        start_training (int): minimum number of samples in the replay buffer before starting training
        device (str): device on which the memory will collate training samples
    """
    env_cls: type = None  # = GenericGymEnv  # dummy environment, used only to retrieve observation and action spaces if needed
    memory_cls: type = None  # = TorchMemory  # replay memory
    training_agent_cls: type = None  # = TrainingAgent  # training agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2000  # number of training steps per round
    update_model_interval: int = 100  # number of training steps between model broadcasts
    update_buffer_interval: int = 100  # number of training steps between retrieving buffered samples
    max_training_steps_per_env_step: float = 1.0  # training will pause when above this ratio
    sleep_between_buffer_retrieval_attempts: float = 1.0  # algorithm will sleep for this amount of time when waiting for needed incoming samples
    profiling: bool = False  # if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
    agent_scheduler: callable = None  # if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
    start_training: int = 0  # minimum number of samples in the replay buffer before starting training
    device: str = None  # device on which the model of the TrainingAgent will live

    total_updates = 0

    def __post_init__(self):
        device = self.device
        self.epoch = 0
        self.memory = self.memory_cls(nb_steps=self.steps, device=device)
        if type(self.env_cls) == tuple:
            observation_space, action_space = self.env_cls
        else:
            with self.env_cls() as env:
                observation_space, action_space = env.observation_space, env.action_space
        self.agent = self.training_agent_cls(observation_space=observation_space,
                                             action_space=action_space,
                                             device=device)
        self.total_samples = len(self.memory)
        logging.info(f" Initial total_samples:{self.total_samples}")

        # === Metrics logger for ablation study ===
        self._init_csv_logger()
        self._ensure_stall_tracking_state()
        self._ensure_best_checkpoint_state()

    def _resolve_run_name(self):
        """Resolve run name from env override, then config.json, then fallback."""
        env_name = os.environ.get("TMRL_RUN_NAME")
        if env_name:
            return env_name

        config_file = Path.home() / "TmrlData" / "config" / "config.json"
        if config_file.exists():
            try:
                import json
                with open(config_file, encoding="utf-8") as f:
                    config = json.load(f)
                cfg_name = config.get("RUN_NAME")
                if cfg_name:
                    return cfg_name
            except Exception:
                pass
        return "default"

    def _init_csv_logger(self):
        """Initialize metrics logger (called from __post_init__ and __setstate__)."""
        self._metrics_writer = None
        self._training_start_time = time.time()
        ablation_dir = Path(os.environ.get("TMRL_ABLATION_DIR", Path.home() / "TmrlData" / "ablation"))
        ablation_dir.mkdir(parents=True, exist_ok=True)
        run_name = self._resolve_run_name()
        self._metrics_writer = MetricsWriter(
            ablation_dir=ablation_dir,
            run_name=run_name,
            start_time=self._training_start_time,
        )
        self._csv_path = self._metrics_writer.stable_path
        self._jsonl_metrics_path = self._metrics_writer.jsonl_path
        self._csv_header_written = self._metrics_writer.stable_header_written
        self._stall_diag_path = ablation_dir / f"{run_name}_stall.json"
        logging.info(f" stable CSV logging to: {self._csv_path}")
        logging.info(f" full metrics JSONL logging to: {self._jsonl_metrics_path}")

    def _ensure_stall_tracking_state(self):
        alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
        # Backward-compatible defaults for older checkpoints.
        if not hasattr(self, "_eval_epoch_returns"):
            self._eval_epoch_returns = []
        if not hasattr(self, "_train_epoch_returns"):
            self._train_epoch_returns = []
        if not hasattr(self, "_best_eval_ma10"):
            self._best_eval_ma10 = None
        if not hasattr(self, "_stall_epochs"):
            self._stall_epochs = 0
        if not hasattr(self, "_stall_improvement_threshold"):
            self._stall_improvement_threshold = float(alg_cfg.get("STALL_IMPROVEMENT_THRESHOLD", 0.05))
        if not hasattr(self, "_stall_patience"):
            self._stall_patience = int(alg_cfg.get("STALL_PATIENCE", 10))
        if not hasattr(self, "_stall_warning_epochs"):
            self._stall_warning_epochs = int(alg_cfg.get("STALL_WARNING_EPOCHS", 5))
        if not hasattr(self, "_stall_last_auto_action"):
            self._stall_last_auto_action = "none"
        if not hasattr(self, "_stall_last_action_epoch"):
            self._stall_last_action_epoch = -1

    def _ensure_best_checkpoint_state(self):
        # Backward-compatible defaults for older trainer checkpoints.
        if not hasattr(self, "best_return_test_det"):
            self.best_return_test_det = 0.0
        if not hasattr(self, "best_return_test_stoch"):
            self.best_return_test_stoch = 0.0
        if not hasattr(self, "best_return_test"):
            self.best_return_test = 0.0
        if not hasattr(self, "best_eval_epoch"):
            self.best_eval_epoch = -1
        if not hasattr(self, "best_eval_round"):
            self.best_eval_round = -1
        if not hasattr(self, "_best_checkpoint_metric_value"):
            self._best_checkpoint_metric_value = float("-inf")
        if not hasattr(self, "_best_checkpoint_tie_breaker_value"):
            self._best_checkpoint_tie_breaker_value = float("-inf")
        if not hasattr(self, "_best_checkpoint_episode_length_test_det"):
            self._best_checkpoint_episode_length_test_det = 0.0

    def _best_checkpoint_path(self, alg_cfg):
        override = alg_cfg.get("BEST_CHECKPOINT_PATH")
        if override:
            return str(Path(os.path.expanduser(str(override))))

        checkpoint_path = Path(cfg.CHECKPOINT_PATH)
        stem = checkpoint_path.stem
        if stem.endswith("_t"):
            best_stem = stem[:-2] + "_best_eval_t"
        else:
            best_stem = stem + "_best_eval"
        return str(checkpoint_path.with_name(best_stem + checkpoint_path.suffix))

    def _best_checkpoint_state(self):
        return {
            "best_metric_value": self._best_checkpoint_metric_value,
            "best_tie_breaker_value": self._best_checkpoint_tie_breaker_value,
        }

    def _set_best_checkpoint_state(self, metrics, decision, rnd):
        self.best_return_test_det = _safe_float(metrics.get("return_test_det"), self.best_return_test_det)
        self.best_return_test_stoch = _safe_float(metrics.get("return_test_stoch"), self.best_return_test_stoch)
        self.best_return_test = _safe_float(metrics.get("return_test"), self.best_return_test)
        self.best_eval_epoch = int(self.epoch)
        self.best_eval_round = int(rnd)
        self._best_checkpoint_metric_value = decision["metric_value"]
        self._best_checkpoint_tie_breaker_value = decision["tie_breaker_value"]
        self._best_checkpoint_episode_length_test_det = _safe_float(
            metrics.get("episode_length_test_det"),
            self._best_checkpoint_episode_length_test_det,
        )

    def _best_checkpoint_log_metrics(self, triggered=0.0, save_failed=0.0):
        return {
            "best_checkpoint/triggered": float(triggered),
            "best_checkpoint/best_return_test_det": float(self.best_return_test_det),
            "best_checkpoint/best_epoch": int(self.best_eval_epoch),
            "best_checkpoint/best_round": int(self.best_eval_round),
            "best_checkpoint/save_failed": float(save_failed),
        }

    def _maybe_save_best_checkpoint(self, metrics, rnd):
        self._ensure_best_checkpoint_state()
        alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
        decision = compute_best_checkpoint_decision(
            metrics=metrics,
            best_state=self._best_checkpoint_state(),
            alg_cfg=alg_cfg,
        )

        if not decision["triggered"]:
            return self._best_checkpoint_log_metrics(triggered=0.0)

        previous_state = {
            "best_return_test_det": self.best_return_test_det,
            "best_return_test_stoch": self.best_return_test_stoch,
            "best_return_test": self.best_return_test,
            "best_eval_epoch": self.best_eval_epoch,
            "best_eval_round": self.best_eval_round,
            "_best_checkpoint_metric_value": self._best_checkpoint_metric_value,
            "_best_checkpoint_tie_breaker_value": self._best_checkpoint_tie_breaker_value,
            "_best_checkpoint_episode_length_test_det": self._best_checkpoint_episode_length_test_det,
        }

        self._set_best_checkpoint_state(metrics, decision, rnd)
        best_path = Path(self._best_checkpoint_path(alg_cfg))
        try:
            best_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(
                f" saving best eval checkpoint: metric={decision['metric_name']} "
                f"value={decision['metric_value']:.6f}, path={best_path}"
            )
            dump(self, best_path)
            logging.info(f" saved best eval checkpoint: {best_path}")
            return self._best_checkpoint_log_metrics(triggered=1.0)
        except Exception as exc:
            for key, value in previous_state.items():
                setattr(self, key, value)
            logging.exception(f" failed to save best eval checkpoint to {best_path}: {exc}")
            return self._best_checkpoint_log_metrics(triggered=0.0, save_failed=1.0)

    def _apply_auto_stall_recovery(self, stall_state):
        """
        Applies safe, bounded auto-recovery actions when stalled.
        """
        action_parts = []
        stall_epochs = int(stall_state.get("stall_epochs", 0))
        warning_epochs = int(getattr(self, "_stall_warning_epochs", 5))
        patience = int(stall_state.get("patience", 10))

        # Apply no mutation while healthy.
        if stall_epochs < warning_epochs:
            self._stall_last_auto_action = "none"
            return "none"

        # Avoid repeated mutations in the same epoch.
        if self._stall_last_action_epoch == self.epoch:
            return self._stall_last_auto_action

        # Escalate recovery strength when fully stalled.
        hard_stall = stall_epochs >= patience
        utd_step = 2 if hard_stall else 1
        entropy_step = 0.5 if hard_stall else 0.2

        # 1) Reduce UTD pressure.
        if hasattr(self.agent, "q_updates_per_policy_update"):
            old_utd = int(self.agent.q_updates_per_policy_update)
            new_utd = max(1, old_utd - utd_step)
            if new_utd != old_utd:
                self.agent.q_updates_per_policy_update = new_utd
                action_parts.append(f"utd:{old_utd}->{new_utd}")

        # 2) Boost Target Entropy organically instead of forcing an unnatural alpha_floor
        old_bump = getattr(self.agent, "stall_entropy_bump", 0.0)
        new_bump = min(2.0, old_bump + entropy_step)
        if hasattr(self, "agent"):
            self.agent.stall_entropy_bump = new_bump
            action_parts.append(f"entropy_bump:+{entropy_step:.1f}")

        mode = "stalled" if hard_stall else "warning"
        action = ",".join(action_parts) if action_parts else f"hold({mode},no_change)"
        self._stall_last_auto_action = action
        self._stall_last_action_epoch = self.epoch
        return action

    def _update_stall_diagnostics(self, epoch_stats):
        if not epoch_stats:
            return

        train_epoch_return = float(sum(float(s.get("return_train", 0.0)) for s in epoch_stats) / len(epoch_stats))
        self._train_epoch_returns.append(train_epoch_return)

        # If eval episodes are disabled, don't derive stall state from stale test stats.
        eval_enabled = int(getattr(cfg, "TEST_EPISODE_INTERVAL", 0)) > 0
        if not eval_enabled:
            train_ma10 = _ma_last(self._train_epoch_returns, window=10)
            self._stall_epochs = 0
            self._stall_last_auto_action = "disabled(eval_off)"
            payload = {
                "epoch": int(self.epoch),
                "eval_enabled": False,
                "eval_epoch_return": None,
                "train_epoch_return": train_epoch_return,
                "eval_return_ma10": None,
                "train_return_ma10": train_ma10,
                "best_eval_ma10": None,
                "stalled": False,
                "stall_epochs": 0,
                "improvement_threshold": self._stall_improvement_threshold,
                "patience": self._stall_patience,
                "status": "disabled",
                "auto_action": self._stall_last_auto_action,
                "updated_at_unix": time.time(),
            }
            with open(self._stall_diag_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            logging.info(
                f" Stall diagnostics: eval=disabled (TEST_EPISODE_INTERVAL=0), "
                f"train_return_ma10={train_ma10:.6f}"
            )
            return

        eval_mode = str(getattr(cfg, "TEST_EVAL_MODE", "dual")).lower()
        eval_epoch_return_det = float(sum(float(s.get("return_test_det", s.get("return_test", 0.0))) for s in epoch_stats) / len(epoch_stats))
        eval_epoch_return_stoch = float(sum(float(s.get("return_test_stoch", s.get("return_test", 0.0))) for s in epoch_stats) / len(epoch_stats))
        eval_det_stoch_gap = eval_epoch_return_stoch - eval_epoch_return_det
        if eval_mode == "deterministic":
            eval_epoch_return = eval_epoch_return_det
        elif eval_mode == "stochastic":
            eval_epoch_return = eval_epoch_return_stoch
        else:
            # In dual mode, deterministic exploitation is the primary health signal.
            # Stochastic success with dead deterministic return is actor starvation, not a healthy run.
            eval_epoch_return = eval_epoch_return_det
        self._eval_epoch_returns.append(eval_epoch_return)

        state = compute_stall_state(
            eval_returns=self._eval_epoch_returns,
            train_returns=self._train_epoch_returns,
            best_eval_ma10=None,
            stall_epochs=0,
            improvement_threshold=self._stall_improvement_threshold,
            patience=self._stall_patience,
        )
        self._best_eval_ma10 = state["best_eval_ma10"]
        self._stall_epochs = state["stall_epochs"]
        dashboard = compute_stall_dashboard(state, warning_epochs=self._stall_warning_epochs)
        auto_action = self._apply_auto_stall_recovery(state)

        payload = {
            "epoch": int(self.epoch),
            "eval_mode": eval_mode,
            "eval_epoch_return": eval_epoch_return,
            "eval_epoch_return_det": eval_epoch_return_det,
            "eval_epoch_return_stoch": eval_epoch_return_stoch,
            "eval_det_stoch_gap": eval_det_stoch_gap,
            "train_epoch_return": train_epoch_return,
            "eval_return_ma10": state["eval_return_ma10"],
            "train_return_ma10": state["train_return_ma10"],
            "best_eval_ma10": state["best_eval_ma10"],
            "stalled": state["stalled"],
            "stall_epochs": state["stall_epochs"],
            "improvement_threshold": state["improvement_threshold"],
            "patience": state["patience"],
            "status": dashboard["status"],
            "auto_action": auto_action,
            "updated_at_unix": time.time(),
        }

        with open(self._stall_diag_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        logging.info(
            f" Stall diagnostics: eval_mode={eval_mode}, eval_return_ma10={state['eval_return_ma10']:.6f}, "
            f"eval_det={eval_epoch_return_det:.6f}, eval_stoch={eval_epoch_return_stoch:.6f}, "
            f"det_stoch_gap={eval_det_stoch_gap:.6f}, "
            f"train_return_ma10={state['train_return_ma10']:.6f}, "
            f"stalled={state['stalled']}, stall_epochs={state['stall_epochs']}"
        )
        if eval_mode == "dual" and eval_epoch_return_stoch > 0.5 and eval_det_stoch_gap > max(0.5, 2.0 * max(eval_epoch_return_det, 0.0)):
            logging.warning(
                " Deterministic starvation warning: stochastic eval is healthy "
                f"({eval_epoch_return_stoch:.6f}) but deterministic eval is weak "
                f"({eval_epoch_return_det:.6f})."
            )
        logging.info(
            f" Anti-stall dashboard: status={dashboard['status']}, "
            f"auto_action={auto_action}, "
            f"eval_ma10={state['eval_return_ma10']:.6f}, "
            f"train_ma10={state['train_return_ma10']:.6f}, "
            f"stall_epochs={state['stall_epochs']}/{state['patience']}"
        )

    def _apply_success_annealing(self, epoch_stats):
        """Smoothly anneal Target Entropy based on continuous performance mapping."""
        if not epoch_stats or not hasattr(self.agent, "target_entropy"):
            return
        
        # Allow disabling the auto-annealer so GHAE manages entropy exclusively
        alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
        if not alg_cfg.get("AUTO_ANNEAL_ENABLED", True):
            return
            
        train_epoch_return = float(sum(float(s.get("return_train", 0.0)) for s in epoch_stats) / len(epoch_stats))
        
        # Read bounds from config.json so the user can tune without code changes.
        # Floor = TARGET_ENTROPY (high exploration), ceiling = floor - 3.0 (low entropy, exploitation)
        alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
        entropy_floor = float(alg_cfg.get("TARGET_ENTROPY", -0.9))  # exploration: loose entropy
        entropy_ceiling = float(alg_cfg.get("TARGET_ENTROPY_CEILING", entropy_floor - 3.0))  # exploitation: tight entropy
        anneal_score = float(alg_cfg.get("ANNEAL_SCORE_THRESHOLD", 65.0))  # return at which max exploitation is reached
        
        # Smoothly map train return (0 to anneal_score) to Target Entropy (floor to ceiling)
        normalized_progress = max(0.0, min(1.0, train_epoch_return / anneal_score))
        computed_target = entropy_floor + (normalized_progress * (entropy_ceiling - entropy_floor))
        
        # Apply anti-stall organic exploration bump
        stall_bump = getattr(self.agent, "stall_entropy_bump", 0.0)
        if stall_bump > 0.0:
            computed_target = min(entropy_floor + 0.5, computed_target + stall_bump)
            # Decay the bump naturally every epoch so it gradually returns to pure performance-based mapping
            self.agent.stall_entropy_bump = max(0.0, stall_bump - 0.1)
            
        current_tensor = self.agent.target_entropy
        dim_act = current_tensor.shape[0] if len(current_tensor.shape) > 0 else 3
        current_val = float(current_tensor.mean().item()) * dim_act  # total entropy across dims
            
        # We limit the maximum change per epoch to prevent extreme shocks if score jumps
        max_step = 0.05
        
        # Move current_val smoothly toward computed_target
        if current_val > computed_target:
            next_val = max(computed_target, current_val - max_step)
        elif current_val < computed_target:
            next_val = min(computed_target, current_val + max_step)
        else:
            next_val = current_val
            
        if abs(next_val - current_val) > 0.01:
            per_dim = next_val / dim_act
            self.agent.target_entropy = torch.full((dim_act,), per_dim, device=self.agent.device)
            logging.info(f" Auto-Anneal: Score [{train_epoch_return:.2f}] -> Smoothly shifted Target Entropy from {current_val:.2f} to {next_val:.3f} (floor={entropy_floor}, ceiling={entropy_ceiling})")
            
            # Smoothly tighten deterministic regularization based on the same progress
            # Map lambda from 0.01 (high exploration) to 0.05 (high exploitation)
            if hasattr(self.agent, "det_reg_lambda"):
                target_lambda = 0.01 + (normalized_progress * 0.04)
                self.agent.det_reg_lambda = target_lambda

    def __getstate__(self):
        """Exclude unpicklable metrics file handles from checkpoint."""
        state = self.__dict__.copy()
        state.pop('_metrics_writer', None)
        state.pop('_csv_file', None)
        state.pop('_csv_writer', None)
        state.pop('_csv_header_written', None)
        state.pop('_jsonl_metrics_path', None)
        state.pop('_training_start_time', None)
        return state

    def __setstate__(self, state):
        """Reinitialize metrics logger after loading checkpoint."""
        self.__dict__.update(state)
        self._init_csv_logger()
        self._ensure_stall_tracking_state()
        self._ensure_best_checkpoint_state()

    def update_buffer(self, interface):
        buffer = interface.retrieve_buffer()
        self.memory.append(buffer)
        self.total_samples += len(buffer)

    def check_ratio(self, interface):
        ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
            logging.info(f" Waiting for new samples")
            while ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                # wait for new samples
                self.update_buffer(interface)
                ratio = self.total_updates / self.total_samples if self.total_samples > 0.0 and self.total_samples >= self.start_training else -1.0
                if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)
            logging.info(f" Resuming training")

    def run_epoch(self, interface):
        stats = []
        state = None

        if self.agent_scheduler is not None:
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            logging.info(f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, '=') + f" round {rnd}/{self.rounds} ".ljust(50, '='))
            logging.debug(f"(Training): current memory size:{len(self.memory)}")

            stats_training = []

            t0 = time.time()
            self.check_ratio(interface)
            t1 = time.time()

            if self.profiling:
                from pyinstrument import Profiler
                pro = Profiler()
                pro.start()

            t2 = time.time()

            t_sample_prev = t2

            for batch in self.memory:  # this samples a fixed number of batches

                t_sample = time.time()

                if self.total_updates % self.update_buffer_interval == 0:
                    # retrieve local buffer in replay memory
                    self.update_buffer(interface)

                t_update_buffer = time.time()

                if self.total_updates == 0:
                    logging.info(f"starting training")

                stats_training_dict = self.agent.train(batch)

                t_train = time.time()
                eval_enabled = int(getattr(cfg, "TEST_EPISODE_INTERVAL", 0)) > 0
                if eval_enabled:
                    stats_training_dict["return_test"] = self.memory.stat_test_return
                    stats_training_dict["episode_length_test"] = self.memory.stat_test_steps
                    stats_training_dict["return_test_det"] = getattr(self.memory, "stat_test_return_det", self.memory.stat_test_return)
                    stats_training_dict["return_test_stoch"] = getattr(self.memory, "stat_test_return_stoch", self.memory.stat_test_return)
                    stats_training_dict["return_test_det_stoch_gap"] = (
                        stats_training_dict["return_test_stoch"] - stats_training_dict["return_test_det"]
                    )
                    stats_training_dict["episode_length_test_det"] = getattr(self.memory, "stat_test_steps_det", self.memory.stat_test_steps)
                    stats_training_dict["episode_length_test_stoch"] = getattr(self.memory, "stat_test_steps_stoch", self.memory.stat_test_steps)
                else:
                    stats_training_dict["return_test"] = 0.0
                    stats_training_dict["episode_length_test"] = 0.0
                    stats_training_dict["return_test_det"] = 0.0
                    stats_training_dict["return_test_stoch"] = 0.0
                    stats_training_dict["return_test_det_stoch_gap"] = 0.0
                    stats_training_dict["episode_length_test_det"] = 0.0
                    stats_training_dict["episode_length_test_stoch"] = 0.0
                _update_hgi_eval_gap_metrics(stats_training_dict)
                skill_feedback_hook = getattr(self.agent, "update_det_skill_transfer_feedback", None)
                if callable(skill_feedback_hook):
                    for key, value in skill_feedback_hook(stats_training_dict).items():
                        stats_training_dict[key] = value
                stats_training_dict["return_train"] = self.memory.stat_train_return
                stats_training_dict["episode_length_train"] = self.memory.stat_train_steps
                stats_training_dict["sampling_duration"] = t_sample - t_sample_prev
                stats_training_dict["training_step_duration"] = t_train - t_update_buffer
                stats_training += stats_training_dict,
                self.total_updates += 1
                if self.total_updates % self.update_model_interval == 0:
                    # broadcast model weights
                    interface.broadcast_model(self.agent.get_actor())
                self.check_ratio(interface)

                t_sample_prev = time.time()

            t3 = time.time()

            round_time = t3 - t0
            idle_time = t1 - t0
            update_buf_time = t2 - t1
            train_time = t3 - t2
            logging.debug(f"round_time:{round_time}, idle_time:{idle_time}, update_buf_time:{update_buf_time}, train_time:{train_time}")
            stats += pandas_dict(memory_len=len(self.memory), round_time=round_time, idle_time=idle_time, **DataFrame(stats_training).mean(skipna=True)),
            for key, value in self._maybe_save_best_checkpoint(stats[-1], rnd).items():
                stats[-1][key] = value
            entropy_floor_hook = getattr(self.agent, "update_entropy_floor_controller", None)
            if callable(entropy_floor_hook):
                round_diag = stats[-1].to_dict() if hasattr(stats[-1], "to_dict") else dict(stats[-1])
                for key, value in entropy_floor_hook(round_diag).items():
                    stats[-1][key] = value

            logging.info("\n" + _format_sectioned_stats(stats[-1]))

            metrics_writer = getattr(self, "_metrics_writer", None)
            if metrics_writer is not None:
                elapsed = time.time() - self._training_start_time
                metrics_writer.write(stats[-1], epoch=self.epoch, rnd=rnd, wall_clock_seconds=elapsed)
                self._csv_header_written = metrics_writer.stable_header_written

            if self.profiling:
                pro.stop()
                logging.info(pro.output_text(unicode=True, color=False, show_all=True))

        # Epoch-level anti-stall diagnostics.
        self._update_stall_diagnostics(stats)
        self._apply_success_annealing(stats)
        self.epoch += 1
        return stats


class TorchTrainingOffline(TrainingOffline):
    """
    TrainingOffline for trainers based on PyTorch.

    This class implements automatic device selection with PyTorch.
    """
    def __init__(self,
                 env_cls: type = None,
                 memory_cls: type = None,
                 training_agent_cls: type = None,
                 epochs: int = 10,
                 rounds: int = 50,
                 steps: int = 2000,
                 update_model_interval: int = 100,
                 update_buffer_interval: int = 100,
                 max_training_steps_per_env_step: float = 1.0,
                 sleep_between_buffer_retrieval_attempts: float = 1.0,
                 profiling: bool = False,
                 agent_scheduler: callable = None,
                 start_training: int = 0,
                 device: str = None):
        """
        Same arguments as `TrainingOffline`, but when `device` is `None` it is selected automatically for torch.

        Args:
            env_cls (type): class of a dummy environment, used only to retrieve observation and action spaces if needed. Alternatively, this can be a tuple of the form (observation_space, action_space).
            memory_cls (type): class of the replay memory
            training_agent_cls (type): class of the training agent
            epochs (int): total number of epochs, we save the agent every epoch
            rounds (int): number of rounds per epoch, we generate statistics every round
            steps (int): number of training steps per round
            update_model_interval (int): number of training steps between model broadcasts
            update_buffer_interval (int): number of training steps between retrieving buffered samples
            max_training_steps_per_env_step (float): training will pause when above this ratio
            sleep_between_buffer_retrieval_attempts (float): algorithm will sleep for this amount of time when waiting for needed incoming samples
            profiling (bool): if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
            agent_scheduler (callable): if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
            start_training (int): minimum number of samples in the replay buffer before starting training
            device (str): device on which the memory will collate training samples (None for automatic)
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(env_cls,
                         memory_cls,
                         training_agent_cls,
                         epochs,
                         rounds,
                         steps,
                         update_model_interval,
                         update_buffer_interval,
                         max_training_steps_per_env_step,
                         sleep_between_buffer_retrieval_attempts,
                         profiling,
                         agent_scheduler,
                         start_training,
                         device)
