# standard library imports
import itertools
import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

# third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD

# local imports
import tmrl.custom.custom_models as core
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from tmrl.training import TrainingAgent
import tmrl.config.config_constants as cfg

import logging


def _add_tensor_stats(logs, prefix, x, dead_threshold=1e-3):
    if logs is None or x is None:
        return logs

    with torch.no_grad():
        x = x.detach()
        if not x.is_floating_point():
            x = x.float()

        if x.numel() == 0:
            logs[f"{prefix}_mean"] = 0.0
            logs[f"{prefix}_std"] = 0.0
            logs[f"{prefix}_abs_mean"] = 0.0
            logs[f"{prefix}_min"] = 0.0
            logs[f"{prefix}_max"] = 0.0
            return logs

        logs[f"{prefix}_mean"] = x.mean().item()
        logs[f"{prefix}_std"] = x.std(unbiased=False).item()
        logs[f"{prefix}_abs_mean"] = x.abs().mean().item()
        logs[f"{prefix}_min"] = x.min().item()
        logs[f"{prefix}_max"] = x.max().item()

        if x.dim() >= 2 and x.shape[-1] > 0:
            feature_view = x.reshape(-1, x.shape[-1])
            dim_std = feature_view.std(dim=0, unbiased=False)
            logs[f"{prefix}_dead_dim_ratio_1e3"] = (dim_std < dead_threshold).float().mean().item()
            logs[f"{prefix}_dim_std_min"] = dim_std.min().item()
            logs[f"{prefix}_dim_std_mean"] = dim_std.mean().item()
            logs[f"{prefix}_dim_std_max"] = dim_std.max().item()

    return logs


def _parameter_grad_norm(parameters):
    if parameters is None:
        return 0.0
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    total_sq_norm = 0.0
    found_grad = False
    for p in parameters:
        if p is None or p.grad is None:
            continue

        grad = p.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        total_sq_norm += grad.norm(2).item() ** 2
        found_grad = True

    return total_sq_norm ** 0.5 if found_grad else 0.0


def _module_grad_norm(module):
    if module is None:
        return 0.0
    return _parameter_grad_norm(module.parameters())


def _clear_parameter_grads(parameters):
    if parameters is None:
        return
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p is not None:
            p.grad = None


def _compute_actor_churn_loss(current_mu, anchor_mu, enabled=True, weight=0.01):
    """
    Output-space actor anchor used to damp rapid policy churn.

    The penalty is computed on deterministic pre-tanh means so it keeps useful
    gradients even when the squashed actions are close to tanh saturation.
    """
    if current_mu is None or anchor_mu is None:
        device = current_mu.device if current_mu is not None else None
        zero = torch.zeros((), device=device)
        return zero, zero

    raw_loss = F.mse_loss(current_mu, anchor_mu.detach())
    if not enabled or float(weight) <= 0.0:
        zero = raw_loss.detach() * 0.0
        return zero, zero
    return raw_loss, raw_loss * float(weight)


def _compute_critic_churn_loss(current_q, anchor_q, enabled=True, weight=0.01):
    """
    Output-space critic anchor used to reduce value churn.

    This is the value-side CHAIN penalty: keep current Q-head predictions close
    to a slow EMA anchor on replay states/actions while the TD objective still
    drives real learning.
    """
    if current_q is None or anchor_q is None:
        device = current_q.device if current_q is not None else None
        zero = torch.zeros((), device=device)
        return zero, zero

    raw_loss = F.mse_loss(current_q, anchor_q.detach())
    if not enabled or float(weight) <= 0.0:
        zero = raw_loss.detach() * 0.0
        return zero, zero
    return raw_loss, raw_loss * float(weight)


_Q_ACTION_SENSITIVITY_LOSS_TYPE_CODES = {
    "squared": 0.0,
    "linear": 1.0,
    "huber": 2.0,
}


def _compute_q_action_sensitivity_regularizer(
    sensitivity,
    floor=0.05,
    base_weight=1.0,
    max_weight=None,
    loss_type="huber",
    huber_beta=None,
    ramp_power=1.0,
    enabled=True,
):
    """
    Critic action-sensitivity repair loss.

    The coefficient is health-gated from base_weight to max_weight as the
    measured sensitivity falls below floor. The gate is detached so it acts as a
    controller knob rather than adding an extra gradient path through lambda.
    """
    if sensitivity is None:
        zero = torch.zeros(())
        return {
            "raw_loss": zero,
            "weighted_loss": zero,
            "gap": zero,
            "drive": zero,
            "effective_weight": 0.0,
            "loss_type_code": _Q_ACTION_SENSITIVITY_LOSS_TYPE_CODES["huber"],
            "huber_beta": 0.0,
        }

    floor_value = max(0.0, float(floor))
    base_value = max(0.0, float(base_weight))
    if max_weight is None:
        max_value = base_value
    else:
        max_value = max(base_value, max(0.0, float(max_weight)))

    zero = sensitivity.detach() * 0.0
    if not enabled or floor_value <= 0.0 or max_value <= 0.0:
        return {
            "raw_loss": zero,
            "weighted_loss": zero,
            "gap": zero,
            "drive": zero,
            "effective_weight": 0.0,
            "loss_type_code": _Q_ACTION_SENSITIVITY_LOSS_TYPE_CODES.get(
                str(loss_type).lower(), _Q_ACTION_SENSITIVITY_LOSS_TYPE_CODES["huber"]
            ),
            "huber_beta": 0.0,
        }

    floor_t = sensitivity.new_tensor(floor_value)
    gap = F.relu(floor_t - sensitivity)
    drive = (gap / floor_t.clamp_min(1e-12)).clamp(0.0, 1.0)
    power = max(0.0, float(ramp_power))
    if power != 1.0:
        drive_for_weight = drive.detach().pow(power)
    else:
        drive_for_weight = drive.detach()
    effective_weight_t = sensitivity.new_tensor(base_value) + (
        sensitivity.new_tensor(max_value - base_value) * drive_for_weight
    )

    loss_key = str(loss_type).lower().strip()
    if loss_key not in _Q_ACTION_SENSITIVITY_LOSS_TYPE_CODES:
        loss_key = "huber"

    beta_value = max(1e-12, float(huber_beta if huber_beta is not None else floor_value))
    if loss_key == "squared":
        raw_loss = gap.pow(2)
    elif loss_key == "linear":
        raw_loss = gap
    else:
        beta_t = sensitivity.new_tensor(beta_value)
        raw_loss = torch.where(
            gap <= beta_t,
            0.5 * gap.pow(2) / beta_t,
            gap - 0.5 * beta_t,
        )

    return {
        "raw_loss": raw_loss,
        "weighted_loss": raw_loss * effective_weight_t,
        "gap": gap,
        "drive": drive.detach(),
        "effective_weight": float(effective_weight_t.detach().item()),
        "loss_type_code": _Q_ACTION_SENSITIVITY_LOSS_TYPE_CODES[loss_key],
        "huber_beta": beta_value if loss_key == "huber" else 0.0,
    }


_HEALTH_ENTROPY_DIMS = ("steer", "gas", "brake")


class HealthGatedFloorController:
    """
    Per-action entropy-floor sidecar for SAC alpha clamps.

    The learned SAC temperature remains untouched. This controller only moves
    the extra hard floor between a configured baseline and a lower safety floor,
    using logged actor-critic health signals as gates.
    """

    def __init__(self, base_floor, alg_cfg=None, device=None):
        self.device = device or base_floor.device
        self.current_floor = None
        self.base_floor = None
        self.alpha_min_floor = None
        self.last_diag = {}
        self._q_real_ema = None
        self._det_gap_ema = None
        self._best_return_det = None
        self._rounds_since_best = 0
        self._last_direction = []
        self._pending_direction = []
        self._pending_count = []
        self.update_base(base_floor, alg_cfg or {})

    @staticmethod
    def _metric(metrics, name, default):
        try:
            value = metrics.get(name, default)
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _sigmoid(value):
        value = torch.as_tensor(value, dtype=torch.float32)
        return torch.sigmoid(value.clamp(-60.0, 60.0))

    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, float(value)))

    def update_base(self, base_floor, alg_cfg=None):
        alg_cfg = alg_cfg or {}
        base_floor = base_floor.detach().to(self.device, dtype=torch.float32).clamp_min(0.0)
        self.enabled = bool(alg_cfg.get("HEALTH_GATED_ENTROPY_ENABLED", True))
        self.k = max(1.0, float(alg_cfg.get("HEALTH_ENTROPY_K", 4.0)))
        self.eta = max(0.0, min(1.0, float(alg_cfg.get("HEALTH_ENTROPY_ETA", 0.02))))
        self.slew_frac = max(0.0, float(alg_cfg.get("HEALTH_ENTROPY_SLEW_FRAC", 0.2)))
        self.hysteresis_rounds = max(0, int(alg_cfg.get("HEALTH_ENTROPY_HYSTERESIS_ROUNDS", 3)))
        self.stagnation_rounds = max(0, int(alg_cfg.get("HEALTH_ENTROPY_STAGNATION_ROUNDS", 30)))
        self.det_gap_beta = max(0.0, min(0.999, float(alg_cfg.get("HEALTH_ENTROPY_DET_GAP_EMA", 0.9))))
        self.q_real_beta = max(0.0, min(0.999, float(alg_cfg.get("HEALTH_ENTROPY_Q_REAL_EMA", 0.9))))

        alpha_min_abs = max(0.0, float(alg_cfg.get("HEALTH_ENTROPY_ALPHA_MIN", 0.0)))
        alpha_min_scale = max(
            0.0,
            min(1.0, float(alg_cfg.get("HEALTH_ENTROPY_ALPHA_MIN_SCALE", 1.0 / max(self.k, 1.0)))),
        )
        self.base_floor = base_floor
        self.alpha_min_floor = torch.maximum(
            torch.full_like(base_floor, alpha_min_abs),
            base_floor * alpha_min_scale,
        )
        self.alpha_min_floor = torch.minimum(self.alpha_min_floor, base_floor)

        needs_reset = self.current_floor is None or self.current_floor.shape != base_floor.shape
        if needs_reset:
            self.current_floor = base_floor.clone()
            self._last_direction = [0 for _ in range(base_floor.numel())]
            self._pending_direction = [0 for _ in range(base_floor.numel())]
            self._pending_count = [0 for _ in range(base_floor.numel())]
        else:
            self.current_floor = self.current_floor.to(self.device, dtype=torch.float32)
            self.current_floor = torch.maximum(torch.minimum(self.current_floor, base_floor), self.alpha_min_floor)

        if not self.enabled:
            self.current_floor = base_floor.clone()
            self.last_diag = self._disabled_diag()

    def _disabled_diag(self):
        diag = {
            "entropy_health/enabled": 0.0,
            "entropy_health/k": float(self.k),
            "entropy_health/eta": float(self.eta),
            "entropy_health/stagnation_active": 0.0,
            "entropy_health/preserve_signal_present": 0.0,
        }
        for idx, name in enumerate(_HEALTH_ENTROPY_DIMS[: self.current_floor.numel()]):
            diag[f"entropy_health/floor_target_{name}"] = self.base_floor[idx].item()
            diag[f"entropy_health/floor_actual_{name}"] = self.current_floor[idx].item()
            diag[f"entropy_health/floor_min_{name}"] = self.alpha_min_floor[idx].item()
        return diag

    def _grad_health_vector(self, metrics):
        fallback = self._clip01(self._metric(metrics, "guard/grad_health_mean", 1.0))
        values = []
        for name in _HEALTH_ENTROPY_DIMS[: self.base_floor.numel()]:
            values.append(self._clip01(self._metric(metrics, f"grad_health_{name}", fallback)))
        while len(values) < self.base_floor.numel():
            values.append(fallback)
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _update_eval_state(self, metrics):
        det_present = "return_test_det" in metrics and "return_test_stoch" in metrics
        if not det_present:
            return 0.0

        det = self._metric(metrics, "return_test_det", 0.0)
        stoch = self._metric(metrics, "return_test_stoch", det)
        gap = det - stoch
        if self._det_gap_ema is None:
            self._det_gap_ema = gap
        else:
            self._det_gap_ema = self.det_gap_beta * self._det_gap_ema + (1.0 - self.det_gap_beta) * gap

        if self._best_return_det is None or det > self._best_return_det + 1e-6:
            self._best_return_det = det
            self._rounds_since_best = 0
        else:
            self._rounds_since_best += 1
        return 1.0

    def _q_drift_signal(self, metrics, alg_cfg):
        if "bridge/q_real_mean" not in metrics:
            return 0.0, 0.0

        q_real = self._metric(metrics, "bridge/q_real_mean", 0.0)
        q_drift = 0.0 if self._q_real_ema is None else abs(q_real - self._q_real_ema)
        if self._q_real_ema is None:
            self._q_real_ema = q_real
        else:
            self._q_real_ema = self.q_real_beta * self._q_real_ema + (1.0 - self.q_real_beta) * q_real

        tau = max(0.0, float(alg_cfg.get("HEALTH_ENTROPY_Q_DRIFT_THRESHOLD", 0.02)))
        sigma = max(1e-12, float(alg_cfg.get("HEALTH_ENTROPY_Q_DRIFT_SIGMA", 0.02)))
        signal = self._sigmoid((q_drift - tau) / sigma).item()
        return q_drift, signal

    def _compute_target(self, metrics, alg_cfg):
        k = self.k
        dqda_tau = max(
            1e-12,
            float(alg_cfg.get("HEALTH_ENTROPY_DQDA_THRESHOLD", alg_cfg.get("DQDA_STARVATION_THRESHOLD", 1e-4))),
        )
        dqda_sigma = max(1e-12, float(alg_cfg.get("HEALTH_ENTROPY_DQDA_SIGMA", dqda_tau * 0.5)))
        qstd_tau = max(
            0.0,
            float(alg_cfg.get("HEALTH_ENTROPY_Q_PI_STD_THRESHOLD", alg_cfg.get("Q_PI_STD_HEALTH_THRESHOLD", 0.03))),
        )
        qstd_sigma = max(1e-12, float(alg_cfg.get("HEALTH_ENTROPY_Q_PI_STD_SIGMA", max(qstd_tau / 3.0, 0.01))))
        churn_tau = float(alg_cfg.get("HEALTH_ENTROPY_CHURN_THRESHOLD", 0.7))
        churn_sigma = max(1e-12, float(alg_cfg.get("HEALTH_ENTROPY_CHURN_SIGMA", 0.2)))
        consolidate_hard_gate = bool(alg_cfg.get("HEALTH_ENTROPY_CONSOLIDATE_HARD_GATE", True))
        critic_min = float(alg_cfg.get("HEALTH_ENTROPY_CRITIC_HEALTH_MIN", alg_cfg.get("HGI_CRITIC_HEALTH_MIN", 0.35)))
        det_tau = float(alg_cfg.get("HEALTH_ENTROPY_DET_ADVANTAGE", 5.0))
        det_sigma = max(1e-12, float(alg_cfg.get("HEALTH_ENTROPY_DET_SIGMA", 10.0)))
        preserve_hard_gate = bool(alg_cfg.get("HEALTH_ENTROPY_PRESERVE_HARD_GATE", True))

        dqda_norm = self._metric(metrics, "bridge/dqda_norm", dqda_tau * 2.0)
        q_pi_std = self._metric(metrics, "bridge/q_pi_std", qstd_tau * 2.0)
        q_action_min = max(
            1e-12,
            float(alg_cfg.get("Q_ACTION_SENSITIVITY_HEALTH_THRESHOLD", 0.003)),
        )
        q_action = self._metric(metrics, "bridge/q_action_sensitivity", q_action_min * 2.0)
        guard_tier = self._metric(metrics, "guard/tier", 0.0)
        local_critic_ok = (
            dqda_norm >= dqda_tau
            and q_pi_std >= qstd_tau
            and q_action >= q_action_min
            and guard_tier < 1.0
        )
        critic_health = self._metric(
            metrics,
            "hgi/critic_health",
            self._metric(metrics, "hgi/critic_health_pre_eval", 1.0 if local_critic_ok else 0.0),
        )
        model_trust = self._clip01(self._metric(metrics, "hgi/model_trust", 1.0))
        actor_churn = self._metric(metrics, "churn/anchor_mu_abs_diff", self._metric(metrics, "loss_churn", 0.0))
        q_churn = self._metric(metrics, "churn/q_anchor_abs_diff", 0.0)

        starve = self._sigmoid((dqda_tau - dqda_norm) / dqda_sigma).to(self.device)
        flat = self._sigmoid((qstd_tau - q_pi_std) / qstd_sigma).to(self.device)
        search_drive = torch.maximum(starve, flat)
        g_search = 1.0 + (k - 1.0) * search_drive

        if consolidate_hard_gate and actor_churn <= churn_tau:
            high_churn = torch.tensor(0.0, device=self.device)
        else:
            high_churn = self._sigmoid((actor_churn - churn_tau) / churn_sigma).to(self.device)
        critic_ok = 1.0 if critic_health > critic_min else 0.0
        g_consolidate = 1.0 / (1.0 + (k - 1.0) * high_churn * critic_ok)

        preserve_present = self._update_eval_state(metrics)
        if preserve_present:
            det_gap_ema = self._det_gap_ema or 0.0
            if preserve_hard_gate and det_gap_ema <= det_tau:
                preserve_drive = torch.tensor(0.0, device=self.device)
            else:
                preserve_drive = self._sigmoid((det_gap_ema - det_tau) / det_sigma).to(self.device)
            g_preserve = 1.0 / (1.0 + (k - 1.0) * preserve_drive)
        else:
            preserve_drive = torch.tensor(0.0, device=self.device)
            g_preserve = torch.tensor(1.0, device=self.device)

        q_drift, q_drift_drive = self._q_drift_signal(metrics, alg_cfg)
        grad_health = self._grad_health_vector(metrics)

        # FIX #2: Suppress recovery drive when log_std is positive.
        # Low grad_health with positive log_std is a SYMPTOM (noisy policy → noisy grads),
        # not gradient starvation. Boosting exploration would worsen the spiral.
        log_std_mean = self._metric(metrics, "debug_log_std_mean", -1.0)
        log_std_positive = log_std_mean > 0.0
        recovery_drive = torch.maximum(1.0 - grad_health, torch.full_like(grad_health, q_drift_drive))
        recovery_drive = torch.maximum(recovery_drive, torch.full_like(grad_health, 1.0 - model_trust))
        recovery_drive = recovery_drive.clamp(0.0, 1.0)
        if log_std_positive:
            recovery_drive = recovery_drive * 0.0  # Don't boost exploration when already too noisy
        g_recovery = 1.0 + (k - 1.0) * recovery_drive

        target = self.base_floor * g_search * g_consolidate * g_preserve * g_recovery
        target = torch.maximum(torch.minimum(target, self.base_floor), self.alpha_min_floor)

        stale_active = (
            preserve_present
            and self.stagnation_rounds > 0
            and self._rounds_since_best >= self.stagnation_rounds
        )
        eta = self.eta
        # FIX #1: Stagnation override should NOT increase exploration when log_std > 0.
        # The policy is already too noisy — forcing max entropy creates an infinite loop.
        if stale_active and not log_std_positive:
            target = self.base_floor.clone()
            eta = min(1.0, eta * 2.0)

        diag = {
            "entropy_health/enabled": float(self.enabled),
            "entropy_health/k": float(k),
            "entropy_health/eta": float(eta),
            "entropy_health/g_search": g_search.item(),
            "entropy_health/g_consolidate": g_consolidate.item(),
            "entropy_health/g_preserve": g_preserve.item(),
            "entropy_health/search_starve": starve.item(),
            "entropy_health/search_flat": flat.item(),
            "entropy_health/high_churn": high_churn.item(),
            "entropy_health/consolidate_active": float(high_churn.item() > 0.0 and critic_ok > 0.0),
            "entropy_health/consolidate_hard_gate": float(consolidate_hard_gate),
            "entropy_health/critic_ok": float(critic_ok),
            "entropy_health/preserve_drive": preserve_drive.item(),
            "entropy_health/preserve_active": float(preserve_drive.item() > 0.0),
            "entropy_health/preserve_hard_gate": float(preserve_hard_gate),
            "entropy_health/preserve_signal_present": float(preserve_present),
            "entropy_health/det_gap_ema": 0.0 if self._det_gap_ema is None else float(self._det_gap_ema),
            "entropy_health/stagnation_active": float(stale_active),
            "entropy_health/rounds_since_best": float(self._rounds_since_best),
            "entropy_health/q_drift": float(q_drift),
            "entropy_health/q_drift_drive": float(q_drift_drive),
            "entropy_health/actor_churn": float(actor_churn),
            "entropy_health/q_churn": float(q_churn),
            "entropy_health/critic_health": float(critic_health),
            "entropy_health/model_trust": float(model_trust),
        }
        for idx, name in enumerate(_HEALTH_ENTROPY_DIMS[: self.base_floor.numel()]):
            diag[f"entropy_health/g_recovery_{name}"] = g_recovery[idx].item()
            diag[f"entropy_health/grad_health_{name}"] = grad_health[idx].item()
            diag[f"entropy_health/floor_target_{name}"] = target[idx].item()
            diag[f"entropy_health/floor_min_{name}"] = self.alpha_min_floor[idx].item()
        return target, eta, diag

    def step(self, metrics=None, alg_cfg=None):
        metrics = {} if metrics is None else dict(metrics)
        alg_cfg = alg_cfg or {}
        self.update_base(self.base_floor, alg_cfg)
        if not self.enabled:
            return self.current_floor.clone(), dict(self.last_diag)

        target, eta, diag = self._compute_target(metrics, alg_cfg)
        next_floor = self.current_floor.clone()
        for idx in range(self.current_floor.numel()):
            current = self.current_floor[idx].item()
            desired = target[idx].item()
            direction = 1 if desired > current + 1e-12 else (-1 if desired < current - 1e-12 else 0)
            allowed = desired
            if (
                direction != 0
                and self._last_direction[idx] != 0
                and direction != self._last_direction[idx]
                and self.hysteresis_rounds > 0
            ):
                if self._pending_direction[idx] == direction:
                    self._pending_count[idx] += 1
                else:
                    self._pending_direction[idx] = direction
                    self._pending_count[idx] = 1
                if self._pending_count[idx] < self.hysteresis_rounds:
                    allowed = current
                else:
                    self._last_direction[idx] = direction
                    self._pending_count[idx] = 0
            elif direction != 0:
                self._last_direction[idx] = direction
                self._pending_count[idx] = 0
            else:
                self._pending_count[idx] = 0

            proposed = (1.0 - eta) * current + eta * allowed
            max_delta = self.slew_frac * max(abs(current), 1e-12)
            proposed = max(current - max_delta, min(current + max_delta, proposed))
            proposed = max(self.alpha_min_floor[idx].item(), min(self.base_floor[idx].item(), proposed))
            next_floor[idx] = proposed

        self.current_floor = next_floor
        for idx, name in enumerate(_HEALTH_ENTROPY_DIMS[: self.current_floor.numel()]):
            diag[f"entropy_health/floor_actual_{name}"] = self.current_floor[idx].item()
            diag[f"entropy_health/hysteresis_pending_{name}"] = float(self._pending_count[idx])
        self.last_diag = diag
        return self.current_floor.clone(), dict(self.last_diag)

    def get_floor(self):
        return self.current_floor.clone()

    def diagnostics(self):
        return dict(self.last_diag)


def _ema_update_module(anchor, source, polyak):
    polyak = max(0.0, min(1.0, float(polyak)))
    with torch.no_grad():
        for p_anchor, p_source in zip(anchor.parameters(), source.parameters()):
            p_anchor.data.mul_(polyak)
            p_anchor.data.add_((1.0 - polyak) * p_source.data)
        for b_anchor, b_source in zip(anchor.buffers(), source.buffers()):
            if b_anchor.is_floating_point():
                b_anchor.data.mul_(polyak)
                b_anchor.data.add_((1.0 - polyak) * b_source.data)
            else:
                b_anchor.data.copy_(b_source.data)


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


def _linear_risk(value, soft, hard):
    value = float(value)
    soft = float(soft)
    hard = float(hard)
    if hard <= soft:
        return float(value > soft)
    return _clip01((value - soft) / (hard - soft))


def _ema_ratio_update(stats, key, value, decay=0.95, warmup=2, eps=1e-8):
    """Return value / previous EMA, then update the EMA in-place."""
    if stats is None:
        stats = {}
    value = abs(float(value))
    decay = _clip01(decay)
    warmup = max(0, int(warmup))
    count_key = f"{key}_count"
    count = int(stats.get(count_key, 0))
    prev = stats.get(key, None)
    if prev is None:
        baseline = value
        ratio = 1.0
        updated = value
    else:
        baseline = max(abs(float(prev)), eps)
        ratio = value / baseline if count >= warmup else 1.0
        updated = decay * baseline + (1.0 - decay) * value
    stats[key] = float(updated)
    stats[count_key] = count + 1
    return float(ratio), float(baseline), stats


def _compute_det_skill_transfer_feedback(metrics, alg_cfg=None, previous=None):
    """Turn signed det/stoch eval gap into a scale-free deterministic skill signal."""
    alg_cfg = alg_cfg or {}
    metrics = metrics or {}
    previous = previous or {}

    def _metric(name, default=0.0):
        try:
            return float(metrics.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    def _prev(name, default=0.0):
        try:
            return float(previous.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    enabled = bool(alg_cfg.get("DET_SKILL_TRANSFER_ENABLED", True))
    det_present = "return_test_det" in metrics and "return_test_stoch" in metrics
    if not enabled or not det_present:
        return {
            "skill_transfer/enabled": float(enabled),
            "skill_transfer/eval_present": 0.0,
            "skill_transfer/stoch_adv_drive": 0.0,
            "skill_transfer/det_adv_drive": 0.0,
            "skill_transfer/stoch_adv_drive_ema": _prev("skill_transfer/stoch_adv_drive_ema", 0.0),
            "skill_transfer/det_adv_drive_ema": _prev("skill_transfer/det_adv_drive_ema", 0.0),
            "skill_transfer/det_stoch_gap_health": _prev("skill_transfer/det_stoch_gap_health_ema", 1.0),
            "skill_transfer/det_stoch_gap_health_ema": _prev("skill_transfer/det_stoch_gap_health_ema", 1.0),
            "skill_transfer/det_distill_multiplier": 1.0,
        }

    det = _metric("return_test_det", 0.0)
    stoch = _metric("return_test_stoch", det)
    stoch_minus_det = stoch - det
    det_minus_stoch = det - stoch
    scale_floor = max(1e-12, float(alg_cfg.get("DET_SKILL_GAP_SCALE_FLOOR", 1.0)))
    gap_scale = max(0.5 * (abs(det) + abs(stoch)), scale_floor)
    stoch_adv_norm = max(0.0, stoch_minus_det / gap_scale)
    det_adv_norm = max(0.0, det_minus_stoch / gap_scale)

    threshold = float(alg_cfg.get("DET_SKILL_TRANSFER_GAP_THRESHOLD", 0.25))
    sigma = max(1e-12, float(alg_cfg.get("DET_SKILL_TRANSFER_GAP_SIGMA", 0.15)))
    stoch_drive = 0.0 if stoch_adv_norm <= 0.0 else float(
        1.0 / (1.0 + np.exp(-(stoch_adv_norm - threshold) / sigma))
    )
    det_drive = 0.0 if det_adv_norm <= 0.0 else float(
        1.0 / (1.0 + np.exp(-(det_adv_norm - threshold) / sigma))
    )
    beta = _clip01(float(alg_cfg.get("DET_SKILL_TRANSFER_EMA_BETA", 0.8)))

    if previous.get("skill_transfer/eval_present", 0.0):
        stoch_drive_ema = beta * _prev("skill_transfer/stoch_adv_drive_ema", 0.0) + (1.0 - beta) * stoch_drive
        det_drive_ema = beta * _prev("skill_transfer/det_adv_drive_ema", 0.0) + (1.0 - beta) * det_drive
    else:
        stoch_drive_ema = stoch_drive
        det_drive_ema = det_drive

    gap_warn = max(1e-12, float(alg_cfg.get("HGI_DET_STOCH_GAP_WARN", 10.0)))
    if stoch <= 0.0 or stoch_minus_det <= 0.0:
        gap_health = 1.0
    else:
        abs_gap_health = _clip01(1.0 - stoch_minus_det / gap_warn)
        ratio_health = _clip01(det / max(stoch, 1e-12))
        gap_health = min(abs_gap_health, ratio_health)
    if previous.get("skill_transfer/eval_present", 0.0):
        gap_health_ema = beta * _prev("skill_transfer/det_stoch_gap_health_ema", 1.0) + (1.0 - beta) * gap_health
    else:
        gap_health_ema = gap_health

    lambda_max = float(alg_cfg.get("DET_SKILL_TRANSFER_LAMBDA_MAX", alg_cfg.get("DET_REG_LAMBDA", 0.03)))
    lambda_base = float(alg_cfg.get("DET_REG_LAMBDA", 0.03))
    if lambda_max < lambda_base:
        lambda_max = lambda_base
    det_distill_multiplier = (
        1.0 if lambda_base <= 1e-12
        else (lambda_base + stoch_drive_ema * (lambda_max - lambda_base)) / lambda_base
    )

    return {
        "skill_transfer/enabled": float(enabled),
        "skill_transfer/eval_present": 1.0,
        "skill_transfer/return_det": float(det),
        "skill_transfer/return_stoch": float(stoch),
        "skill_transfer/stoch_minus_det": float(stoch_minus_det),
        "skill_transfer/det_minus_stoch": float(det_minus_stoch),
        "skill_transfer/gap_scale": float(gap_scale),
        "skill_transfer/stoch_adv_norm": float(stoch_adv_norm),
        "skill_transfer/det_adv_norm": float(det_adv_norm),
        "skill_transfer/stoch_adv_drive": float(stoch_drive),
        "skill_transfer/det_adv_drive": float(det_drive),
        "skill_transfer/stoch_adv_drive_ema": float(stoch_drive_ema),
        "skill_transfer/det_adv_drive_ema": float(det_drive_ema),
        "skill_transfer/det_stoch_gap_health": float(gap_health),
        "skill_transfer/det_stoch_gap_health_ema": float(gap_health_ema),
        "skill_transfer/det_distill_multiplier": float(det_distill_multiplier),
    }


def _gate_awdb_min_weight(advantage_mean, positive_advantage_mean, target_weight, min_mean=0.0):
    """
    Only apply the AWDB minimum bridge weight when the critic ranks the sampled
    stochastic action better on average.

    A few positive outliers are not enough: if the mean advantage is negative,
    clamping every sample to a nonzero bridge weight teaches from stochastic
    noise rather than from an actually better action.
    """
    try:
        advantage_mean = float(advantage_mean)
        positive_advantage_mean = float(positive_advantage_mean)
        target_weight = float(target_weight)
        min_mean = float(min_mean)
    except (TypeError, ValueError):
        return 0.0
    if target_weight <= 0.0:
        return 0.0
    if advantage_mean <= min_mean:
        return 0.0
    if positive_advantage_mean <= 1e-6:
        return 0.0
    return target_weight


def _compute_actor_guard_decision(
    *,
    guard_enabled,
    grad_health_mean,
    dqda_norm_value,
    q_pi_std_value,
    q_action_sensitivity_value,
    q_overconfidence_norm=0.0,
    churn_norm=1.0,
    data_liveness_health=1.0,
    tier1_health_min=0.05,
    tier3_health_min=0.01,
    dqda_starvation_threshold=1e-4,
    dqda_explosion_threshold=0.01,
    q_pi_std_health_min=0.03,
    q_action_sensitivity_health_min=0.003,
    starvation_lr_scale=0.5,
    tier1_lr_scale=0.1,
    grad_health_only_lr_scale=1.0,
    severe_grad_health_min=0.001,
    tier3_consecutive_blocks=0,
    adaptive_safety_enabled=True,
    q_overconfidence_soft=0.5,
    q_overconfidence_hard=1.25,
    churn_soft_ratio=3.0,
    churn_hard_ratio=8.0,
    adaptive_lr_floor=0.05,
):
    """Decide actor guard intervention without touching model state."""
    dqda_exploding = dqda_norm_value > dqda_explosion_threshold
    dqda_critical = dqda_norm_value > dqda_explosion_threshold * 10.0
    dqda_starving = dqda_norm_value < dqda_starvation_threshold
    grad_health_low = grad_health_mean < tier1_health_min
    grad_health_critical = grad_health_mean < tier3_health_min
    grad_health_severe = grad_health_mean < severe_grad_health_min
    q_pi_std_healthy = (
        q_pi_std_health_min <= 0.0 or q_pi_std_value >= q_pi_std_health_min
    )
    q_action_sensitivity_healthy = (
        q_action_sensitivity_health_min <= 0.0
        or q_action_sensitivity_value >= q_action_sensitivity_health_min
    )
    dqda_healthy = not dqda_starving and not dqda_exploding
    grad_health_only = grad_health_low and dqda_healthy
    healthy_signal_override = (
        guard_enabled
        and grad_health_only
        and q_pi_std_healthy
        and q_action_sensitivity_healthy
        and not grad_health_severe
    )
    q_overconfidence_risk = _linear_risk(
        q_overconfidence_norm,
        q_overconfidence_soft,
        q_overconfidence_hard,
    )
    churn_risk = _linear_risk(churn_norm, churn_soft_ratio, churn_hard_ratio)
    data_liveness_health = _clip01(data_liveness_health)
    data_liveness_risk = 1.0 - data_liveness_health
    adaptive_risks = {
        "q_overconfidence": q_overconfidence_risk,
        "churn": churn_risk,
        "data_liveness": data_liveness_risk,
    }
    adaptive_risk_name, adaptive_risk = max(adaptive_risks.items(), key=lambda item: item[1])
    adaptive_safety_active = bool(
        guard_enabled and adaptive_safety_enabled and adaptive_risk > 0.0
    )
    adaptive_lr_scale = 1.0
    if adaptive_safety_active:
        adaptive_lr_scale = max(float(adaptive_lr_floor), 1.0 - float(adaptive_risk))

    tier1_active = guard_enabled and (
        (grad_health_low and not healthy_signal_override)
        or dqda_starving
        or dqda_exploding
    )
    tier3_active = guard_enabled and (grad_health_severe or dqda_critical)

    tier3_hard_block_active = False
    tier3_forced_exploration = False
    lr_scale = 1.0
    grad_clip_norm = 1.0
    next_tier3_consecutive_blocks = tier3_consecutive_blocks
    throttle_reason = 0.0  # 0 none, 1 starvation, 2 explosion, 3 grad-health, 4 tier3-health, 5 tier3-dqda, 6 forced, 7 q-gap, 8 churn, 9 data

    if not guard_enabled:
        next_tier3_consecutive_blocks = 0
    elif tier3_active:
        throttle_reason = 5.0 if dqda_critical else 4.0
        if tier3_consecutive_blocks > 3:
            tier3_forced_exploration = True
            lr_scale = 0.05
            next_tier3_consecutive_blocks = 0
            throttle_reason = 6.0
        else:
            tier3_hard_block_active = True
            next_tier3_consecutive_blocks = tier3_consecutive_blocks + 1
    else:
        next_tier3_consecutive_blocks = 0
        if tier1_active:
            if dqda_starving and not dqda_exploding:
                lr_scale = starvation_lr_scale
                throttle_reason = 1.0
                if grad_health_low:
                    grad_clip_norm = 0.1
            else:
                lr_scale = tier1_lr_scale
                grad_clip_norm = 0.1
                throttle_reason = 2.0 if dqda_exploding else 3.0
        elif healthy_signal_override:
            lr_scale = grad_health_only_lr_scale

    if adaptive_safety_active and not tier3_hard_block_active:
        if adaptive_lr_scale < lr_scale:
            lr_scale = adaptive_lr_scale
            if adaptive_risk_name == "q_overconfidence":
                throttle_reason = 7.0
            elif adaptive_risk_name == "churn":
                throttle_reason = 8.0
            else:
                throttle_reason = 9.0
        elif throttle_reason == 0.0:
            if adaptive_risk_name == "q_overconfidence":
                throttle_reason = 7.0
            elif adaptive_risk_name == "churn":
                throttle_reason = 8.0
            else:
                throttle_reason = 9.0

    if tier3_hard_block_active or tier3_forced_exploration:
        guard_tier = 3.0
    elif tier1_active or adaptive_safety_active:
        guard_tier = 1.0
    else:
        guard_tier = 0.0

    return {
        "dqda_exploding": dqda_exploding,
        "dqda_critical": dqda_critical,
        "dqda_starving": dqda_starving,
        "grad_health_low": grad_health_low,
        "grad_health_critical": grad_health_critical,
        "grad_health_severe": grad_health_severe,
        "grad_health_only": grad_health_only,
        "q_pi_std_healthy": q_pi_std_healthy,
        "q_action_sensitivity_healthy": q_action_sensitivity_healthy,
        "healthy_signal_override": healthy_signal_override,
        "q_overconfidence_risk": float(q_overconfidence_risk),
        "q_overconfidence_health": float(1.0 - q_overconfidence_risk),
        "churn_risk": float(churn_risk),
        "churn_health": float(1.0 - churn_risk),
        "data_liveness_health": float(data_liveness_health),
        "data_liveness_risk": float(data_liveness_risk),
        "adaptive_safety_active": adaptive_safety_active,
        "adaptive_risk": float(adaptive_risk),
        "adaptive_lr_scale": float(adaptive_lr_scale),
        "tier1_active": tier1_active,
        "tier3_active": tier3_active,
        "tier3_hard_block_active": tier3_hard_block_active,
        "tier3_forced_exploration": tier3_forced_exploration,
        "tier3_consecutive_blocks": next_tier3_consecutive_blocks,
        "guard_tier": guard_tier,
        "lr_scale": lr_scale,
        "grad_clip_norm": grad_clip_norm,
        "throttle_reason": throttle_reason,
    }


_HGI_ACTOR_HEALTH_REQUIRED_KEYS = (
    "bridge/dqda_norm",
    "bridge/q_pi_std",
)

_HGI_ACTOR_HEALTH_CACHE_KEYS = (
    "bridge/dqda_norm",
    "bridge/dqda_abs_mean",
    "bridge/dqda_clipped",
    "bridge/dqda_norm_max",
    "bridge/dqda_norm_p95",
    "bridge/q_pi_mean",
    "bridge/q_pi_std",
    "bridge/q_pi_action_corr_d0",
    "bridge/q_pi_action_corr_d1",
    "bridge/q_pi_action_corr_d2",
    "guard/q_gap",
    "guard/q_scale",
    "guard/q_gap_norm",
    "guard/q_overconfidence_risk",
    "guard/q_overconfidence_health",
    "guard/churn_baseline",
    "guard/churn_norm",
    "guard/churn_risk",
    "guard/churn_health",
    "guard/reward_energy",
    "guard/reward_energy_baseline",
    "guard/reward_energy_ratio",
    "guard/data_liveness_health",
    "guard/data_liveness_risk",
    "guard/adaptive_safety_active",
    "guard/adaptive_risk",
    "guard/adaptive_lr_scale",
    "guard/actor_stability_active",
    "guard/actor_mu_step_blocked",
    "guard/actor_std_step_allowed",
    "guard/tier",
    "guard/tier3_hard_block_active",
    "guard/tier3_forced_exploration",
    "guard/lr_scale",
    "guard/q_shield_triggered",
    "guard/q_drop_value",
    "guard/dqda_min_norm",
    "guard/grad_health_min",
    "guard/severe_grad_health_min",
    "guard/grad_health_mean",
    "guard/dqda_explosion",
    "guard/dqda_critical",
    "guard/dqda_starvation",
    "guard/grad_health_severe",
    "guard/starvation_lr_scale",
    "guard/grad_health_only",
    "guard/healthy_signal_override",
    "guard/throttle_reason",
    "guard/q_pi_std_health_min",
    "guard/q_pi_std_healthy",
    "guard/q_action_sensitivity_health_min",
    "guard/q_action_sensitivity_healthy",
    "guard/grad_health_only_lr_scale",
    "guard/alpha_ceiling_hit",
)


def _compute_hgi_trust_metrics(metrics, alg_cfg=None, actor_health_cache=None):
    """Compute SPECTRE-HGI trust diagnostics from already-logged signals."""
    alg_cfg = alg_cfg or {}
    metrics = metrics or {}
    metric_view = dict(metrics)
    actor_health_cache = actor_health_cache or {}
    actor_health_cache_fill_count = 0
    for key in _HGI_ACTOR_HEALTH_CACHE_KEYS:
        if key not in metric_view and key in actor_health_cache:
            metric_view[key] = actor_health_cache[key]
            actor_health_cache_fill_count += 1
    actor_health_fresh_present = all(key in metrics for key in _HGI_ACTOR_HEALTH_REQUIRED_KEYS)

    def _metric(name, default=0.0):
        try:
            return float(metric_view.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    def _metric_present(name):
        return name in metric_view

    def _clip01(value):
        return max(0.0, min(1.0, float(value)))

    def _exp_decay(value, scale):
        return float(np.exp(-max(0.0, float(value)) * max(0.0, float(scale))))

    kl_value = _metric("wm/kl_mean", _metric("wm_kl", 0.0))
    reward_error = _metric("wm/reward_error_abs_mean", _metric("wm_recon_reward", 0.0))
    reward_std = max(abs(_metric("wm/reward_target_std", _metric("wm_input/reward_std", 0.0))), 1e-6)
    recon_ratio = max(_metric("wm_val/recon_prior_post_ratio", 1.0), 1e-6)
    verifier_trust = _clip01(_metric("wm/verifier_trust_mean", _metric("verifier_trust_mt_mean", 1.0)))
    post_prior_mu_diff = _metric("wm/post_prior_mu_abs_diff", -1.0)
    post_advantage_ratio = _metric("wm_val/post_advantage_ratio", -1.0)
    latent_signal_present = post_prior_mu_diff >= 0.0 or post_advantage_ratio >= 0.0

    kl_trust_mode = str(alg_cfg.get("HGI_KL_TRUST_MODE", "band")).lower()
    if kl_trust_mode == "exp":
        kl_trust = _exp_decay(kl_value, alg_cfg.get("HGI_KL_TRUST_SCALE", 10.0))
    else:
        kl_ref = _metric(
            "wm_kl_clamped",
            _metric("wm_kl_loss", alg_cfg.get("HGI_KL_HEALTH_REF", 1.0)),
        )
        kl_ref = max(abs(float(kl_ref)), 1e-8)
        kl_low = max(float(alg_cfg.get("HGI_KL_HEALTH_LOW_RATIO", 0.03)) * kl_ref, 1e-8)
        kl_high = max(float(alg_cfg.get("HGI_KL_HEALTH_HIGH_RATIO", 2.5)) * kl_ref, kl_low)
        kl_low_health = _clip01(kl_value / kl_low)
        kl_high_health = 1.0 if kl_value <= kl_high else _clip01(kl_high / max(kl_value, 1e-12))
        kl_trust = min(kl_low_health, kl_high_health)
    if latent_signal_present:
        latent_kl_min = float(alg_cfg.get("HGI_LATENT_KL_MIN", 0.01))
        latent_diff_min = float(alg_cfg.get("HGI_POST_PRIOR_DIFF_MIN", 0.005))
        latent_advantage_min = float(alg_cfg.get("HGI_LATENT_ADVANTAGE_RATIO_MIN", 0.05))
        latent_kl_health = _clip01(kl_value / max(latent_kl_min, 1e-12))
        latent_diff_health = _clip01(max(0.0, post_prior_mu_diff) / max(latent_diff_min, 1e-12))
        latent_advantage_health = _clip01(max(0.0, post_advantage_ratio) / max(latent_advantage_min, 1e-12))
        if post_advantage_ratio >= 0.0:
            latent_alive_trust = latent_advantage_health
        else:
            latent_alive_trust = 0.0
    else:
        latent_kl_health = 1.0
        latent_diff_health = 1.0
        latent_advantage_health = 1.0
        latent_alive_trust = 1.0
    reward_trust = _exp_decay(
        reward_error / reward_std,
        alg_cfg.get("HGI_REWARD_ERROR_TRUST_SCALE", 0.25),
    )
    recon_ratio_trust = _exp_decay(
        abs(float(np.log(recon_ratio))),
        alg_cfg.get("HGI_RECON_RATIO_TRUST_SCALE", 2.0),
    )
    model_components = [kl_trust, reward_trust, recon_ratio_trust, verifier_trust, latent_alive_trust]
    model_trust = float(np.prod(model_components) ** (1.0 / len(model_components)))

    dqda_norm = _metric("bridge/dqda_norm", 0.0)
    dqda_starvation_threshold = float(alg_cfg.get("DQDA_STARVATION_THRESHOLD", 1e-4))
    dqda_explosion_threshold = float(alg_cfg.get("DQDA_EXPLOSION_THRESHOLD", 0.01))
    dqda_low_health = _clip01(dqda_norm / max(dqda_starvation_threshold, 1e-12))
    dqda_high_health = 1.0
    if dqda_norm > dqda_explosion_threshold:
        dqda_high_health = _clip01(dqda_explosion_threshold / max(dqda_norm, 1e-12))
    dqda_health = min(dqda_low_health, dqda_high_health)

    q_pi_std = _metric("bridge/q_pi_std", 0.0)
    q_pi_std_min = float(alg_cfg.get("Q_PI_STD_HEALTH_THRESHOLD", 0.03))
    q_pi_std_health = _clip01(q_pi_std / max(q_pi_std_min, 1e-12)) if q_pi_std_min > 0.0 else 1.0

    q_action_sensitivity = _metric("bridge/q_action_sensitivity", 0.0)
    q_action_sensitivity_min = float(alg_cfg.get("Q_ACTION_SENSITIVITY_HEALTH_THRESHOLD", 0.003))
    q_action_sensitivity_health = (
        _clip01(q_action_sensitivity / max(q_action_sensitivity_min, 1e-12))
        if q_action_sensitivity_min > 0.0 else 1.0
    )

    grad_health_mean = _metric("guard/grad_health_mean", 1.0)
    tier1_health_min = float(alg_cfg.get("TIER1_HEALTH_THRESHOLD", 0.05))
    grad_health = _clip01(grad_health_mean / max(tier1_health_min, 1e-12))
    if _metric("guard/healthy_signal_override", 0.0) > 0.5:
        grad_health = max(grad_health, float(alg_cfg.get("HGI_GRAD_HEALTH_OVERRIDE_FLOOR", 0.8)))

    guard_tier = _metric("guard/tier", 0.0)
    guard_health = _clip01(1.0 - guard_tier / 3.0)
    if _metric("guard/healthy_signal_override", 0.0) > 0.5:
        guard_health = 1.0

    det_stoch_gap_health = _clip01(
        _metric(
            "hgi/det_stoch_gap_health",
            _metric(
                "skill_transfer/det_stoch_gap_health_ema",
                _metric("skill_transfer/det_stoch_gap_health", 1.0),
            ),
        )
    )
    if _metric_present("guard/q_overconfidence_health"):
        q_overconfidence_health = _clip01(_metric("guard/q_overconfidence_health", 1.0))
    elif _metric_present("guard/q_overconfidence_risk"):
        q_overconfidence_health = _clip01(1.0 - _metric("guard/q_overconfidence_risk", 0.0))
    elif _metric_present("bridge/q_pi_mean") and _metric_present("bridge/q_real_mean"):
        q_gap = _metric("bridge/q_pi_mean", 0.0) - _metric("bridge/q_real_mean", 0.0)
        q_scale = max(
            abs(_metric("bridge/q_pi_std", 0.0)) + abs(_metric("bridge/q_real_std", 0.0)),
            1e-6,
        )
        q_gap_norm = max(0.0, q_gap / q_scale)
        q_overconfidence_health = 1.0 - _linear_risk(
            q_gap_norm,
            alg_cfg.get("ADAPTIVE_Q_GAP_SOFT", 0.5),
            alg_cfg.get("ADAPTIVE_Q_GAP_HARD", 1.25),
        )
    else:
        q_overconfidence_health = 1.0
    churn_health = _clip01(_metric("guard/churn_health", 1.0))
    if _metric_present("guard/churn_risk"):
        churn_health = min(churn_health, _clip01(1.0 - _metric("guard/churn_risk", 0.0)))
    data_liveness_health = _clip01(_metric("guard/data_liveness_health", 1.0))
    if _metric_present("guard/data_liveness_risk"):
        data_liveness_health = min(
            data_liveness_health,
            _clip01(1.0 - _metric("guard/data_liveness_risk", 0.0)),
        )
    episode_length_keys = (
        "episode_length_train",
        "episode_length_test",
        "episode_length_test_det",
        "episode_length_test_stoch",
    )
    present_lengths = [
        _metric(key, 0.0)
        for key in episode_length_keys
        if _metric_present(key)
    ]
    if present_lengths and max(present_lengths) <= 0.0:
        data_liveness_health = 0.0
    critic_health = min(
        dqda_health,
        q_pi_std_health,
        q_action_sensitivity_health,
        grad_health,
        guard_health,
        det_stoch_gap_health,
        q_overconfidence_health,
        churn_health,
        data_liveness_health,
    )

    return {
        "hgi/model_trust": _clip01(model_trust),
        "hgi/critic_health": _clip01(critic_health),
        "hgi/imag_trust": _clip01(model_trust * critic_health),
        "hgi/model_trust_kl": _clip01(kl_trust),
        "hgi/model_trust_reward": _clip01(reward_trust),
        "hgi/model_trust_recon_ratio": _clip01(recon_ratio_trust),
        "hgi/model_trust_verifier": _clip01(verifier_trust),
        "hgi/model_trust_latent_alive": _clip01(latent_alive_trust),
        "hgi/latent_kl_health": _clip01(latent_kl_health),
        "hgi/latent_diff_health": _clip01(latent_diff_health),
        "hgi/latent_advantage_health": _clip01(latent_advantage_health),
        "hgi/critic_health_dqda": _clip01(dqda_health),
        "hgi/critic_health_q_pi_std": _clip01(q_pi_std_health),
        "hgi/critic_health_q_action_sensitivity": _clip01(q_action_sensitivity_health),
        "hgi/critic_health_grad": _clip01(grad_health),
        "hgi/critic_health_guard": _clip01(guard_health),
        "hgi/det_stoch_gap_health": det_stoch_gap_health,
        "hgi/critic_health_q_overconfidence": _clip01(q_overconfidence_health),
        "hgi/critic_health_churn": _clip01(churn_health),
        "hgi/critic_health_data_liveness": _clip01(data_liveness_health),
        "hgi/actor_health_cache_used": float(actor_health_cache_fill_count > 0),
        "hgi/actor_health_cache_fill_count": float(actor_health_cache_fill_count),
        "hgi/actor_health_fresh_present": float(actor_health_fresh_present),
    }


def _compute_hgi_imagination_gate(metrics, alg_cfg=None, base_horizon=15, post_warmup_steps=0):
    """Choose skip/short/full imagination from HGI trust metrics."""
    alg_cfg = alg_cfg or {}

    def _metric(name, default=0.0):
        try:
            return float(metrics.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    def _clip01(value):
        return max(0.0, min(1.0, float(value)))

    base_horizon = max(0, int(base_horizon))
    full_horizon = int(alg_cfg.get("HGI_FULL_HORIZON", base_horizon))
    full_horizon = max(0, min(base_horizon, full_horizon))
    short_horizon = int(alg_cfg.get("HGI_SHORT_HORIZON", min(3, full_horizon)))
    if full_horizon >= 2:
        short_horizon = max(2, min(short_horizon, full_horizon))
    else:
        short_horizon = 0

    hgi_enabled = bool(alg_cfg.get("HGI_ENABLED", True))
    skip_unhealthy = bool(alg_cfg.get("HGI_SKIP_UNHEALTHY_IMAGINATION", True))
    model_min = float(alg_cfg.get("HGI_MODEL_TRUST_MIN", 0.35))
    critic_min = float(alg_cfg.get("HGI_CRITIC_HEALTH_MIN", 0.35))
    full_trust_min = float(alg_cfg.get("HGI_FULL_TRUST_MIN", 0.70))
    ramp_enabled = bool(alg_cfg.get("HGI_POST_WARMUP_RAMP_ENABLED", True))
    post_warmup_short_steps = max(
        0.0,
        float(alg_cfg.get("HGI_POST_WARMUP_SHORT_STEPS", 5000.0)),
    )
    post_warmup_steps = max(0.0, float(post_warmup_steps))
    ramp_window_active = (
        ramp_enabled
        and post_warmup_short_steps > 0.0
        and post_warmup_steps < post_warmup_short_steps
    )
    ramp_remaining_steps = max(0.0, post_warmup_short_steps - post_warmup_steps)

    model_trust = _clip01(_metric("hgi/model_trust", 1.0))
    critic_health = _clip01(_metric("hgi/critic_health", 1.0))
    imag_trust = _clip01(_metric("hgi/imag_trust", model_trust * critic_health))
    skip_model = model_trust < model_min
    skip_critic = critic_health < critic_min

    effective_horizon = full_horizon
    skipped = 0.0
    gate_state = 3.0  # 0 disabled, 1 skipped, 2 short, 3 full
    ramp_active = 0.0

    if not hgi_enabled:
        effective_horizon = 0
        skipped = 1.0
        gate_state = 0.0
    elif skip_unhealthy and (skip_model or skip_critic):
        effective_horizon = 0
        skipped = 1.0
        gate_state = 1.0
    elif ramp_window_active:
        effective_horizon = short_horizon
        gate_state = 2.0
        ramp_active = 1.0
    elif imag_trust < full_trust_min:
        effective_horizon = short_horizon
        gate_state = 2.0

    if effective_horizon == 1:
        effective_horizon = 0
        skipped = 1.0
        gate_state = 1.0

    return {
        "hgi/effective_horizon": float(effective_horizon),
        "hgi/skipped_ratio": float(skipped),
        "hgi/skip_reason_model": float(skip_model and skipped > 0.0),
        "hgi/skip_reason_critic": float(skip_critic and skipped > 0.0),
        "hgi/gate_state": float(gate_state),
        "hgi/model_trust_min": float(model_min),
        "hgi/critic_health_min": float(critic_min),
        "hgi/full_trust_min": float(full_trust_min),
        "hgi/short_horizon": float(short_horizon),
        "hgi/full_horizon": float(full_horizon),
        "hgi/ramp_active": float(ramp_active),
        "hgi/post_warmup_steps": float(post_warmup_steps),
        "hgi/ramp_remaining_steps": float(ramp_remaining_steps),
        "hgi/post_warmup_short_steps": float(post_warmup_short_steps),
    }


def _compute_hgi_warmup_gate(alg_cfg=None, base_horizon=15, warmup_remaining_steps=0):
    """Expose stable HGI gate columns while the world model is still warming up."""
    gate = _compute_hgi_imagination_gate(
        {
            "hgi/model_trust": 1.0,
            "hgi/critic_health": 1.0,
            "hgi/imag_trust": 1.0,
        },
        alg_cfg,
        base_horizon=base_horizon,
    )
    gate["hgi/effective_horizon"] = 0.0
    gate["hgi/skipped_ratio"] = 1.0
    gate["hgi/skip_reason_model"] = 0.0
    gate["hgi/skip_reason_critic"] = 0.0
    gate["hgi/skip_reason_warmup"] = 1.0
    gate["hgi/warmup_active"] = 1.0
    gate["hgi/warmup_remaining_steps"] = float(max(0.0, warmup_remaining_steps))
    gate["hgi/gate_state"] = 1.0
    gate["hgi/ramp_active"] = 0.0
    gate["hgi/post_warmup_steps"] = 0.0
    return gate


# Soft Actor-Critic ====================================================================================================


@dataclass(eq=0)
class SpinupSacAgent(TrainingAgent):  # Adapted from Spinup
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = core.MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning (SAC v2)
    learn_entropy_coef: bool = True  # if True, SAC v2 is used, else, SAC v1 is used
    target_entropy: float = None  # if None, the target entropy for SAC v2 is set automatically
    optimizer_actor: str = "adam"  # one of ["adam", "adamw", "sgd"]
    optimizer_critic: str = "adam"  # one of ["adam", "adamw", "sgd"]
    betas_actor: tuple = None  # for Adam and AdamW
    betas_critic: tuple = None  # for Adam and AdamW
    l2_actor: float = None  # weight decay
    l2_critic: float = None  # weight decay

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up optimizers for policy and q-function:

        self.optimizer_actor = self.optimizer_actor.lower()
        self.optimizer_critic = self.optimizer_critic.lower()
        if self.optimizer_actor not in ["adam", "adamw", "sgd"]:
            logging.warning(f"actor optimizer {self.optimizer_actor} is not valid, defaulting to sgd")
        if self.optimizer_critic not in ["adam", "adamw", "sgd"]:
            logging.warning(f"critic optimizer {self.optimizer_critic} is not valid, defaulting to sgd")
        if self.optimizer_actor == "adam":
            pi_optimizer_cls = Adam
        elif self.optimizer_actor == "adamw":
            pi_optimizer_cls = AdamW
        else:
            pi_optimizer_cls = SGD
        pi_optimizer_kwargs = {"lr": self.lr_actor}
        if self.optimizer_actor in ["adam, adamw"] and self.betas_actor is not None:
            pi_optimizer_kwargs["betas"] = tuple(self.betas_actor)
        if self.l2_actor is not None:
            pi_optimizer_kwargs["weight_decay"] = self.l2_actor

        if self.optimizer_critic == "adam":
            q_optimizer_cls = Adam
        elif self.optimizer_critic == "adamw":
            q_optimizer_cls = AdamW
        else:
            q_optimizer_cls = SGD
        q_optimizer_kwargs = {"lr": self.lr_critic}
        if self.optimizer_critic in ["adam, adamw"] and self.betas_critic is not None:
            q_optimizer_kwargs["betas"] = tuple(self.betas_critic)
        if self.l2_critic is not None:
            q_optimizer_kwargs["weight_decay"] = self.l2_critic

        self.pi_optimizer = pi_optimizer_cls(self.model.actor.parameters(), **pi_optimizer_kwargs)
        self.q_optimizer = q_optimizer_cls(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()), **q_optimizer_kwargs)

        # entropy coefficient:

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        o, a, r, o2, d, _ = batch

        pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        # loss_alpha:

        loss_alpha = None
        if self.learn_entropy_coef:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            limit = getattr(self, "alpha_floor", 0.05)
            self.log_alpha.clamp_(min=np.log(limit))
        # Run one gradient descent step for Q1 and Q2

        # loss_q:

        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.actor(o2)

            # Target Q-values
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = (loss_q1 + loss_q2) / 2  # averaged for homogeneity with REDQ

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        # Next run one gradient descent step for actor.

        # loss_pi:

        # pi, logp_pi = self.model.actor(o)
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha_t * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # FIXME: remove debug info
        with torch.no_grad():

            if not cfg.DEBUG_MODE:
                ret_dict = dict(
                    loss_actor=loss_pi.detach().item(),
                    loss_critic=loss_q.detach().item(),
                )
            else:
                q1_o2_a2 = self.model.q1(o2, a2)
                q2_o2_a2 = self.model.q2(o2, a2)
                q1_targ_pi = self.model_target.q1(o, pi)
                q2_targ_pi = self.model_target.q2(o, pi)
                q1_targ_a = self.model_target.q1(o, a)
                q2_targ_a = self.model_target.q2(o, a)

                diff_q1pt_qpt = (q1_pi_targ - q_pi_targ).detach()
                diff_q2pt_qpt = (q2_pi_targ - q_pi_targ).detach()
                diff_q1_q1t_a2 = (q1_o2_a2 - q1_pi_targ).detach()
                diff_q2_q2t_a2 = (q2_o2_a2 - q2_pi_targ).detach()
                diff_q1_q1t_pi = (q1_pi - q1_targ_pi).detach()
                diff_q2_q2t_pi = (q2_pi - q2_targ_pi).detach()
                diff_q1_q1t_a = (q1 - q1_targ_a).detach()
                diff_q2_q2t_a = (q2 - q2_targ_a).detach()
                diff_q1_backup = (q1 - backup).detach()
                diff_q2_backup = (q2 - backup).detach()
                diff_q1_backup_r = (q1 - backup + r).detach()
                diff_q2_backup_r = (q2 - backup + r).detach()

                ret_dict = dict(
                    loss_actor=loss_pi.detach().item(),
                    loss_critic=loss_q.detach().item(),
                    # debug:
                    debug_log_pi=logp_pi.detach().mean().item(),
                    debug_log_pi_std=logp_pi.detach().std().item(),
                    debug_logp_a2=logp_a2.detach().mean().item(),
                    debug_logp_a2_std=logp_a2.detach().std().item(),
                    debug_q_a1=q_pi.detach().mean().item(),
                    debug_q_a1_std=q_pi.detach().std().item(),
                    debug_q_a1_targ=q_pi_targ.detach().mean().item(),
                    debug_q_a1_targ_std=q_pi_targ.detach().std().item(),
                    debug_backup=backup.detach().mean().item(),
                    debug_backup_std=backup.detach().std().item(),
                    debug_q1=q1.detach().mean().item(),
                    debug_q1_std=q1.detach().std().item(),
                    debug_q2=q2.detach().mean().item(),
                    debug_q2_std=q2.detach().std().item(),
                    debug_diff_q1=diff_q1_backup.mean().item(),
                    debug_diff_q1_std=diff_q1_backup.std().item(),
                    debug_diff_q2=diff_q2_backup.mean().item(),
                    debug_diff_q2_std=diff_q2_backup.std().item(),
                    debug_diff_r_q1=diff_q1_backup_r.mean().item(),
                    debug_diff_r_q1_std=diff_q1_backup_r.std().item(),
                    debug_diff_r_q2=diff_q2_backup_r.mean().item(),
                    debug_diff_r_q2_std=diff_q2_backup_r.std().item(),
                    debug_diff_q1pt_qpt=diff_q1pt_qpt.mean().item(),
                    debug_diff_q2pt_qpt=diff_q2pt_qpt.mean().item(),
                    debug_diff_q1_q1t_a2=diff_q1_q1t_a2.mean().item(),
                    debug_diff_q2_q2t_a2=diff_q2_q2t_a2.mean().item(),
                    debug_diff_q1_q1t_pi=diff_q1_q1t_pi.mean().item(),
                    debug_diff_q2_q2t_pi=diff_q2_q2t_pi.mean().item(),
                    debug_diff_q1_q1t_a=diff_q1_q1t_a.mean().item(),
                    debug_diff_q2_q2t_a=diff_q2_q2t_a.mean().item(),
                    debug_diff_q1pt_qpt_std=diff_q1pt_qpt.std().item(),
                    debug_diff_q2pt_qpt_std=diff_q2pt_qpt.std().item(),
                    debug_diff_q1_q1t_a2_std=diff_q1_q1t_a2.std().item(),
                    debug_diff_q2_q2t_a2_std=diff_q2_q2t_a2.std().item(),
                    debug_diff_q1_q1t_pi_std=diff_q1_q1t_pi.std().item(),
                    debug_diff_q2_q2t_pi_std=diff_q2_q2t_pi.std().item(),
                    debug_diff_q1_q1t_a_std=diff_q1_q1t_a.std().item(),
                    debug_diff_q2_q2t_a_std=diff_q2_q2t_a.std().item(),
                    debug_r=r.detach().mean().item(),
                    debug_r_std=r.detach().std().item(),
                    debug_d=d.detach().mean().item(),
                    debug_d_std=d.detach().std().item(),
                    debug_a_0=a[:, 0].detach().mean().item(),
                    debug_a_0_std=a[:, 0].detach().std().item(),
                    debug_a_1=a[:, 1].detach().mean().item(),
                    debug_a_1_std=a[:, 1].detach().std().item(),
                    debug_a_2=a[:, 2].detach().mean().item(),
                    debug_a_2_std=a[:, 2].detach().std().item(),
                    debug_a1_0=pi[:, 0].detach().mean().item(),
                    debug_a1_0_std=pi[:, 0].detach().std().item(),
                    debug_a1_1=pi[:, 1].detach().mean().item(),
                    debug_a1_1_std=pi[:, 1].detach().std().item(),
                    debug_a1_2=pi[:, 2].detach().mean().item(),
                    debug_a1_2_std=pi[:, 2].detach().std().item(),
                    debug_a2_0=a2[:, 0].detach().mean().item(),
                    debug_a2_0_std=a2[:, 0].detach().std().item(),
                    debug_a2_1=a2[:, 1].detach().mean().item(),
                    debug_a2_1_std=a2[:, 1].detach().std().item(),
                    debug_a2_2=a2[:, 2].detach().mean().item(),
                    debug_a2_2_std=a2[:, 2].detach().std().item(),
                )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# REDQ-SAC =============================================================================================================

@dataclass(eq=0)
class REDQSACAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = core.REDQMLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning
    learn_entropy_coef: bool = True
    target_entropy: float = None  # if None, the target entropy is set automatically
    n: int = 10  # number of REDQ parallel Q networks
    m: int = 2  # number of REDQ randomly sampled target networks
    q_updates_per_policy_update: int = 1  # in REDQ, this is the "UTD ratio" (20), this interplays with lr_actor

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device REDQ-SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer_list = [Adam(q.parameters(), lr=self.lr_critic) for q in self.model.qs]
        self.criterion = torch.nn.MSELoss()
        self.loss_pi = torch.zeros((1,), device=device)

        self.i_update = 0  # for UTD ratio

        if self.target_entropy is None:  # automatic entropy coefficient
            self.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        self.i_update += 1
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)
        # DEBUG: Confirm new code is running
        # print(f"DEBUG: Train Step {self.i_update}, Update Policy: {update_policy}, Learn Ent: {self.learn_entropy_coef}")

        o, a, r, o2, d, _ = batch

        if update_policy:
            pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        loss_alpha = None
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
            if update_policy:
                loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)

            sample_idxs = np.random.choice(self.n, self.m, replace=False)

            q_prediction_next_list = [self.model_target.qs[i](o2, a2) for i in sample_idxs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_t * logp_a2.unsqueeze(dim=-1))

        q_prediction_list = [q(o, a) for q in self.model.qs]
        q_prediction_cat = torch.stack(q_prediction_list, -1)
        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        loss_q = self.criterion(q_prediction_cat, backup)  # * self.n  # averaged for homogeneity with SAC

        for q in self.q_optimizer_list:
            q.zero_grad()
        loss_q.backward()

        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)

            qs_pi = [q(o, pi) for q in self.model.qs]
            qs_pi_cat = torch.stack(qs_pi, -1)
            ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)
            loss_pi = (alpha_t * logp_pi.unsqueeze(dim=-1) - ave_q).mean()
            self.pi_optimizer.zero_grad()
            loss_pi.backward()

            for q in self.model.qs:
                q.requires_grad_(True)

        for q_optimizer in self.q_optimizer_list:
            q_optimizer.step()

        if update_policy:
            self.pi_optimizer.step()

        if update_policy:
            with torch.no_grad():
                for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        if update_policy:
            self.loss_pi = loss_pi.detach()
        ret_dict = dict(
            loss_actor=self.loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )

        if self.learn_entropy_coef and loss_alpha is not None:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# ========== SHARED BACKBONE REDQ-SAC AGENT ==========
# Optimized for 4GB VRAM: Uses shared CNN backbone across actor + critics


@dataclass(eq=0)
class SharedBackboneREDQSACAgent(TrainingAgent):
    """
    REDQ-SAC Agent optimized for low VRAM GPUs.

    Uses SharedBackboneHybridActorCritic which shares ONE CNN across all networks.
    Key optimization: Features are extracted ONCE and reused for all Q-heads.
    """
    observation_space: type
    action_space: type
    device: str = None
    model_cls: type = core.SharedBackboneHybridActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_entropy: float = 1e-3
    learn_entropy_coef: bool = True
    target_entropy: float = None
    n: int = 2  # number of Q networks (default 2 for VRAM)
    m: int = 2  # number of randomly sampled target networks
    q_updates_per_policy_update: int = 1

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space, n=self.n)
        logging.debug(f" device SharedBackbone-REDQ-SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Optimizer for actor's policy head only (features are fixed/detached for actor)
        self.pi_optimizer = Adam(
            list(self.model.actor.net.parameters()) +
            list(self.model.actor.mu_layer.parameters()) +
            list(self.model.actor.std_net.parameters()) +
            list(self.model.actor.log_std_layer.parameters()),
            lr=self.lr_actor
        )

        # Optimizer for Q-heads AND Encoder (Critic drives representation)
        self._q_params = [p for q in self.model.qs for p in q.parameters()]
        self._encoder_params = list(self.model.actor.cnn.parameters()) + list(self.model.actor.float_mlp.parameters())
        if hasattr(self.model.actor, "fusion_norm"):
            self._encoder_params += list(self.model.actor.fusion_norm.parameters())
        if hasattr(self.model.actor, "fusion_gate"):
            self._encoder_params += list(self.model.actor.fusion_gate.parameters())
        if hasattr(self.model.actor, "context_encoder"):
            self._encoder_params += list(self.model.actor.context_encoder.parameters())
        if hasattr(self.model.actor, "film_generator"):
            self._encoder_params += list(self.model.actor.film_generator.parameters())

        # Keep shared representation updates slower than Q-head updates at high UTD.
        utd_ratio = max(1.0, float(self.q_updates_per_policy_update))
        encoder_lr = min(self.lr_critic / utd_ratio, 3e-5)
        self.q_optimizer = Adam([
            {'params': self._q_params, 'lr': self.lr_critic},
            {'params': self._encoder_params, 'lr': encoder_lr},
        ])

        self.criterion = torch.nn.MSELoss()
        self.loss_pi = torch.zeros((1,), device=device)
        self.loss_q = torch.zeros((1,), device=device)  # Initialize loss_q
        self.i_update = 0

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        self.i_update += 1
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)

        o, a, r, o2, d, _ = batch

        # Get current alpha
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
        else:
            alpha_t = self.alpha_t

        # === Target Q computation (with no_grad) ===
        with torch.no_grad():
            # Use current policy for next action, target critics for evaluation.
            features_o2_curr, _, _ = self.model.forward_features(o2)
            a2, logp_a2, _ = self.model.actor_from_features(features_o2_curr, None)

            # Extract target features for target Q-values
            features_o2_target, _, _ = self.model_target.forward_features(o2)

            sample_idxs = np.random.choice(self.n, self.m, replace=False)
            q_prediction_next_list = [self.model_target.qs[i](features_o2_target, a2) for i in sample_idxs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_t * logp_a2.unsqueeze(dim=-1))

        # === Critic update ===
        # Extract features for current obs (WITH gradients for encoder from critics)
        features_o_critic, _, _ = self.model.forward_features(o)

        q_prediction_list = [q(features_o_critic, a) for q in self.model.qs]
        q_prediction_cat = torch.stack(q_prediction_list, -1)
        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        loss_q = self.criterion(q_prediction_cat, backup)
        self.loss_q = loss_q.detach()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # === Actor update (includes encoder gradients) ===
        loss_alpha = None
        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)

            # Extract features DETACHED for actor (Actor doesn't update encoder)
            with torch.no_grad():
                features_o_actor, _, _ = self.model.forward_features(o)
            pi, logp_pi, _ = self.model.actor_from_features(features_o_actor, None)

            # --- NEW DQDA CLAMP LOGIC ---
            alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
            max_dqda_grad_norm = float(alg_cfg.get("MAX_DQDA_GRAD_NORM", 0.01))

            class DQDAClamp(torch.autograd.Function):
                @staticmethod
                def forward(ctx, action):
                    return action
                @staticmethod
                def backward(ctx, grad_output):
                    grad_norm = grad_output.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    clip_coef = (max_dqda_grad_norm / grad_norm).clamp(max=1.0)
                    return grad_output * clip_coef

            pi_clamped = DQDAClamp.apply(pi)

            qs_pi = [q(features_o_actor, pi_clamped) for q in self.model.qs]
            qs_pi_cat = torch.stack(qs_pi, -1)
            ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)
            loss_pi = (alpha_t * logp_pi.unsqueeze(dim=-1) - ave_q).mean()
            # -----------------------------

            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()

            for q in self.model.qs:
                q.requires_grad_(True)

            # Entropy coefficient update (after actor update)
            if self.learn_entropy_coef:
                # Need fresh logp_pi for alpha gradient
                with torch.no_grad():
                    features_alpha, _, _ = self.model.forward_features(o)
                _, logp_pi_alpha, _ = self.model.actor_from_features(features_alpha, None)
                loss_alpha = -(self.log_alpha * (logp_pi_alpha + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                self.alpha_optimizer.step()

            self.loss_pi = loss_pi.detach()

        # === Polyak averaging ===
        if update_policy:
            with torch.no_grad():
                for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        ret_dict = dict(
            loss_actor=self.loss_pi.detach().item(),
            loss_critic=self.loss_q.detach().item(),
        )

        if self.learn_entropy_coef and loss_alpha is not None:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict


# ============== GRAC: Gradient-Rescaled Actor-Critic Components ==============

class GradientHealthTracker:
    """
    GHAE: Gradient-Health Adaptive Entropy (Novel — part of GRAC algorithm).

    Tracks per-dimension gradient health of the actor's mu_layer and provides
    entropy target boosts for dimensions with persistently vanishing gradients.

    Creates a self-healing feedback loop:
      vanishing gradients → more exploration → actions move from boundaries
      → gradients recover → exploration decreases → exploitation resumes
    """
    def __init__(self, dim_act, ema_decay=0.99, threshold=0.01,
                 boost=0.5, device='cpu'):
        self.dim_act = dim_act
        self.ema_decay = ema_decay
        self.threshold = threshold
        self.boost = boost
        self.grad_health = torch.ones(dim_act, device=device)

    def update(self, mu_layer_grad):
        """Update health scores from mu_layer weight gradient."""
        if mu_layer_grad is None:
            return
        # mu_layer.weight has shape (dim_act, hidden_dim)
        # Per-dimension gradient magnitude (mean across input dims)
        per_dim_mag = mu_layer_grad.abs().mean(dim=1)  # (dim_act,)
        self.grad_health = (self.ema_decay * self.grad_health +
                           (1 - self.ema_decay) * per_dim_mag.detach())

    def get_entropy_boost(self):
        """Compute per-dimension entropy target boost for unhealthy dims."""
        deficit = torch.relu(self.threshold - self.grad_health)
        return self.boost * deficit


# ============== DroQ SAC Agent ==========================

class DroQSACAgent(TrainingAgent):
    """
    DroQ (Dropout Q-functions) SAC Agent for maximum sample efficiency.

    Key features:
    - Uses only 2 Q-networks with Dropout+LayerNorm
    - Supports high UTD (Update-to-Data) ratios (20+)
    - Dropout provides ensemble-like diversity for uncertainty
    - Compatible with shared backbone architecture
    - EWC (Elastic Weight Consolidation) for continual learning across maps
    """
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space,
                 action_space,
                 device,
                 model_cls=core.DroQHybridActorCritic,
                 gamma=0.99,
                 polyak=0.995,
                 alpha=0.2,
                 lr_actor=1e-3,
                 lr_critic=1e-3,
                 lr_entropy=1e-3,
                 learn_entropy_coef=True,
                 target_entropy=None,
                 q_updates_per_policy_update=20,
                 model=None):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_entropy = lr_entropy
        self.learn_entropy_coef = learn_entropy_coef
        self.target_entropy = target_entropy
        self.n = 2  # DroQ always uses 2 Q-networks
        self.m = 2  # Use both for min-Q
        self.q_updates_per_policy_update = q_updates_per_policy_update
        self.alpha_floor = cfg.TMRL_CONFIG.get("ALG", {}).get("ALPHA_FLOOR", 0.08)

        # EWC (Elastic Weight Consolidation) for continual learning
        self.ewc_lambda = cfg.TMRL_CONFIG.get("ALG", {}).get("EWC_LAMBDA", 0.0)
        self._ewc_fisher = {}   # Fisher Information Matrix (diagonal)
        self._ewc_params = {}   # Optimal parameters from previous task
        self._ewc_active = False
        # Auto-load EWC state if it exists
        ewc_path = Path(cfg.WEIGHTS_FOLDER if hasattr(cfg, 'WEIGHTS_FOLDER') else r'C:\Users\felix\TmrlData\weights') / 'ewc_state.pkl'
        if ewc_path.exists() and self.ewc_lambda > 0:
            self.load_ewc_state(str(ewc_path))
            logging.info(f"EWC: Loaded consolidation state from {ewc_path}, lambda={self.ewc_lambda}")

        model = model if model is not None else model_cls(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        # Optimizer for actor's policy head only (features are fixed/detached for actor)
        self.pi_optimizer = Adam([
            {'params': list(self.model.actor.net.parameters()) +
                       list(self.model.actor.mu_layer.parameters()),
             'lr': self.lr_actor},
            {'params': list(self.model.actor.std_net.parameters()) +
                       (list(self.model.actor.log_std_layer.parameters())
                        if hasattr(self.model.actor, 'log_std_layer') else []),
             'lr': self.lr_actor},
        ], lr=self.lr_actor)

        # Optimizer for Q-heads AND Encoder (Critic drives representation)
        # Context encoder gets a lower LR because Q-loss gradients travel through
        # ~15 layers to reach it, becoming noisy. A lower LR prevents noise amplification.
        self._q_params = [p for q in self.model.qs for p in q.parameters()]
        self._encoder_params = list(self.model.actor.cnn.parameters()) + list(self.model.actor.float_mlp.parameters())
        if hasattr(self.model.actor, "fusion_norm"):
            self._encoder_params += list(self.model.actor.fusion_norm.parameters())
        if hasattr(self.model.actor, "fusion_gate"):
            self._encoder_params += list(self.model.actor.fusion_gate.parameters())

        self._context_params = []
        if hasattr(self.model.actor, "context_encoder"):
            self._context_params = list(self.model.actor.context_encoder.parameters())
        if hasattr(self.model.actor, "film_generator"):
            self._context_params += list(self.model.actor.film_generator.parameters())

        # Scale encoder-related learning rates by UTD to prevent representation churn.
        utd_ratio = max(1.0, float(self.q_updates_per_policy_update))
        encoder_lr = min(self.lr_critic / utd_ratio, 3e-5)
        context_lr = min(self.lr_critic / utd_ratio, 3e-5)

        optimizer_groups = [
            {'params': self._q_params, 'lr': self.lr_critic},
            {'params': self._encoder_params, 'lr': encoder_lr},
        ]
        if self._context_params:
            optimizer_groups.append({'params': self._context_params, 'lr': context_lr})
        self.q_optimizer = Adam(optimizer_groups)

        # Use MSELoss so critics can aggressively correct large Q-target errors.
        self.criterion = torch.nn.MSELoss()
        self.loss_pi = torch.zeros((1,), device=device)
        self.loss_q = torch.zeros((1,), device=device)
        self.i_update = 0

        dim_act = action_space.shape[0]
        if self.target_entropy is None:
            self.target_entropy = torch.full((dim_act,), -1.0, device=self.device)
        else:
            per_dim = float(self.target_entropy) / dim_act
            self.target_entropy = torch.full((dim_act,), per_dim, device=self.device)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(dim_act, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.full((dim_act,), float(self.alpha), device=self.device)

        # GRAC: Gradient-Health Adaptive Entropy tracker (configurable from config.json)
        ghae_threshold = cfg.TMRL_CONFIG.get("ALG", {}).get("GHAE_THRESHOLD", 0.02)
        ghae_boost = cfg.TMRL_CONFIG.get("ALG", {}).get("GHAE_BOOST", 2.0)
        self.grad_tracker = GradientHealthTracker(
            dim_act=dim_act, device=self.device,
            threshold=ghae_threshold, boost=ghae_boost
        )

        # ── World Model (RSSM) ─────────────────────────────────────────────
        wm_cfg = cfg.TMRL_CONFIG.get("WORLD_MODEL", {})
        self.wm_enabled = wm_cfg.get("ENABLED", False)
        if self.wm_enabled:
            from tmrl.custom.custom_models import LatentWorldModel
            wm_state_dim = core.EGO_WM_STATE_DIM
            wm_latent = wm_cfg.get("LATENT_DIM", 32)
            wm_gru = wm_cfg.get("GRU_DIM", 128)
            wm_hidden = wm_cfg.get("HIDDEN_DIM", 256)
            wm_kl_free = wm_cfg.get("KL_FREE_NATS", 1.0)
            wm_dyn_scale = wm_cfg.get("DYN_LOSS_SCALE", 1.0)
            wm_rep_scale = wm_cfg.get("REP_LOSS_SCALE", 0.1)
            wm_latent_probe_scale = wm_cfg.get("LATENT_PROBE_SCALE", 0.1)
            wm_anti_collapse_scale = wm_cfg.get("ANTI_COLLAPSE_SCALE", 0.1)
            wm_decoder_latent_use_scale = wm_cfg.get("DECODER_LATENT_USE_SCALE", 0.1)
            wm_decoder_latent_margin_ratio = wm_cfg.get("DECODER_LATENT_MARGIN_RATIO", 0.05)
            self.dynamics = LatentWorldModel(
                state_dim=wm_state_dim, action_dim=dim_act,
                latent_dim=wm_latent, gru_dim=wm_gru,
                hidden_dim=wm_hidden, kl_free_nats=wm_kl_free,
                dyn_loss_scale=wm_dyn_scale,
                rep_loss_scale=wm_rep_scale,
                latent_probe_scale=wm_latent_probe_scale,
                anti_collapse_scale=wm_anti_collapse_scale,
                decoder_latent_use_scale=wm_decoder_latent_use_scale,
                decoder_latent_margin_ratio=wm_decoder_latent_margin_ratio,
            ).to(device)
            self.dynamics_optimizer = Adam(self.dynamics.parameters(),
                                          lr=wm_cfg.get("MODEL_LR", 3e-4))
            self.wm_warmup = wm_cfg.get("WARMUP_STEPS", 3000)
            self.wm_horizon = wm_cfg.get("ROLLOUT_HORIZON", 15)
            self.wm_batch_size = wm_cfg.get("IMAGINED_BATCH_SIZE", 256)
            self.wm_train_steps = 0
            logging.info(f"World Model RSSM: latent={wm_latent}, gru={wm_gru}, "
                         f"horizon={self.wm_horizon}, warmup={self.wm_warmup}")
            self.curiosity_scale = wm_cfg.get("CURIOSITY_SCALE", 0.1)
            self._last_surprise_mean = 0.0  # tracks latest surprise for dynamic alpha floor
            self._last_verifier_trust = 1.0 # conviction score for dynamic alpha floor
            self._last_hgi_actor_health_metrics = {}
            self._adaptive_safety_stats = {}
            self._last_det_skill_transfer_feedback = {}

            # Imagination Actor: learned policy for realistic imagined rollouts
            from tmrl.custom.custom_models import ImaginationActor, RunningMeanStd
            self.imag_actor = ImaginationActor(
                state_dim=wm_state_dim, action_dim=dim_act,
            ).to(device)
            self.imag_actor_optimizer = Adam(self.imag_actor.parameters(), lr=1e-3)
            self.imag_noise_scale = wm_cfg.get("IMAGINATION_NOISE_SCALE", 0.3)
            self.curiosity_reward_clip = wm_cfg.get("CURIOSITY_REWARD_CLIP", 5.0)
            self.curiosity_rms = RunningMeanStd()

    def get_actor(self):
        # Avoid deepcopy-based export for DroQ models:
        # certain parametrized tensors are not deepcopy-compatible in recent torch.
        return self.model.actor

    def _base_alpha_floor_tensor(self, alg_cfg=None):
        alg_cfg = alg_cfg or cfg.TMRL_CONFIG.get("ALG", {})
        dim_act = self.action_space.shape[0]
        floor_all = float(alg_cfg.get("ALPHA_FLOOR", 0.0))
        floor_cfg = alg_cfg.get("ALPHA_FLOOR_PER_DIM", None)
        if floor_cfg is not None:
            floors = [float(v) for v in floor_cfg]
        else:
            floors = [
                float(alg_cfg.get("ALPHA_FLOOR_STEER", floor_all)),
                float(alg_cfg.get("ALPHA_FLOOR_GAS", floor_all)),
                float(alg_cfg.get("ALPHA_FLOOR_BRAKE", floor_all)),
            ]
        if len(floors) < dim_act:
            floors.extend([floor_all] * (dim_act - len(floors)))
        return torch.tensor(floors[:dim_act], device=self.device, dtype=torch.float32)

    def _ensure_entropy_floor_controller(self, base_floor=None, alg_cfg=None):
        alg_cfg = alg_cfg or cfg.TMRL_CONFIG.get("ALG", {})
        base_floor = base_floor if base_floor is not None else self._base_alpha_floor_tensor(alg_cfg)
        controller = getattr(self, "_entropy_floor_controller", None)
        if controller is None:
            controller = HealthGatedFloorController(base_floor, alg_cfg, device=self.device)
            self._entropy_floor_controller = controller
        else:
            controller.update_base(base_floor, alg_cfg)
        return controller

    def _alpha_floor_tensor(self):
        alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
        base_floor = self._base_alpha_floor_tensor(alg_cfg)
        controller = self._ensure_entropy_floor_controller(base_floor, alg_cfg)
        self._last_entropy_floor_diag = controller.diagnostics()
        return controller.get_floor()

    def update_entropy_floor_controller(self, diagnostics):
        alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
        base_floor = self._base_alpha_floor_tensor(alg_cfg)
        controller = self._ensure_entropy_floor_controller(base_floor, alg_cfg)
        diagnostics = {} if diagnostics is None else dict(diagnostics)
        _, diag = controller.step(diagnostics, alg_cfg)
        self._last_entropy_floor_diag = diag
        return diag

    def update_det_skill_transfer_feedback(self, diagnostics):
        alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
        previous = getattr(self, "_last_det_skill_transfer_feedback", {})
        feedback = _compute_det_skill_transfer_feedback(diagnostics, alg_cfg, previous=previous)
        self._last_det_skill_transfer_feedback = feedback
        return feedback

    def _det_skill_transfer_feedback(self):
        feedback = getattr(self, "_last_det_skill_transfer_feedback", {})
        if not feedback:
            feedback = _compute_det_skill_transfer_feedback({}, cfg.TMRL_CONFIG.get("ALG", {}))
            self._last_det_skill_transfer_feedback = feedback
        return dict(feedback)

    def _apply_alpha_floor(self):
        floor = self._alpha_floor_tensor()
        if self.learn_entropy_coef:
            with torch.no_grad():
                log_floor = torch.log(floor.clamp_min(1e-12))
                self.log_alpha.data.copy_(torch.maximum(self.log_alpha.data, log_floor))
            return torch.exp(self.log_alpha.detach()), floor
        return torch.maximum(self.alpha_t, floor), floor

    def _ensure_actor_churn_anchor(self):
        anchor = getattr(self, "_actor_churn_anchor", None)
        needs_init = anchor is None or not hasattr(anchor, "net") or not hasattr(anchor, "mu_layer")
        if needs_init:
            anchor = no_grad(deepcopy(self.model.actor)).to(self.device)
            self._actor_churn_anchor = anchor
        else:
            anchor = no_grad(anchor).to(self.device)
        anchor.eval()
        return anchor

    def _actor_churn_anchor_mu(self, fused, z):
        anchor = self._ensure_actor_churn_anchor()
        with torch.no_grad():
            fused = fused.detach()
            if z is not None:
                actor_input = torch.cat([fused, z.detach()], dim=-1)
            else:
                actor_input = fused
            net_out = anchor.net(actor_input)
            return anchor.mu_layer(net_out)

    def _update_actor_churn_anchor(self, polyak):
        anchor = self._ensure_actor_churn_anchor()
        _ema_update_module(anchor, self.model.actor, polyak)
        anchor.eval()

    def _ensure_critic_churn_anchor(self):
        anchor = getattr(self, "_critic_churn_anchor", None)
        needs_init = anchor is None or len(anchor) != len(self.model.qs)
        if needs_init:
            anchor = no_grad(deepcopy(self.model.qs)).to(self.device)
            self._critic_churn_anchor = anchor
        else:
            anchor = no_grad(anchor).to(self.device)
        anchor.eval()
        return anchor

    def _critic_churn_anchor_q(self, critic_features, action, film):
        anchor = self._ensure_critic_churn_anchor()
        with torch.no_grad():
            critic_features = critic_features.detach()
            action = action.detach()
            film = film.detach() if film is not None else None
            q_anchor_list = [q(critic_features, action, film) for q in anchor]
            return torch.stack(q_anchor_list, -1)

    def _update_critic_churn_anchor(self, polyak):
        anchor = self._ensure_critic_churn_anchor()
        _ema_update_module(anchor, self.model.qs, polyak)
        anchor.eval()

    # ── World Model helpers ──────────────────────────────────────────────

    def _extract_critic_state(self, obs):
        """
        Extract ego-local state for WM/imagination.
        Absolute xyz and absolute track progress are intentionally scrubbed.
        Returns: (B, 24) tensor on self.device.
        """
        if len(obs) == 9:
            speed, gear, rpm, images, lidar, _xyz, _progress, act1, act2 = obs
            B = images.shape[0]
            crash = torch.zeros((B, 1), device=speed.device)
            progress_gain = torch.zeros((B, 1), device=speed.device)
        else:
            speed, gear, rpm, images, lidar, _xyz, _progress, crash, progress_gain, act1, act2 = obs
            B = images.shape[0]

        speed = speed.view(B, -1)
        gear = gear.view(B, -1)
        rpm = rpm.view(B, -1)
        lidar = lidar.view(B, -1)
        crash = crash.view(B, -1)
        progress_gain = progress_gain.view(B, -1)
        return torch.cat((speed, gear, rpm, lidar, crash, progress_gain), dim=-1)  # (B, 24)

    def _train_dynamics(self, o, a, r, o2, context_z=None):
        """
        Train the RSSM world model on real transitions.
        Uses reconstruction + KL divergence loss.
        Args:
            o:  observation tuple (batched)
            a:  actions (B, 3)
            r:  rewards (B,)
            o2: next observation tuple (batched)
            context_z: (B, 64) — PEARL context vector (optional)
        Returns:
            metrics: dict with component losses for logging
        """
        state = self._extract_critic_state(o).detach()
        next_state = self._extract_critic_state(o2).detach()
        reward = r.unsqueeze(-1) if r.dim() == 1 else r  # (B, 1)

        wm_input_metrics = {}
        _add_tensor_stats(wm_input_metrics, "wm_input/state", state)
        _add_tensor_stats(wm_input_metrics, "wm_input/next_state", next_state)
        _add_tensor_stats(wm_input_metrics, "wm_input/action", a)
        _add_tensor_stats(wm_input_metrics, "wm_input/reward", reward)
        wm_input_metrics["bridge/context_z_to_wm_present"] = 1.0 if context_z is not None else 0.0
        wm_input_metrics["bridge/context_z_requires_grad"] = float(context_z.requires_grad) if context_z is not None else 0.0
        if context_z is not None:
            _add_tensor_stats(wm_input_metrics, "wm_input/context_z", context_z)
            _add_tensor_stats(wm_input_metrics, "bridge/context_z", context_z)

        loss, metrics = self.dynamics.train_step(state, a, next_state, reward, context_z=context_z)
        metrics = dict(metrics or {})
        metrics.update(wm_input_metrics)

        self.dynamics_optimizer.zero_grad()
        loss.backward()
        metrics["wm_grad/encoder"] = _module_grad_norm(getattr(self.dynamics, "encoder", None))
        metrics["wm_grad/prior"] = _module_grad_norm(getattr(self.dynamics, "prior", None))
        metrics["wm_grad/posterior"] = _module_grad_norm(getattr(self.dynamics, "posterior", None))
        decoder = getattr(self.dynamics, "decoder", None)
        metrics["wm_grad/decoder"] = _module_grad_norm(getattr(decoder, "state_net", None))
        metrics["wm_grad/reward_head"] = _module_grad_norm(getattr(decoder, "reward_net", None))
        metrics["wm_grad/total"] = _module_grad_norm(self.dynamics)
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 5.0)
        self.dynamics_optimizer.step()

        self.wm_train_steps += 1
        return metrics

    def _extract_prev_actions(self, obs, B=None):
        """
        Extract previous actions (act1, act2) from a batched observation tuple.
        Returns: act1 (B, 3), act2 (B, 3)
        """
        if len(obs) == 9:
            _, _, _, images, _, _, _, act1, act2 = obs
        else:
            _, _, _, images, _, _, _, _, _, act1, act2 = obs
        if B is None:
            B = images.shape[0]
        return act1[:B].view(B, -1).detach(), act2[:B].view(B, -1).detach()

    def _imagined_critic_update(self, o, ctx=None, context_z=None, real_action=None, horizon=None):
        """
        Generate imagined rollouts in latent space and perform Critic gradient updates.
        Uses the RSSM prior to roll forward, decodes to critic state, then
        computes proper 1-step TD targets using the actual Q-network pipeline.
        """
        state = self._extract_critic_state(o).detach()
        B = min(state.shape[0], self.wm_batch_size)
        state = state[:B]
        if context_z is not None:
            context_z = context_z[:B].detach()
        if real_action is not None:
            real_action = real_action[:B].detach()

        # Extract real previous actions from batch to seed action history
        real_act1, real_act2 = self._extract_prev_actions(o, B)  # (B, 3) each

        # Policy function for imagination: use learned ImaginationActor + noise
        noise_scale = getattr(self, 'imag_noise_scale', 0.3)
        def policy_fn(critic_state):
            with torch.no_grad():
                base_action = self.imag_actor(critic_state)
            noise = torch.randn_like(base_action) * noise_scale
            return (base_action + noise).clamp(-1.0, 1.0)

        # Imagine H steps into the future (conditioned on PEARL context)
        horizon = self.wm_horizon if horizon is None else int(horizon)
        with torch.no_grad():
            imag_states, imag_rewards, imag_actions, imag_uncertainties = self.dynamics.imagine(
                state, policy_fn, horizon, self.gamma, context_z=context_z
            )

        # Use the last decoded state + reward for a 1-step TD critic update
        # Take transitions from each horizon step as independent training data
        H = imag_states.shape[0]
        imag_metrics = {}
        imag_metrics["hgi/effective_horizon"] = float(horizon)
        for h in range(H):
            _add_tensor_stats(imag_metrics, f"wm_imag/h{h}_state", imag_states[h])
            _add_tensor_stats(imag_metrics, f"wm_imag/h{h}_action", imag_actions[h])
            _add_tensor_stats(imag_metrics, f"wm_imag/h{h}_reward", imag_rewards[h])
            if imag_uncertainties is not None:
                _add_tensor_stats(imag_metrics, f"wm_imag/h{h}_uncertainty", imag_uncertainties[h])

        imag_metrics["bridge/context_z_to_critic_present"] = 1.0 if context_z is not None else 0.0
        imag_metrics["bridge/context_z_requires_grad"] = float(context_z.requires_grad) if context_z is not None else 0.0
        if context_z is not None:
            _add_tensor_stats(imag_metrics, "bridge/context_z", context_z)

        if H < 2:
            imag_metrics["wm_imagined_steps"] = 0
            return imag_metrics

        # === Build proper action history for critic input ===
        # The critic expects (ego_state_24, act1=a_{t-1}, act2=a_{t-2}, z_64)
        # We must track the 2-step action history through the imagination rollout.
        #
        # At horizon step h, the "previous actions" are:
        #   act1_h = imag_actions[h-1]  (action taken at step h-1)
        #   act2_h = imag_actions[h-2]  (action taken at step h-2)
        #
        # For h=0: act1=real_act1, act2=real_act2 (from the batch)
        # For h=1: act1=imag_actions[0], act2=real_act1
        # For h>=2: act1=imag_actions[h-1], act2=imag_actions[h-2]

        # Build act1_history and act2_history for each horizon step (H, B, 3)
        act1_history = []
        act2_history = []
        for h in range(H):
            if h == 0:
                act1_history.append(real_act1)
                act2_history.append(real_act2)
            elif h == 1:
                act1_history.append(imag_actions[0])
                act2_history.append(real_act1)
            else:
                act1_history.append(imag_actions[h - 1])
                act2_history.append(imag_actions[h - 2])

        act1_stack = torch.stack(act1_history, dim=0)  # (H, B, 3)
        act2_stack = torch.stack(act2_history, dim=0)  # (H, B, 3)

        # Flatten horizon: treat each (s_t, a_t, r_t, s_{t+1}) as a transition
        s_flat = imag_states[:-1].reshape(-1, self.dynamics.state_dim)    # ((H-1)*B, state_dim)
        a_flat = imag_actions[:-1].reshape(-1, self.dynamics.action_dim)  # ((H-1)*B, 3)
        r_flat = imag_rewards[:-1].reshape(-1, 1)                         # ((H-1)*B, 1)
        ns_flat = imag_states[1:].reshape(-1, self.dynamics.state_dim)    # ((H-1)*B, state_dim)
        u_t_flat = imag_uncertainties[:-1].reshape(-1, 1)                 # ((H-1)*B, 1)

        # Previous actions for current states (s_flat) and next states (ns_flat)
        act1_s = act1_stack[:-1].reshape(-1, self.dynamics.action_dim)   # ((H-1)*B, 3)
        act2_s = act2_stack[:-1].reshape(-1, self.dynamics.action_dim)   # ((H-1)*B, 3)
        act1_ns = act1_stack[1:].reshape(-1, self.dynamics.action_dim)   # ((H-1)*B, 3)
        act2_ns = act2_stack[1:].reshape(-1, self.dynamics.action_dim)   # ((H-1)*B, 3)

        _add_tensor_stats(imag_metrics, "bridge/state_real", state)
        _add_tensor_stats(imag_metrics, "bridge/state_imag", s_flat)
        if state.shape[-1] == s_flat.shape[-1]:
            imag_metrics["bridge/state_distribution_gap"] = (
                state.detach().mean(dim=0) - s_flat.detach().mean(dim=0)
            ).abs().mean().item()
        if real_action is not None:
            _add_tensor_stats(imag_metrics, "bridge/action_real", real_action)
        _add_tensor_stats(imag_metrics, "bridge/action_imag", a_flat)
        _add_tensor_stats(imag_metrics, "bridge/act1_real", real_act1)
        _add_tensor_stats(imag_metrics, "bridge/act2_real", real_act2)
        imag_metrics["bridge/act1_vs_imag_action_abs_diff"] = (act1_s.detach() - a_flat.detach()).abs().mean().item()
        imag_metrics["bridge/act2_vs_act1_abs_diff"] = (act2_s.detach() - act1_s.detach()).abs().mean().item()

        # Compute target Q-values using imagined next states
        with torch.no_grad():
            a2 = self.imag_actor(ns_flat)
            a2_noise = torch.randn_like(a2) * noise_scale
            a2 = (a2 + a2_noise).clamp(-1.0, 1.0)
            # Build critic_floats: the Q-heads take 30-dim (24 state + 3 act1 + 3 act2)
            # Use proper action history for next-state critic input
            critic_ns = torch.cat([ns_flat, act1_ns, act2_ns], dim=-1)

            # Use real PEARL context for Q-network input during imagination
            if context_z is not None:
                z_imag = context_z.unsqueeze(0).expand(H-1, -1, -1).reshape(-1, 64)
                critic_ns = torch.cat([critic_ns, z_imag], dim=-1)
            else:
                z_neutral = torch.zeros(ns_flat.shape[0], 64, device=self.device)
                critic_ns = torch.cat([critic_ns, z_neutral], dim=-1)

            # Get FiLM params — use zeros (neutral modulation) for imagined data
            if hasattr(self.model, 'film_generator'):
                z_for_film = torch.zeros(ns_flat.shape[0], 64, device=self.device)
                film_params_neutral = self.model.film_generator(z_for_film)
            else:
                film_params_neutral = None

            # Target Q from target network
            q_next_list = [q(critic_ns, a2, film_params_neutral) for q in self.model_target.qs]
            q_next_cat = torch.stack(q_next_list, -1)
            min_q_next = torch.min(q_next_cat, dim=1)[0]  # (N,)

            target_q = r_flat.squeeze(-1) + self.gamma * min_q_next  # (N,)

        # Critic update on imagined transitions — use proper action history
        critic_s = torch.cat([s_flat, act1_s, act2_s], dim=-1)

        if context_z is not None:
            z_imag_grad = context_z.unsqueeze(0).expand(H-1, -1, -1).reshape(-1, 64)
            critic_s = torch.cat([critic_s, z_imag_grad], dim=-1)
        else:
            if hasattr(self.model, 'context_encoder'):
                z_neutral_grad = torch.zeros(s_flat.shape[0], 64, device=self.device)
                critic_s = torch.cat([critic_s, z_neutral_grad], dim=-1)

        if hasattr(self.model, 'film_generator'):
            z_for_film_grad = torch.zeros(s_flat.shape[0], 64, device=self.device)
            film_params_grad = self.model.film_generator(z_for_film_grad)
        else:
            film_params_grad = None

        q_pred_list = [q(critic_s, a_flat, film_params_grad) for q in self.model.qs]
        q_pred_cat = torch.stack(q_pred_list, -1)  # (N, 2)
        target_q_expanded = target_q.unsqueeze(-1).expand_as(q_pred_cat)
        with torch.no_grad():
            q_imag = q_pred_cat.detach()
            target_q_detached = target_q.detach()
            imag_td_error = q_imag - target_q_expanded.detach()
            imag_metrics["bridge/q_imag_mean"] = q_imag.mean().item()
            imag_metrics["bridge/q_imag_std"] = q_imag.std(unbiased=False).item()
            imag_metrics["bridge/target_q_imag_mean"] = target_q_detached.mean().item()
            imag_metrics["bridge/target_q_imag_std"] = target_q_detached.std(unbiased=False).item()
            imag_metrics["bridge/imag_td_error_mean"] = imag_td_error.mean().item()
            imag_metrics["bridge/imag_td_error_abs_mean"] = imag_td_error.abs().mean().item()

        loss_unreduced = F.mse_loss(q_pred_cat, target_q_expanded, reduction='none')

        # === Verifier-Gated Imagination ===
        # Compute trust metric m_t from uncertainty u_t
        lambda_trust = cfg.TMRL_CONFIG.get("ALG", {}).get("VERIFIER_LAMBDA", 10.0)
        m_t = torch.exp(-lambda_trust * u_t_flat)  # ((H-1)*B, 1)
        m_t_expanded = m_t.expand_as(loss_unreduced)
        _add_tensor_stats(imag_metrics, "wm/verifier_uncertainty", u_t_flat)
        _add_tensor_stats(imag_metrics, "wm/verifier_trust", m_t)
        imag_metrics["wm/verifier_trust_saturation_low"] = (m_t.detach() < 0.05).float().mean().item()
        imag_metrics["wm/verifier_trust_saturation_high"] = (m_t.detach() > 0.95).float().mean().item()
        _add_tensor_stats(imag_metrics, "bridge/trust", m_t)
        imag_metrics["bridge/trust_saturation_low"] = (m_t.detach() < 0.05).float().mean().item()
        imag_metrics["bridge/trust_saturation_high"] = (m_t.detach() > 0.95).float().mean().item()

        # Weight loss by trust metric
        loss_q_imagined = (loss_unreduced * m_t_expanded).mean()

        self.q_optimizer.zero_grad()
        loss_q_imagined.backward()
        # Clip actor gradients loosely to prevent extreme spikes from deterministic pull
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self._q_params, 1.0)
        self.q_optimizer.step()

        imag_metrics.update({
            "wm_imagined_steps": H * B,
            "wm_imagined_q_loss": loss_q_imagined.item(),
            "verifier_uncertainty_mean": u_t_flat.mean().item(),
            "verifier_trust_mt_mean": m_t.mean().item(),
            "grad_norm_critic": actor_grad_norm.item(),
        })
        return imag_metrics

    # ── EWC: Elastic Weight Consolidation ────────────────────────────────

    def consolidate_task(self, memory=None, n_samples=2000):
        """
        Compute Fisher Information Matrix for the current task.
        Call this BEFORE switching to a new map.

        The Fisher matrix captures which parameters are most important
        for the current task. During future training, deviations from
        these parameters are penalized proportionally to their importance.
        """
        logging.info("EWC: Computing Fisher Information Matrix...")
        self.model.eval()

        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        # Use replay buffer if available, else use random noise
        if memory is not None:
            n_batches = max(1, n_samples // 64)
            for _ in range(n_batches):
                try:
                    batch = memory.sample()
                    # Forward pass to get loss
                    img_ctx = None
                    if len(batch) == 8:
                        o, a, r, o2, d, _, ctx_full, img_ctx_full = batch
                        ctx = ctx_full[:, :-1, :]
                        img_ctx = img_ctx_full[:, :-1, :, :, :]
                    elif len(batch) == 7:
                        o, a, r, o2, d, _, ctx_full = batch
                        ctx = ctx_full[:, :-1, :]
                    else:
                        o, a, r, o2, d, _ = batch
                        ctx = None

                    uses_context = hasattr(self.model, 'context_encoder') and self.model.context_encoder is not None
                    if uses_context and ctx is not None:
                        fused, critic, film, _ = self.model.forward_features(o, ctx, img_context=img_ctx)
                    else:
                        fused, critic, film, _ = self.model.forward_features(o)

                    q_list = [q(fused, a, film) for q in self.model.qs]
                    # Use Q-values as proxy for task importance
                    for q_val in q_list:
                        self.model.zero_grad()
                        q_val.mean().backward(retain_graph=True)
                        for name, param in self.model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                fisher[name] += param.grad.data.pow(2) / n_batches
                except Exception as e:
                    logging.warning(f"EWC: Skipping batch due to: {e}")
                    continue

        # Normalize and store
        for name in fisher:
            fisher[name] = fisher[name].clamp(max=100.0)  # prevent extreme values

        self._ewc_fisher = fisher
        self._ewc_params = {name: param.data.clone()
                           for name, param in self.model.named_parameters()
                           if param.requires_grad}
        self._ewc_active = True

        n_params = sum(f.numel() for f in fisher.values())
        mean_fisher = sum(f.mean().item() for f in fisher.values()) / max(len(fisher), 1)
        logging.info(f"EWC: Consolidated {n_params:,} parameters, mean Fisher={mean_fisher:.4f}")
        self.model.train()

    def _ewc_loss(self):
        """Compute EWC penalty: Σ F_i * (θ_i - θ*_i)²"""
        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self._ewc_fisher and name in self._ewc_params:
                fisher = self._ewc_fisher[name]
                optimal = self._ewc_params[name]
                loss += (fisher * (param - optimal).pow(2)).sum()
        return loss

    def save_ewc_state(self, path=None):
        """Save Fisher matrix + optimal params to disk."""
        if path is None:
            path = str(Path(r'C:\Users\felix\TmrlData\weights') / 'ewc_state.pkl')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            'fisher': {k: v.cpu() for k, v in self._ewc_fisher.items()},
            'params': {k: v.cpu() for k, v in self._ewc_params.items()},
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logging.info(f"EWC: Saved consolidation state to {path}")

    def load_ewc_state(self, path=None):
        """Load Fisher matrix + optimal params from disk."""
        if path is None:
            path = str(Path(r'C:\Users\felix\TmrlData\weights') / 'ewc_state.pkl')
        if not Path(path).exists():
            logging.warning(f"EWC: No state file at {path}")
            return
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self._ewc_fisher = {k: v.to(self.device) for k, v in state['fisher'].items()}
        self._ewc_params = {k: v.to(self.device) for k, v in state['params'].items()}
        self._ewc_active = True
        logging.info(f"EWC: Loaded consolidation state ({len(self._ewc_fisher)} params)")


    def train(self, batch):
        """
        DroQ training step.
        Same as SharedBackboneREDQSAC but with dropout enabled during training.
        Supports context-augmented batches for context-based meta-learning.
        """
        # Backward compatibility for checkpoints (init logic bypassed)
        if not hasattr(self, "q_updates_per_policy_update"):
             self.q_updates_per_policy_update = cfg.TMRL_CONFIG["ALG"]["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]
        if not hasattr(self, "det_reg_lambda"):
             self.det_reg_lambda = cfg.TMRL_CONFIG.get("ALG", {}).get("DET_REG_LAMBDA", 0.03)
        # Entropy is now fully managed by the Target Entropy controller (Auto-Annealer).
        # We removed alpha_floor here to perfectly align the SAC optimizer.

        # Lazy-init World Model for checkpoints loaded via pickle (bypasses __init__)
        if not hasattr(self, "wm_enabled"):
            wm_cfg = cfg.TMRL_CONFIG.get("WORLD_MODEL", {})
            self.wm_enabled = wm_cfg.get("ENABLED", False)
            if self.wm_enabled:
                from tmrl.custom.custom_models import LatentWorldModel
                dim_act = self.action_space.shape[0]
                wm_state_dim = core.EGO_WM_STATE_DIM
                wm_latent = wm_cfg.get("LATENT_DIM", 32)
                wm_gru = wm_cfg.get("GRU_DIM", 128)
                wm_hidden = wm_cfg.get("HIDDEN_DIM", 256)
                wm_kl_free = wm_cfg.get("KL_FREE_NATS", 1.0)
                wm_dyn_scale = wm_cfg.get("DYN_LOSS_SCALE", 1.0)
                wm_rep_scale = wm_cfg.get("REP_LOSS_SCALE", 0.1)
                wm_latent_probe_scale = wm_cfg.get("LATENT_PROBE_SCALE", 0.1)
                wm_anti_collapse_scale = wm_cfg.get("ANTI_COLLAPSE_SCALE", 0.1)
                wm_decoder_latent_use_scale = wm_cfg.get("DECODER_LATENT_USE_SCALE", 0.1)
                wm_decoder_latent_margin_ratio = wm_cfg.get("DECODER_LATENT_MARGIN_RATIO", 0.05)
                self.dynamics = LatentWorldModel(
                    state_dim=wm_state_dim, action_dim=dim_act,
                    latent_dim=wm_latent, gru_dim=wm_gru,
                    hidden_dim=wm_hidden, kl_free_nats=wm_kl_free,
                    dyn_loss_scale=wm_dyn_scale,
                    rep_loss_scale=wm_rep_scale,
                    latent_probe_scale=wm_latent_probe_scale,
                    anti_collapse_scale=wm_anti_collapse_scale,
                    decoder_latent_use_scale=wm_decoder_latent_use_scale,
                    decoder_latent_margin_ratio=wm_decoder_latent_margin_ratio,
                ).to(self.device)
                self.dynamics_optimizer = Adam(self.dynamics.parameters(),
                                              lr=wm_cfg.get("MODEL_LR", 3e-4))
                self.wm_warmup = wm_cfg.get("WARMUP_STEPS", 3000)
                self.wm_horizon = wm_cfg.get("ROLLOUT_HORIZON", 15)
                self.wm_batch_size = wm_cfg.get("IMAGINED_BATCH_SIZE", 256)
                self.wm_train_steps = 0
                logging.info(f"World Model RSSM (lazy init): latent={wm_latent}, gru={wm_gru}")
                self.curiosity_scale = wm_cfg.get("CURIOSITY_SCALE", 0.1)
                self._last_surprise_mean = 0.0
                self._last_verifier_trust = 1.0
                self._last_hgi_actor_health_metrics = {}
                self._adaptive_safety_stats = {}
                self._last_det_skill_transfer_feedback = {}

        # Lazy-init ImaginationActor for checkpoints that predate this feature
        if self.wm_enabled and not hasattr(self, 'imag_actor'):
            from tmrl.custom.custom_models import ImaginationActor, RunningMeanStd
            wm_cfg = cfg.TMRL_CONFIG.get("WORLD_MODEL", {})
            dim_act = self.action_space.shape[0]
            self.imag_actor = ImaginationActor(
                state_dim=core.EGO_WM_STATE_DIM, action_dim=dim_act,
            ).to(self.device)
            self.imag_actor_optimizer = Adam(self.imag_actor.parameters(), lr=1e-3)
            self.imag_noise_scale = wm_cfg.get("IMAGINATION_NOISE_SCALE", 0.3)
            self.curiosity_reward_clip = wm_cfg.get("CURIOSITY_REWARD_CLIP", 5.0)
            self.curiosity_rms = RunningMeanStd()
            logging.info("ImaginationActor lazy-initialized for existing checkpoint")

        # Always refresh curiosity config from live config (so config.json changes
        # take effect on trainer restart without needing RESET_TRAINING=true)
        if self.wm_enabled:
            wm_cfg_live = cfg.TMRL_CONFIG.get("WORLD_MODEL", {})
            self.curiosity_scale = wm_cfg_live.get("CURIOSITY_SCALE", 0.1)
            self.curiosity_reward_clip = wm_cfg_live.get("CURIOSITY_REWARD_CLIP", 5.0)
            if hasattr(self, "dynamics"):
                prior_param_ids = {
                    id(param)
                    for group in self.dynamics_optimizer.param_groups
                    for param in group.get("params", [])
                }
                if hasattr(self.dynamics, "ensure_latent_probe"):
                    self.dynamics.ensure_latent_probe()
                    new_probe_params = [
                        param for param in self.dynamics.latent_probe.parameters()
                        if id(param) not in prior_param_ids
                    ]
                    if new_probe_params:
                        self.dynamics_optimizer.add_param_group({"params": new_probe_params})
                        logging.info("World Model latent probe backfilled into optimizer")
                self.dynamics.kl_free_nats = wm_cfg_live.get("KL_FREE_NATS", getattr(self.dynamics, "kl_free_nats", 1.0))
                self.dynamics.dyn_loss_scale = wm_cfg_live.get("DYN_LOSS_SCALE", getattr(self.dynamics, "dyn_loss_scale", 1.0))
                self.dynamics.rep_loss_scale = wm_cfg_live.get("REP_LOSS_SCALE", getattr(self.dynamics, "rep_loss_scale", 0.1))
                self.dynamics.latent_probe_scale = wm_cfg_live.get(
                    "LATENT_PROBE_SCALE",
                    getattr(self.dynamics, "latent_probe_scale", 0.1),
                )
                self.dynamics.anti_collapse_scale = wm_cfg_live.get(
                    "ANTI_COLLAPSE_SCALE",
                    getattr(self.dynamics, "anti_collapse_scale", 0.1),
                )
                self.dynamics.decoder_latent_use_scale = wm_cfg_live.get(
                    "DECODER_LATENT_USE_SCALE",
                    getattr(self.dynamics, "decoder_latent_use_scale", 0.1),
                )
                self.dynamics.decoder_latent_margin_ratio = wm_cfg_live.get(
                    "DECODER_LATENT_MARGIN_RATIO",
                    getattr(self.dynamics, "decoder_latent_margin_ratio", 0.05),
                )

        if not hasattr(self, "_adaptive_safety_stats"):
            self._adaptive_safety_stats = {}
        if not hasattr(self, "_last_det_skill_transfer_feedback"):
            self._last_det_skill_transfer_feedback = {}

        self.i_update += 1
        # DroQ uses high UTD ratio
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)
        det_skill_feedback = self._det_skill_transfer_feedback()

        # Unpack batch - support context-augmented (7 or 8 elements) and standard (6 elements)
        img_ctx = None
        img_ctx_next = None
        if len(batch) == 8:
            o, a, r, o2, d, _, ctx_full, img_ctx_full = batch
            ctx = ctx_full[:, :-1, :]
            ctx_next = ctx_full[:, 1:, :]
            img_ctx = img_ctx_full[:, :-1, :, :, :]
            img_ctx_next = img_ctx_full[:, 1:, :, :, :]
        elif len(batch) == 7:
            o, a, r, o2, d, _, ctx_full = batch
            ctx = ctx_full[:, :-1, :]
            ctx_next = ctx_full[:, 1:, :]
        else:
            o, a, r, o2, d, _ = batch
            ctx = None
            ctx_next = None

        # Force update learning rates and target entropy from live config on every trainer restart.
        # Uses a versioned flag so bumping the version forces re-application on existing checkpoints.
        _OVERRIDE_VERSION = 5  # Bump this to force re-apply on next restart
        if getattr(self, "_override_version", 0) < _OVERRIDE_VERSION:
            import logging
            alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
            new_lr_actor = 1e-4
            new_lr_critic = 1e-4
            for param_group in self.pi_optimizer.param_groups:
                param_group['lr'] = new_lr_actor
            for param_group in self.q_optimizer.param_groups:
                param_group['lr'] = new_lr_critic
            # Fix GAMMA baked in checkpoint
            self.gamma = 0.99
            # Refresh target entropy from live config.json (prevents entropy death spiral)
            cfg_target = alg_cfg.get("TARGET_ENTROPY", None)
            if cfg_target is not None:
                dim_act = self.action_space.shape[0]
                per_dim = float(cfg_target) / dim_act
                self.target_entropy = torch.full((dim_act,), per_dim, device=self.device)
                logging.info(f" === OVERRIDE: TARGET_ENTROPY={per_dim:.2f} per dim (total={cfg_target}) ===")
            # Refresh GHAE params from live config
            ghae_threshold = alg_cfg.get("GHAE_THRESHOLD", 0.02)
            ghae_boost = alg_cfg.get("GHAE_BOOST", 2.0)
            if hasattr(self, 'grad_tracker'):
                self.grad_tracker.threshold = ghae_threshold
                self.grad_tracker.boost = ghae_boost
                logging.info(f" === OVERRIDE: GHAE(threshold={ghae_threshold}, boost={ghae_boost}) ===")
            self.det_reg_lambda = float(alg_cfg.get("DET_REG_LAMBDA", 0.03))
            logging.info(f" === OVERRIDE: DET_REG_LAMBDA={self.det_reg_lambda} ===")
            alpha_floor_t = self._alpha_floor_tensor()
            if alpha_floor_t.numel() >= 3:
                logging.info(
                    f" === OVERRIDE: ALPHA_FLOOR(steer={alpha_floor_t[0].item():.4f}, "
                    f"gas={alpha_floor_t[1].item():.4f}, brake={alpha_floor_t[2].item():.4f}) ==="
            )

            # Refresh Polyak from live config
            new_polyak = alg_cfg.get("POLYAK", 0.995)
            self.polyak = new_polyak

            logging.info(f" === OVERRIDE: LR(Actor={new_lr_actor}, Critic={new_lr_critic}), GAMMA=0.99, POLYAK={new_polyak} ===")
            self._override_version = _OVERRIDE_VERSION

        # Ensure Q-networks are in training mode (dropout active)
        self.model.train()

        # Get current alpha
        alpha_t, alpha_floor_t = self._apply_alpha_floor()

        # Check if model supports FiLM context (None = Vanilla baseline)
        uses_context = hasattr(self.model, 'context_encoder') and self.model.context_encoder is not None

        # === Target Q computation (with no_grad, dropout active) ===
        with torch.no_grad():
            self.model_target.train()
            self.model.train()

            # BUG 1 FIX: Use CURRENT policy to get next action a2
            if uses_context and ctx_next is not None:
                fused_o2_curr, _, film_o2_curr, z_o2 = self.model.forward_features(o2, ctx_next, img_context=img_ctx_next)
            else:
                fused_o2_curr, _, film_o2_curr, z_o2 = self.model.forward_features(o2)
            a2, logp_a2, _ = self.model.actor_from_features(fused_o2_curr, film_o2_curr, z=z_o2)
            self.model.train()        # Re-enable dropout

            # Now evaluate the Q-value of that action using the TARGET network
            if uses_context and ctx_next is not None:
                _, critic_o2_tgt, film_o2_tgt, _ = self.model_target.forward_features(o2, ctx_next, img_context=img_ctx_next)
            else:
                _, critic_o2_tgt, film_o2_tgt, _ = self.model_target.forward_features(o2)

            # Use both Q-networks for min-Q
            q_prediction_next_list = [q(critic_o2_tgt, a2, film_o2_tgt) for q in self.model_target.qs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            alpha_scalar = alpha_t.mean()

            # === Curiosity bonus: reward novel states the WM hasn't seen ===
            r_augmented = r
            if self.wm_enabled and self.wm_train_steps > self.wm_warmup:
                state_cur = self._extract_critic_state(o).detach()
                state_nxt = self._extract_critic_state(o2).detach()
                surprise = self.dynamics.compute_surprise(state_cur, a, state_nxt)  # (B,)
                # Normalize surprise with running stats for scale consistency
                self.curiosity_rms.update(surprise)
                surprise_norm = self.curiosity_rms.normalize(surprise)
                clip_val = getattr(self, 'curiosity_reward_clip', 5.0)
                surprise_norm = surprise_norm.clamp(-clip_val, clip_val)
                curiosity_bonus = self.curiosity_scale * surprise_norm
                r_augmented = r + curiosity_bonus

            backup = r_augmented.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_scalar * logp_a2.unsqueeze(dim=-1))

        # === Critic update (with dropout) ===
        self.model.train()  # Ensure dropout is active
        if uses_context and ctx is not None:
            _, critic_o_curr, film_o_critic, z_critic = self.model.forward_features(o, ctx, img_context=img_ctx)
        else:
            _, critic_o_curr, film_o_critic, z_critic = self.model.forward_features(o)

        q_prediction_list = [q(critic_o_curr, a, film_o_critic) for q in self.model.qs]
        q_prediction_cat = torch.stack(q_prediction_list, -1)
        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        loss_q = self.criterion(q_prediction_cat, backup)

        alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})

        # === Critic Churn Regularization (CHAIN value side) ===
        # Penalize rapid Q-output drift against a slow EMA copy of the Q heads.
        # This targets value churn without changing TD targets, replay learning,
        # world-model architecture, or actor/critic network structure.
        q_churn_enabled = bool(alg_cfg.get("CRITIC_CHURN_REG_ENABLED", True))
        q_churn_lambda = float(alg_cfg.get("CRITIC_CHURN_REG_LAMBDA", 0.01))
        q_churn_polyak = float(alg_cfg.get("CRITIC_CHURN_ANCHOR_POLYAK", 0.995))
        loss_q_churn = torch.zeros((), device=loss_q.device)
        loss_q_churn_weighted = torch.zeros((), device=loss_q.device)
        q_churn_anchor_abs_diff = 0.0
        if q_churn_enabled and q_churn_lambda > 0.0:
            q_anchor_cat = self._critic_churn_anchor_q(critic_o_curr, a, film_o_critic)
            was_training = self.model.training
            self.model.eval()
            try:
                q_churn_current_list = [q(critic_o_curr, a, film_o_critic) for q in self.model.qs]
                q_churn_current_cat = torch.stack(q_churn_current_list, -1)
            finally:
                if was_training:
                    self.model.train()
            loss_q_churn, loss_q_churn_weighted = _compute_critic_churn_loss(
                q_churn_current_cat,
                q_anchor_cat,
                enabled=q_churn_enabled,
                weight=q_churn_lambda,
            )
            loss_q = loss_q + loss_q_churn_weighted
            with torch.no_grad():
                q_churn_anchor_abs_diff = (q_churn_current_cat.detach() - q_anchor_cat).abs().mean().item()

        # Keep the critic action-sensitive so the actor receives a usable dQ/da signal.
        # A low TD loss is not enough if Q is nearly constant over nearby actions.
        q_action_sensitivity = torch.zeros((), device=loss_q.device)
        q_action_sensitivity_loss = torch.zeros((), device=loss_q.device)
        q_action_sensitivity_loss_raw = torch.zeros((), device=loss_q.device)
        q_action_sensitivity_gap = torch.zeros((), device=loss_q.device)
        q_action_sensitivity_drive = torch.zeros((), device=loss_q.device)
        q_action_std_floor = float(alg_cfg.get("Q_ACTION_STD_FLOOR", 0.05))
        q_action_sens_lambda = float(alg_cfg.get("Q_ACTION_SENSITIVITY_LAMBDA", 1.0))
        q_action_sens_lambda_max = float(
            alg_cfg.get(
                "Q_ACTION_SENSITIVITY_LAMBDA_MAX",
                2.0 if q_action_sens_lambda > 0.0 else 0.0,
            )
        )
        q_action_sens_loss_type = str(
            alg_cfg.get("Q_ACTION_SENSITIVITY_LOSS_TYPE", "huber")
        ).lower().strip()
        q_action_sens_huber_beta = float(
            alg_cfg.get("Q_ACTION_SENSITIVITY_HUBER_BETA", q_action_std_floor)
        )
        q_action_sens_ramp_power = float(
            alg_cfg.get("Q_ACTION_SENSITIVITY_RAMP_POWER", 1.0)
        )
        q_action_sensitivity_lambda_eff = 0.0
        q_action_sensitivity_loss_type_code = _Q_ACTION_SENSITIVITY_LOSS_TYPE_CODES.get(
            q_action_sens_loss_type,
            _Q_ACTION_SENSITIVITY_LOSS_TYPE_CODES["huber"],
        )
        q_action_noise_std = float(alg_cfg.get("Q_ACTION_SENSITIVITY_NOISE_STD", 0.15))
        q_action_probe_count = max(2, int(alg_cfg.get("Q_ACTION_SENSITIVITY_PROBES", 4)))
        if q_action_sens_lambda_max > 0.0 and q_action_std_floor > 0.0:
            action_limit = float(getattr(self.model.actor, "act_limit", 1.0))
            action_base = a.detach()
            probe_q_values = []
            was_training = self.model.training
            self.model.eval()
            try:
                for _ in range(q_action_probe_count):
                    action_probe = (
                        action_base + torch.randn_like(action_base) * q_action_noise_std
                    ).clamp(-action_limit, action_limit)
                    qs_probe = [q(critic_o_curr, action_probe, film_o_critic) for q in self.model.qs]
                    probe_q_values.append(torch.min(torch.stack(qs_probe, -1), dim=-1)[0])
            finally:
                if was_training:
                    self.model.train()
            probe_q_cat = torch.stack(probe_q_values, dim=0)
            q_action_sensitivity = probe_q_cat.std(dim=0, unbiased=False).mean()
            q_action_sens_reg = _compute_q_action_sensitivity_regularizer(
                q_action_sensitivity,
                floor=q_action_std_floor,
                base_weight=q_action_sens_lambda,
                max_weight=q_action_sens_lambda_max,
                loss_type=q_action_sens_loss_type,
                huber_beta=q_action_sens_huber_beta,
                ramp_power=q_action_sens_ramp_power,
            )
            q_action_sensitivity_loss_raw = q_action_sens_reg["raw_loss"]
            q_action_sensitivity_loss = q_action_sens_reg["weighted_loss"]
            q_action_sensitivity_gap = q_action_sens_reg["gap"]
            q_action_sensitivity_drive = q_action_sens_reg["drive"]
            q_action_sensitivity_lambda_eff = q_action_sens_reg["effective_weight"]
            q_action_sensitivity_loss_type_code = q_action_sens_reg["loss_type_code"]
            q_action_sens_huber_beta = q_action_sens_reg["huber_beta"]
            loss_q = loss_q + q_action_sensitivity_loss

        # EWC: penalize deviation from previous task's optimal params
        loss_ewc = torch.zeros(1, device=loss_q.device)
        if self._ewc_active and self.ewc_lambda > 0:
            loss_ewc = self._ewc_loss()
            loss_q = loss_q + self.ewc_lambda * loss_ewc

        # === Variational KL divergence loss (PEARL) ===
        loss_kl = torch.zeros(1, device=loss_q.device)
        if uses_context and ctx is not None and hasattr(self.model.context_encoder, 'last_kl_div'):
            loss_kl = self.model.context_encoder.last_kl_div
            # β-VAE weighting: small enough not to overwhelm RL signal
            kl_beta = cfg.TMRL_CONFIG.get("ALG", {}).get("KL_BETA", 0.05)
            # Free nats: allow a minimum KL without penalty to prevent posterior collapse.
            # Without this, the KL term pushes q(z|c) → N(0,I), making z uninformative.
            kl_free_nats = cfg.TMRL_CONFIG.get("ALG", {}).get("KL_FREE_NATS", 1.0)
            loss_kl_clipped = torch.clamp(loss_kl - kl_free_nats, min=0.0)
            loss_q = loss_q + kl_beta * loss_kl_clipped

        self.loss_q = loss_q.detach()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        # Per-module gradient clipping: context encoder gets tighter clip
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self._q_params, 1.0)
        torch.nn.utils.clip_grad_norm_(self._encoder_params, 1.0)
        if self._context_params:
            torch.nn.utils.clip_grad_norm_(self._context_params, 0.5)
        self.q_optimizer.step()
        if q_churn_enabled and q_churn_lambda > 0.0:
            self._update_critic_churn_anchor(q_churn_polyak)

        # === Actor update ===
        loss_alpha = None
        effective_target_entropy_diag = None
        actor_bridge_metrics = {}
        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)

            with torch.no_grad():
                if uses_context and ctx is not None:
                    fused_o_actor, critic_o_actor, film_o_actor, z_actor = self.model.forward_features(o, ctx, img_context=img_ctx)
                else:
                    fused_o_actor, critic_o_actor, film_o_actor, z_actor = self.model.forward_features(o)

            pi, logp_pi, logp_per_dim, mu_pre_stochastic = self.model.actor_from_features(fused_o_actor, film_o_actor, z=z_actor, return_pretanh_mu=True)

            alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
            max_dqda_grad_norm = float(alg_cfg.get("MAX_DQDA_GRAD_NORM", 0.01))

            # === Fix #1: True dqda Gradient Clamp ===
            # Acts as a firewall between the Critic and the Actor.
            # Intercepts and clamps only the gradient flowing out of the Q-networks.
            class DQDAClamp(torch.autograd.Function):
                @staticmethod
                def forward(ctx, action):
                    return action
                @staticmethod
                def backward(ctx, grad_output):
                    grad_norm = grad_output.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    clip_coef = (max_dqda_grad_norm / grad_norm).clamp(max=1.0)
                    return grad_output * clip_coef

            pi_clamped = DQDAClamp.apply(pi)

            # Feed the CLAMPED action to the Q-networks
            qs_pi = [q(critic_o_actor, pi_clamped, film_o_actor) for q in self.model.qs]
            qs_pi_cat = torch.stack(qs_pi, -1)
            min_q_pi = torch.min(qs_pi_cat, dim=1, keepdim=True)[0]
            std_q_pi = torch.std(qs_pi_cat, dim=1, keepdim=True)

            # Small SAC variant: uncertainty bonus and KL brake (ensemble disagreement)
            unc_bonus = alg_cfg.get("UNCERTAINTY_BONUS", 0.0)
            kl_brake = alg_cfg.get("KL_BRAKE", 0.0)
            q_target_pi = min_q_pi + (unc_bonus - kl_brake) * std_q_pi

            actor_bridge_metrics["bridge/actor_action_requires_grad"] = float(pi.requires_grad)
            actor_bridge_metrics["bridge/critic_input_requires_grad"] = float(critic_o_actor.requires_grad)

            # Autograd calculates gradient back to pi_clamped so we can log the pre-clamped norm
            try:
                dqda = torch.autograd.grad(
                    q_target_pi.mean(), pi_clamped, retain_graph=True, allow_unused=True
                )[0]
            except RuntimeError:
                dqda = None

            if dqda is not None:
                dqda_d = dqda.detach()
                dqda_raw_norm = dqda_d.norm(dim=-1).mean().item()
                actor_bridge_metrics["bridge/dqda_norm"] = dqda_raw_norm
                actor_bridge_metrics["bridge/dqda_abs_mean"] = dqda_d.abs().mean().item()
                actor_bridge_metrics["bridge/dqda_clipped"] = 1.0 if dqda_raw_norm > max_dqda_grad_norm else 0.0
                # Phase 2 diagnostic: per-sample max dqda norm. If batch mean is 0 but max is huge,
                # the gradient is exploding on a few samples and being averaged/clamped away.
                actor_bridge_metrics["bridge/dqda_norm_max"] = dqda_d.norm(dim=-1).max().item()
                actor_bridge_metrics["bridge/dqda_norm_p95"] = dqda_d.norm(dim=-1).quantile(0.95).item()
            else:
                actor_bridge_metrics["bridge/dqda_norm"] = 0.0
                actor_bridge_metrics["bridge/dqda_abs_mean"] = 0.0
                actor_bridge_metrics["bridge/dqda_clipped"] = 0.0
                actor_bridge_metrics["bridge/dqda_norm_max"] = 0.0
                actor_bridge_metrics["bridge/dqda_norm_p95"] = 0.0

            # Phase 2 diagnostic: critic-action sensitivity probe.
            # If q_pi_std (across batch) ~= 0, the critic is constant w.r.t. state for the chosen
            # actions — the actor has no signal regardless of any clamp.
            # Spearman-style per-dim correlation between action component and Q tells us whether
            # Q actually responds to action variation.
            with torch.no_grad():
                q_t = q_target_pi.detach().squeeze(-1)
                pi_d = pi.detach()
                actor_bridge_metrics["bridge/q_pi_mean"] = q_t.mean().item()
                actor_bridge_metrics["bridge/q_pi_std"] = q_t.std(unbiased=False).item()
                # Pearson correlation per action dim (cheap proxy for "Q listens to action").
                if q_t.numel() > 1 and q_t.std() > 1e-9:
                    q_centered = q_t - q_t.mean()
                    for d in range(pi_d.shape[-1]):
                        a_d = pi_d[:, d]
                        if a_d.std() > 1e-9:
                            a_centered = a_d - a_d.mean()
                            corr = (a_centered * q_centered).mean() / (
                                a_centered.std(unbiased=False) * q_t.std(unbiased=False) + 1e-12
                            )
                            actor_bridge_metrics[f"bridge/q_pi_action_corr_d{d}"] = corr.item()
                        else:
                            actor_bridge_metrics[f"bridge/q_pi_action_corr_d{d}"] = 0.0
                else:
                    for d in range(pi_d.shape[-1]):
                        actor_bridge_metrics[f"bridge/q_pi_action_corr_d{d}"] = 0.0

            entropy_cost = (alpha_t * logp_per_dim).sum(dim=-1)
            loss_pi = (entropy_cost.unsqueeze(dim=-1) - q_target_pi).mean()

            # === SBR: Steering-Only Boundary Repulsion (GRAC Component 3) ===
            # Adapted "inverting gradients" concept: mild L2 penalty on steering
            # to discourage gratuitous saturation that causes gradient death.
            # Applied to steering ONLY (index 0) — gas/brake are unpenalized.
            # Coefficient is ~200x smaller than reward signal to avoid sluggishness.
            sbr_lambda = cfg.TMRL_CONFIG.get("ALG", {}).get("SBR_LAMBDA", 0.005)
            steering_action = pi[:, 0]  # index 0 = steer
            loss_sbr = sbr_lambda * (steering_action ** 2).mean()
            loss_pi = loss_pi + loss_sbr

            # === Generation 6 Fix: Pre-Activation Logit Penalty ===
            # Prevents Unbounded Q-Ascent and Actor Gradient Starvation
            # by penalizing the pre-tanh mean (mu_pre_stochastic) from drifting to infinity
            logit_penalty_lambda = 1e-3
            loss_logit_penalty = logit_penalty_lambda * (mu_pre_stochastic ** 2).mean()
            loss_pi = loss_pi + loss_logit_penalty

            # === Actor Churn Regularization ===
            # Penalize rapid deterministic policy-head drift against a slow EMA
            # copy of the actor. This directly targets eval churn while keeping
            # real replay learning and critic/WM objectives unchanged.
            churn_enabled = bool(alg_cfg.get("ACTOR_CHURN_REG_ENABLED", True))
            churn_lambda = float(alg_cfg.get("ACTOR_CHURN_REG_LAMBDA", 0.01))
            churn_polyak = float(alg_cfg.get("ACTOR_CHURN_ANCHOR_POLYAK", 0.995))
            loss_churn = torch.zeros((), device=self.device)
            loss_churn_weighted = torch.zeros((), device=self.device)
            churn_anchor_mu_abs_diff = 0.0
            if churn_enabled and churn_lambda > 0.0:
                anchor_mu = self._actor_churn_anchor_mu(fused_o_actor, z_actor)
                loss_churn, loss_churn_weighted = _compute_actor_churn_loss(
                    mu_pre_stochastic,
                    anchor_mu,
                    enabled=churn_enabled,
                    weight=churn_lambda,
                )
                loss_pi = loss_pi + loss_churn_weighted
                with torch.no_grad():
                    churn_anchor_mu_abs_diff = (mu_pre_stochastic.detach() - anchor_mu).abs().mean().item()

            # EWC: lighter penalty on actor (0.1x) to allow policy adaptation
            if self._ewc_active and self.ewc_lambda > 0:
                loss_pi = loss_pi + self.ewc_lambda * 0.1 * self._ewc_loss()

            # === Advantage-Weighted Deterministic Bridge (AWDB) ===
            # Replaces volatile DPG mathematically with a normalized supervised teacher bridge.
            # Pull dynamically from self so tracking_offline.py (Auto-Annealer) can control it
            det_lambda_base = float(getattr(self, "det_reg_lambda", alg_cfg.get("DET_REG_LAMBDA", 0.03)))
            det_lambda_max = float(alg_cfg.get("DET_SKILL_TRANSFER_LAMBDA_MAX", det_lambda_base))
            if det_lambda_max < det_lambda_base:
                det_lambda_max = det_lambda_base
            det_skill_drive = float(det_skill_feedback.get("skill_transfer/stoch_adv_drive_ema", 0.0))
            det_preserve_drive = float(det_skill_feedback.get("skill_transfer/det_adv_drive_ema", 0.0))
            det_lambda = det_lambda_base + det_skill_drive * (det_lambda_max - det_lambda_base)
            loss_awdb = torch.zeros(1, device=self.device)
            awdb_q_adv_mean = 0.0
            awdb_q_adv_positive_mean = 0.0
            awdb_weight_mean = 0.0
            awdb_mse_mean = 0.0
            min_awdb_weight_eff = 0.0
            awdb_min_weight_target = 0.0
            awdb_min_weight_applied = 0.0
            if det_lambda > 0:
                # To prevent vanishing gradients on track corners (where tanh squashes the derivative to 0),
                # we must evaluate the MSE bridge mathematically BEFORE the tanh activation.

                # Extract the post-tanh squashed action (for Q-evaluation) and the pre-tanh 'mu' (for learning)
                pi_det_squashed, _, _, mu_pre = self.model.actor_from_features(
                    fused_o_actor, film_o_actor, test=True, with_logprob=False, z=z_actor, return_pretanh_mu=True)

                qs_det = [q(critic_o_actor, pi_det_squashed, film_o_actor) for q in self.model.qs]
                min_q_det = torch.min(torch.stack(qs_det, -1), dim=1)[0]

                # Advantage: How much better is the pure stochastic action (pi) vs deterministic action (pi_det)?
                advantage = (q_target_pi - min_q_det).detach()

                # Only pull deterministic towards stochastic IF stochastic is actually better (adv > 0)
                advantage_mask = torch.relu(advantage)

                # Reverse the successful stochastic 'pi' backwards into its pre-tanh infinite-space counterpart.
                # We detach() it permanently so Atanh instability can never traverse back into the network gradients.
                act_limit = getattr(self.model.actor, "act_limit", 1.0)
                pi_scaled = (pi.detach() / act_limit).clamp(-0.99999, 0.99999)
                pi_pre_target = torch.atanh(pi_scaled)

                # Direct pre-tanh Supervised Bridge (Absolutely zero vanishing gradients!)
                mse_bridge = torch.nn.functional.mse_loss(mu_pre, pi_pre_target, reduction='none').mean(dim=-1, keepdim=True)

                # Normalize advantage to solve explosion gradients, but keep a small
                # supervised bridge alive when the critic advantage is too flat to teach.
                adv_weight = advantage_mask / (advantage_mask.max() + 1e-8)
                min_awdb_weight_base = float(alg_cfg.get("MIN_AWDB_WEIGHT", 0.10))
                min_awdb_weight_max = float(alg_cfg.get("DET_SKILL_TRANSFER_MIN_AWDB_MAX", min_awdb_weight_base))
                if min_awdb_weight_max < min_awdb_weight_base:
                    min_awdb_weight_max = min_awdb_weight_base
                min_awdb_weight_target = min_awdb_weight_base + det_skill_drive * (
                    min_awdb_weight_max - min_awdb_weight_base
                )
                # FIX #4: Only apply min_awdb_weight when Q actually ranks stochastic
                # samples as better than deterministic. Without this gate, det_skill_drive
                # (from eval gap) forces distillation from fresh stochastic samples that
                # the critic may NOT rank as better — teaching from noise, not skill.
                q_adv_mean = advantage.mean().item()
                q_adv_positive = advantage_mask.mean().item()
                q_adv_gate_min = float(alg_cfg.get("DET_SKILL_AWDB_MIN_Q_ADV_MEAN", 0.0))
                min_awdb_weight_applied = _gate_awdb_min_weight(
                    q_adv_mean,
                    q_adv_positive,
                    min_awdb_weight_target,
                    min_mean=q_adv_gate_min,
                )
                if min_awdb_weight_applied > 0.0:
                    min_awdb_weight_eff = min_awdb_weight_applied
                    adv_weight = adv_weight.clamp(min=min_awdb_weight_eff, max=1.0)
                else:
                    min_awdb_weight_eff = 0.0

                loss_awdb = (mse_bridge * adv_weight).mean()
                loss_pi = loss_pi + det_lambda * loss_awdb
                with torch.no_grad():
                    awdb_q_adv_mean = q_adv_mean
                    awdb_q_adv_positive_mean = advantage_mask.mean().item()
                    awdb_weight_mean = adv_weight.mean().item()
                    awdb_mse_mean = mse_bridge.mean().item()
                    awdb_min_weight_target = float(min_awdb_weight_target)
                    awdb_min_weight_applied = float(min_awdb_weight_applied)

            loss_det_log_std = torch.zeros((), device=self.device)
            loss_det_log_std_weighted = torch.zeros((), device=self.device)
            log_std_above_ceiling = 0.0
            # FIX #3: log_std ceiling penalty should always be active at full weight,
            # not gated by det_skill_drive. When log_std > 0, the penalty is the only
            # thing that can push it back down. Gating it by skill drive made it ~100x
            # weaker than the entropy bonus from alpha ≈ 0.05.
            log_std_penalty_weight = float(alg_cfg.get("DET_SKILL_LOG_STD_PENALTY_MAX", 0.0))
            log_std_ceiling = float(alg_cfg.get("DET_SKILL_LOG_STD_CEILING", 0.0))
            if log_std_penalty_weight > 0.0:
                if z_actor is not None:
                    actor_input_std = torch.cat([fused_o_actor, z_actor], dim=-1)
                else:
                    actor_input_std = fused_o_actor
                log_std_raw_skill = self.model.actor.std_net(actor_input_std)
                log_std_skill = core._compute_log_std_smooth(log_std_raw_skill)
                log_std_excess = torch.relu(log_std_skill - log_std_ceiling)
                loss_det_log_std = (log_std_excess ** 2).mean()
                loss_det_log_std_weighted = log_std_penalty_weight * loss_det_log_std
                loss_pi = loss_pi + loss_det_log_std_weighted
                with torch.no_grad():
                    log_std_above_ceiling = (log_std_skill.detach() > log_std_ceiling).float().mean().item()

            self.pi_optimizer.zero_grad()
            loss_pi.backward()

            # GRAC: Update gradient health tracker AFTER backward, BEFORE optimizer step
            if hasattr(self, 'grad_tracker') and self.model.actor.mu_layer.weight.grad is not None:
                self.grad_tracker.update(self.model.actor.mu_layer.weight.grad)

            alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
            guard_enabled = alg_cfg.get("ACTOR_STABILITY_GUARD_ENABLED", True)

            # Hybrid Guard Thresholds
            tier1_health_min = float(alg_cfg.get("TIER1_HEALTH_THRESHOLD", 0.05))
            tier3_health_min = float(alg_cfg.get("TIER3_HEALTH_THRESHOLD", 0.01))
            severe_grad_health_min = float(alg_cfg.get("SEVERE_GRAD_HEALTH_THRESHOLD", 0.001))
            tier3_dqda_min = float(alg_cfg.get("TIER3_DQDA_THRESHOLD", 1e-5))
            q_drop_threshold = float(alg_cfg.get("Q_DROP_THRESHOLD", 0.02))
            dqda_starvation_threshold = float(alg_cfg.get("DQDA_STARVATION_THRESHOLD", 1e-4))
            starvation_lr_scale = float(alg_cfg.get("STARVATION_LR_SCALE", 0.5))
            tier1_lr_scale = float(alg_cfg.get("TIER1_LR_SCALE", 0.1))
            q_pi_std_health_min = float(alg_cfg.get("Q_PI_STD_HEALTH_THRESHOLD", 0.03))
            q_action_sensitivity_health_min = float(
                alg_cfg.get("Q_ACTION_SENSITIVITY_HEALTH_THRESHOLD", 0.003)
            )
            grad_health_only_lr_scale = float(alg_cfg.get("GRAD_HEALTH_ONLY_LR_SCALE", 1.0))

            if hasattr(self, 'grad_tracker'):
                grad_health_mean = self.grad_tracker.grad_health.detach().mean().item()
            else:
                grad_health_mean = 1.0
            dqda_norm_value = actor_bridge_metrics.get("bridge/dqda_norm", 0.0)
            q_pi_std_value = actor_bridge_metrics.get("bridge/q_pi_std", 0.0)
            q_action_sensitivity_value = q_action_sensitivity.detach().item()
            q_pi_mean_value = actor_bridge_metrics.get("bridge/q_pi_mean", 0.0)
            with torch.no_grad():
                q_real_guard = q_prediction_cat.detach()
                q_real_mean_guard = q_real_guard.mean().item()
                q_real_std_guard = q_real_guard.std(unbiased=False).item()
                reward_energy = (
                    r.detach().abs().mean() + r.detach().std(unbiased=False)
                ).item()
            q_scale_value = max(
                abs(q_pi_std_value) + abs(q_real_std_guard),
                0.05 * max(abs(q_pi_mean_value), abs(q_real_mean_guard)),
                1e-6,
            )
            q_gap_value = q_pi_mean_value - q_real_mean_guard
            q_gap_norm = max(0.0, q_gap_value / q_scale_value)

            adaptive_safety_enabled = bool(alg_cfg.get("ADAPTIVE_SAFETY_ENABLED", True))
            adaptive_decay = float(alg_cfg.get("ADAPTIVE_SAFETY_EMA_DECAY", 0.95))
            adaptive_warmup = int(alg_cfg.get("ADAPTIVE_SAFETY_EMA_WARMUP", 2))
            churn_norm, churn_baseline, self._adaptive_safety_stats = _ema_ratio_update(
                self._adaptive_safety_stats,
                "actor_churn_abs",
                churn_anchor_mu_abs_diff,
                decay=adaptive_decay,
                warmup=adaptive_warmup,
            )
            reward_energy_ratio, reward_energy_baseline, self._adaptive_safety_stats = _ema_ratio_update(
                self._adaptive_safety_stats,
                "reward_energy",
                reward_energy,
                decay=adaptive_decay,
                warmup=adaptive_warmup,
            )
            reward_liveness_soft = float(alg_cfg.get("ADAPTIVE_REWARD_LIVENESS_SOFT_RATIO", 0.50))
            reward_liveness_hard = float(alg_cfg.get("ADAPTIVE_REWARD_LIVENESS_HARD_RATIO", 0.10))
            if reward_energy_baseline <= 1e-8:
                data_liveness_health = 1.0
            elif reward_liveness_soft <= reward_liveness_hard:
                data_liveness_health = float(reward_energy_ratio > reward_liveness_hard)
            else:
                data_liveness_health = _clip01(
                    (reward_energy_ratio - reward_liveness_hard)
                    / (reward_liveness_soft - reward_liveness_hard)
                )

            dqda_explosion_threshold = float(alg_cfg.get("DQDA_EXPLOSION_THRESHOLD", 0.01))
            if not hasattr(self, "tier3_consecutive_blocks"):
                self.tier3_consecutive_blocks = 0

            guard_decision = _compute_actor_guard_decision(
                guard_enabled=guard_enabled,
                grad_health_mean=grad_health_mean,
                dqda_norm_value=dqda_norm_value,
                q_pi_std_value=q_pi_std_value,
                q_action_sensitivity_value=q_action_sensitivity_value,
                tier1_health_min=tier1_health_min,
                tier3_health_min=tier3_health_min,
                dqda_starvation_threshold=dqda_starvation_threshold,
                dqda_explosion_threshold=dqda_explosion_threshold,
                q_pi_std_health_min=q_pi_std_health_min,
                q_action_sensitivity_health_min=q_action_sensitivity_health_min,
                starvation_lr_scale=starvation_lr_scale,
                tier1_lr_scale=tier1_lr_scale,
                grad_health_only_lr_scale=grad_health_only_lr_scale,
                severe_grad_health_min=severe_grad_health_min,
                tier3_consecutive_blocks=self.tier3_consecutive_blocks,
                q_overconfidence_norm=q_gap_norm,
                churn_norm=churn_norm,
                data_liveness_health=data_liveness_health,
                adaptive_safety_enabled=adaptive_safety_enabled,
                q_overconfidence_soft=float(alg_cfg.get("ADAPTIVE_Q_GAP_SOFT", 0.5)),
                q_overconfidence_hard=float(alg_cfg.get("ADAPTIVE_Q_GAP_HARD", 1.25)),
                churn_soft_ratio=float(alg_cfg.get("ADAPTIVE_CHURN_SOFT_RATIO", 3.0)),
                churn_hard_ratio=float(alg_cfg.get("ADAPTIVE_CHURN_HARD_RATIO", 8.0)),
                adaptive_lr_floor=float(alg_cfg.get("ADAPTIVE_LR_FLOOR", 0.05)),
            )
            self.tier3_consecutive_blocks = guard_decision["tier3_consecutive_blocks"]
            tier3_hard_block_active = guard_decision["tier3_hard_block_active"]
            tier3_forced_exploration = guard_decision["tier3_forced_exploration"]
            lr_scale = guard_decision["lr_scale"]
            grad_clip_norm = guard_decision["grad_clip_norm"]
            dqda_exploding = guard_decision["dqda_exploding"]
            dqda_critical = guard_decision["dqda_critical"]
            dqda_starving = guard_decision["dqda_starving"]
            guard_tier = guard_decision["guard_tier"]

            q_old_mean = q_target_pi.detach().mean().item() if q_target_pi is not None else 0.0

            # Legacy metric continuity
            actor_bridge_metrics["guard/actor_stability_active"] = 1.0 if guard_tier > 0 else 0.0
            actor_bridge_metrics["guard/actor_mu_step_blocked"] = float(tier3_hard_block_active)
            actor_bridge_metrics["guard/actor_std_step_allowed"] = 1.0

            # New metrics
            actor_bridge_metrics["guard/q_gap"] = float(q_gap_value)
            actor_bridge_metrics["guard/q_scale"] = float(q_scale_value)
            actor_bridge_metrics["guard/q_gap_norm"] = float(q_gap_norm)
            actor_bridge_metrics["guard/q_overconfidence_risk"] = float(guard_decision["q_overconfidence_risk"])
            actor_bridge_metrics["guard/q_overconfidence_health"] = float(guard_decision["q_overconfidence_health"])
            actor_bridge_metrics["guard/churn_baseline"] = float(churn_baseline)
            actor_bridge_metrics["guard/churn_norm"] = float(churn_norm)
            actor_bridge_metrics["guard/churn_risk"] = float(guard_decision["churn_risk"])
            actor_bridge_metrics["guard/churn_health"] = float(guard_decision["churn_health"])
            actor_bridge_metrics["guard/reward_energy"] = float(reward_energy)
            actor_bridge_metrics["guard/reward_energy_baseline"] = float(reward_energy_baseline)
            actor_bridge_metrics["guard/reward_energy_ratio"] = float(reward_energy_ratio)
            actor_bridge_metrics["guard/data_liveness_health"] = float(guard_decision["data_liveness_health"])
            actor_bridge_metrics["guard/data_liveness_risk"] = float(guard_decision["data_liveness_risk"])
            actor_bridge_metrics["guard/adaptive_safety_active"] = float(guard_decision["adaptive_safety_active"])
            actor_bridge_metrics["guard/adaptive_risk"] = float(guard_decision["adaptive_risk"])
            actor_bridge_metrics["guard/adaptive_lr_scale"] = float(guard_decision["adaptive_lr_scale"])
            actor_bridge_metrics["guard/tier"] = float(guard_tier)
            actor_bridge_metrics["guard/tier3_hard_block_active"] = float(tier3_hard_block_active)
            actor_bridge_metrics["guard/tier3_forced_exploration"] = float(tier3_forced_exploration)
            actor_bridge_metrics["guard/lr_scale"] = float(lr_scale)
            actor_bridge_metrics["guard/q_shield_triggered"] = 0.0
            actor_bridge_metrics["guard/q_drop_value"] = 0.0

            actor_bridge_metrics["guard/dqda_min_norm"] = tier3_dqda_min
            actor_bridge_metrics["guard/grad_health_min"] = tier3_health_min
            actor_bridge_metrics["guard/severe_grad_health_min"] = severe_grad_health_min
            actor_bridge_metrics["guard/grad_health_mean"] = grad_health_mean
            actor_bridge_metrics["guard/dqda_explosion"] = float(dqda_exploding)
            actor_bridge_metrics["guard/dqda_critical"] = float(dqda_critical)
            actor_bridge_metrics["guard/dqda_starvation"] = float(dqda_starving)
            actor_bridge_metrics["guard/grad_health_severe"] = float(guard_decision["grad_health_severe"])
            actor_bridge_metrics["guard/starvation_lr_scale"] = float(starvation_lr_scale)
            actor_bridge_metrics["guard/grad_health_only"] = float(guard_decision["grad_health_only"])
            actor_bridge_metrics["guard/healthy_signal_override"] = float(guard_decision["healthy_signal_override"])
            actor_bridge_metrics["guard/throttle_reason"] = float(guard_decision["throttle_reason"])
            actor_bridge_metrics["guard/q_pi_std_health_min"] = q_pi_std_health_min
            actor_bridge_metrics["guard/q_pi_std_healthy"] = float(guard_decision["q_pi_std_healthy"])
            actor_bridge_metrics["guard/q_action_sensitivity_health_min"] = q_action_sensitivity_health_min
            actor_bridge_metrics["guard/q_action_sensitivity_healthy"] = float(
                guard_decision["q_action_sensitivity_healthy"]
            )
            actor_bridge_metrics["guard/grad_health_only_lr_scale"] = grad_health_only_lr_scale

            if tier3_hard_block_active:
                _clear_parameter_grads(
                    list(self.model.actor.net.parameters()) +
                    list(self.model.actor.mu_layer.parameters())
                )

            # Apply dynamic clipping for actor (Tier 1 tightens this to 0.1)
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.actor.net.parameters()) +
                list(self.model.actor.mu_layer.parameters()), grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.actor.std_net.parameters()), 0.5)

            # Backup state for Tier 2 Q-Shield
            actor_state_dict_backup = deepcopy(self.model.actor.state_dict())
            opt_state_dict_backup = deepcopy(self.pi_optimizer.state_dict())

            # Apply dynamic LR scale temporarily
            original_lrs = []
            for param_group in self.pi_optimizer.param_groups:
                original_lrs.append(param_group['lr'])
                param_group['lr'] = param_group['lr'] * lr_scale

            self.pi_optimizer.step()

            # Tier 2: Q-Value Difference Shield
            q_drop = 0.0
            q_shield_triggered = False
            if guard_enabled and not tier3_hard_block_active:
                with torch.no_grad():
                    pi_new, _, _, _ = self.model.actor_from_features(fused_o_actor, film_o_actor, z=z_actor, return_pretanh_mu=True)
                    qs_new = [q(critic_o_actor, pi_new, film_o_actor) for q in self.model.qs]
                    q_new_mean = torch.min(torch.stack(qs_new, -1), dim=1)[0].mean().item()

                    q_drop = q_old_mean - q_new_mean
                    # === Fix #4: Relative Q-shield threshold ===
                    # test_6 had Q-values ~0.03, so threshold=2.0 was impossible to hit.
                    # Use the larger of the absolute threshold and a fraction of current Q.
                    effective_q_threshold = max(q_drop_threshold, 0.5 * abs(q_old_mean))
                    if q_drop > effective_q_threshold:
                        q_shield_triggered = True

            if q_shield_triggered:
                actor_bridge_metrics["guard/tier"] = 2.0
                actor_bridge_metrics["guard/q_shield_triggered"] = 1.0
                actor_bridge_metrics["guard/q_drop_value"] = float(q_drop)

                # Revert
                self.model.actor.load_state_dict(actor_state_dict_backup)
                self.pi_optimizer.load_state_dict(opt_state_dict_backup)

                # Shrink update: Combine Tier 1 + Tier 2
                lr_scale = lr_scale * 0.1
                actor_bridge_metrics["guard/lr_scale"] = float(lr_scale)

                for i, param_group in enumerate(self.pi_optimizer.param_groups):
                    param_group['lr'] = original_lrs[i] * lr_scale

                self.pi_optimizer.step()

            # Restore original LRs
            for i, param_group in enumerate(self.pi_optimizer.param_groups):
                param_group['lr'] = original_lrs[i]

            if churn_enabled and churn_lambda > 0.0:
                self._update_actor_churn_anchor(churn_polyak)

            for q in self.model.qs:
                q.requires_grad_(True)

            entropy_health_metrics = dict(actor_bridge_metrics)
            entropy_health_metrics["bridge/q_action_sensitivity"] = q_action_sensitivity.detach().item()
            entropy_health_metrics["churn/anchor_mu_abs_diff"] = float(churn_anchor_mu_abs_diff)
            entropy_health_metrics["churn/q_anchor_abs_diff"] = float(q_churn_anchor_abs_diff)
            self._last_entropy_health_metrics = entropy_health_metrics

            # Entropy coefficient update
            if self.learn_entropy_coef:
                with torch.no_grad():
                    if uses_context and ctx is not None:
                        fused_alpha, _, film_alpha, z_alpha = self.model.forward_features(o, ctx)
                    else:
                        fused_alpha, _, film_alpha, z_alpha = self.model.forward_features(o)
                _, _, logp_per_dim_alpha = self.model.actor_from_features(fused_alpha, film_alpha, z=z_alpha)
                # Apply stall recovery entropy bump dynamically
                stall_bump = getattr(self, "stall_entropy_bump", 0.0)
                # GRAC: Apply GHAE entropy boost for dimensions with vanishing gradients
                ghae_boost = torch.zeros_like(self.target_entropy)
                if hasattr(self, 'grad_tracker'):
                    ghae_boost = self.grad_tracker.get_entropy_boost()
                # SAC target entropy is negative in this codepath. A positive
                # exploration boost must therefore make the target more negative.
                effective_target_entropy = self.target_entropy - stall_bump - ghae_boost
                effective_target_entropy_diag = effective_target_entropy.detach()

                loss_alpha = -(self.log_alpha * (logp_per_dim_alpha.detach().mean(dim=0) + effective_target_entropy)).sum()
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                torch.nn.utils.clip_grad_norm_(self.log_alpha, 1.0)
                self.alpha_optimizer.step()
                # Velocity clamp: prevent entropy coefficient from crashing in 1-2 epochs.
                # Limits log_alpha movement to ±max_alpha_delta per gradient step.
                with torch.no_grad():
                    max_alpha_delta = 0.01
                    if hasattr(self, '_prev_log_alpha'):
                        delta = self.log_alpha.data - self._prev_log_alpha
                        self.log_alpha.data.copy_(
                            self._prev_log_alpha + delta.clamp(-max_alpha_delta, max_alpha_delta)
                        )
                    self._prev_log_alpha = self.log_alpha.data.clone()

                    # === Fix #2: Entropy coefficient ceiling ===
                    # test_6 saw entropy_coef run away from 0.06 to 0.51 after collapse.
                    # SAC alpha auto-tuning overreacted to the policy shock. Hard ceiling
                    # prevents the entropy term from dominating the loss.
                    alpha_ceiling = float(alg_cfg.get("ALPHA_CEILING", 0.15))
                    log_alpha_ceiling = torch.log(torch.tensor(alpha_ceiling, device=self.log_alpha.device))
                    ceiling_hit = (self.log_alpha.data > log_alpha_ceiling).any().item()
                    self.log_alpha.data.clamp_(max=log_alpha_ceiling.item())
                    actor_bridge_metrics["guard/alpha_ceiling_hit"] = float(ceiling_hit)

                alpha_t, alpha_floor_t = self._apply_alpha_floor()

            self.loss_pi = loss_pi.detach()

        # === Polyak averaging ===
        if update_policy:
            with torch.no_grad():
                for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        # Diagnostics: expose alpha and log_std trends to catch entropy collapse early.
        with torch.no_grad():
            if uses_context and ctx is not None:
                fused_diag, critic_diag, film_diag, z_diag = self.model.forward_features(o, ctx)
            else:
                fused_diag, critic_diag, film_diag, z_diag = self.model.forward_features(o)
            if z_diag is None:
                z_diag = torch.zeros(fused_diag.shape[0], 64, device=fused_diag.device)
            actor_input_diag = torch.cat([fused_diag, z_diag], dim=-1)
            log_std_raw_diag = self.model.actor.std_net(actor_input_diag)
            log_std_diag = core._compute_log_std_smooth(log_std_raw_diag)

        ret_dict = dict(
            loss_actor=self.loss_pi.detach().item(),
            loss_critic=self.loss_q.detach().item(),
        )
        if 'loss_q_churn' in dir():
            ret_dict["loss_q_churn"] = (
                loss_q_churn.detach().item()
                if isinstance(loss_q_churn, torch.Tensor)
                else 0.0
            )
            ret_dict["loss_q_churn_weighted"] = (
                loss_q_churn_weighted.detach().item()
                if isinstance(loss_q_churn_weighted, torch.Tensor)
                else 0.0
            )
            ret_dict["churn/q_anchor_abs_diff"] = float(q_churn_anchor_abs_diff)
            ret_dict["churn/q_lambda"] = float(q_churn_lambda)
            ret_dict["churn/q_anchor_polyak"] = float(q_churn_polyak)
            ret_dict["churn/q_enabled"] = float(q_churn_enabled)
        if update_policy and 'loss_awdb' in dir():
            ret_dict["loss_awdb"] = loss_awdb.detach().item() if isinstance(loss_awdb, torch.Tensor) else 0.0
            ret_dict["skill_transfer/det_lambda_base"] = float(det_lambda_base)
            ret_dict["skill_transfer/det_lambda_eff"] = float(det_lambda)
            ret_dict["skill_transfer/det_lambda_max"] = float(det_lambda_max)
            ret_dict["skill_transfer/stoch_adv_drive_used"] = float(det_skill_drive)
            ret_dict["skill_transfer/det_preserve_drive_used"] = float(det_preserve_drive)
            ret_dict["skill_transfer/min_awdb_weight_eff"] = float(min_awdb_weight_eff)
            ret_dict["skill_transfer/min_awdb_weight_target"] = float(awdb_min_weight_target)
            ret_dict["skill_transfer/min_awdb_weight_applied"] = float(awdb_min_weight_applied)
            ret_dict["skill_transfer/q_stoch_minus_det_mean"] = float(awdb_q_adv_mean)
            ret_dict["skill_transfer/q_stoch_minus_det_positive_mean"] = float(awdb_q_adv_positive_mean)
            ret_dict["skill_transfer/awdb_weight_mean"] = float(awdb_weight_mean)
            ret_dict["skill_transfer/awdb_mse_mean"] = float(awdb_mse_mean)
        if update_policy and 'loss_det_log_std' in dir():
            ret_dict["skill_transfer/loss_log_std_ceiling"] = (
                loss_det_log_std.detach().item()
                if isinstance(loss_det_log_std, torch.Tensor)
                else 0.0
            )
            ret_dict["skill_transfer/loss_log_std_ceiling_weighted"] = (
                loss_det_log_std_weighted.detach().item()
                if isinstance(loss_det_log_std_weighted, torch.Tensor)
                else 0.0
            )
            ret_dict["skill_transfer/log_std_ceiling"] = float(log_std_ceiling)
            ret_dict["skill_transfer/log_std_penalty_weight"] = float(log_std_penalty_weight)
            ret_dict["skill_transfer/log_std_above_ceiling"] = float(log_std_above_ceiling)
        if update_policy and 'loss_sbr' in dir():
            ret_dict["loss_sbr"] = loss_sbr.detach().item() if isinstance(loss_sbr, torch.Tensor) else 0.0
        if update_policy and 'loss_churn' in dir():
            ret_dict["loss_churn"] = loss_churn.detach().item() if isinstance(loss_churn, torch.Tensor) else 0.0
            ret_dict["loss_churn_weighted"] = (
                loss_churn_weighted.detach().item()
                if isinstance(loss_churn_weighted, torch.Tensor)
                else 0.0
            )
            ret_dict["churn/anchor_mu_abs_diff"] = float(churn_anchor_mu_abs_diff)
            ret_dict["churn/lambda"] = float(churn_lambda)
            ret_dict["churn/anchor_polyak"] = float(churn_polyak)
            ret_dict["churn/enabled"] = float(churn_enabled)
        if self._ewc_active:
            ret_dict["ewc_loss"] = loss_ewc.detach().item() if 'loss_ewc' in dir() else 0.0
        ret_dict.update(det_skill_feedback)
        if uses_context:
            ret_dict["kl_div_loss"] = loss_kl.detach().item() if isinstance(loss_kl, torch.Tensor) else 0.0
            if ctx is not None and ctx.shape[-1] > 23:
                with torch.no_grad():
                    context_reward = ctx[:, :, 23].detach()
                    ret_dict["bridge/context_reward_present"] = 1.0
                    ret_dict["bridge/context_reward_mean"] = context_reward.mean().item()
                    ret_dict["bridge/context_reward_std"] = context_reward.std(unbiased=False).item()
                    ret_dict["bridge/context_reward_abs_mean"] = context_reward.abs().mean().item()
            else:
                ret_dict["bridge/context_reward_present"] = 0.0
        ret_dict["debug_alpha_steer"] = alpha_t[0].item()
        ret_dict["debug_alpha_gas"] = alpha_t[1].item()
        ret_dict["debug_alpha_brake"] = alpha_t[2].item()
        ret_dict["debug_alpha_floor_steer"] = alpha_floor_t[0].item()
        ret_dict["debug_alpha_floor_gas"] = alpha_floor_t[1].item()
        ret_dict["debug_alpha_floor_brake"] = alpha_floor_t[2].item()
        for key, value in getattr(self, "_last_entropy_floor_diag", {}).items():
            ret_dict[key] = float(value)
        with torch.no_grad():
            q_real = q_prediction_cat.detach()
            ret_dict["bridge/q_real_mean"] = q_real.mean().item()
            ret_dict["bridge/q_real_std"] = q_real.std(unbiased=False).item()
        ret_dict["bridge/q_action_sensitivity"] = q_action_sensitivity.detach().item()
        ret_dict["bridge/q_action_sensitivity_loss"] = q_action_sensitivity_loss.detach().item()
        ret_dict["bridge/q_action_sensitivity_loss_raw"] = q_action_sensitivity_loss_raw.detach().item()
        ret_dict["bridge/q_action_sensitivity_gap"] = q_action_sensitivity_gap.detach().item()
        ret_dict["bridge/q_action_sensitivity_drive"] = q_action_sensitivity_drive.detach().item()
        ret_dict["bridge/q_action_sensitivity_lambda"] = float(q_action_sensitivity_lambda_eff)
        ret_dict["bridge/q_action_sensitivity_lambda_base"] = float(q_action_sens_lambda)
        ret_dict["bridge/q_action_sensitivity_lambda_max"] = float(q_action_sens_lambda_max)
        ret_dict["bridge/q_action_sensitivity_loss_type"] = float(q_action_sensitivity_loss_type_code)
        ret_dict["bridge/q_action_sensitivity_probe_std"] = float(q_action_noise_std)
        ret_dict["bridge/q_action_sensitivity_probe_count"] = float(q_action_probe_count)
        ret_dict["bridge/q_action_sensitivity_floor"] = float(q_action_std_floor)
        ret_dict["bridge/q_action_sensitivity_huber_beta"] = float(q_action_sens_huber_beta)
        ret_dict.update(actor_bridge_metrics)
        if not hasattr(self, "_last_hgi_actor_health_metrics"):
            self._last_hgi_actor_health_metrics = {}
        if all(key in actor_bridge_metrics for key in _HGI_ACTOR_HEALTH_REQUIRED_KEYS):
            actor_health_snapshot = {}
            for key in _HGI_ACTOR_HEALTH_CACHE_KEYS:
                if key in actor_bridge_metrics:
                    try:
                        actor_health_snapshot[key] = float(actor_bridge_metrics[key])
                    except (TypeError, ValueError):
                        pass
            self._last_hgi_actor_health_metrics = actor_health_snapshot

        # Expose gradient movements for live tracking
        if 'q_grad_norm' in locals() and isinstance(q_grad_norm, torch.Tensor):
            ret_dict["grad_norm_critic"] = q_grad_norm.detach().item()
        if 'actor_grad_norm' in locals() and isinstance(actor_grad_norm, torch.Tensor):
            ret_dict["grad_norm_actor"] = actor_grad_norm.detach().item()

        ret_dict["debug_log_std_mean"] = log_std_diag.detach().mean().item()
        ret_dict["debug_log_std_min"] = log_std_diag.detach().min().item()

        # GRAC diagnostics: gradient health and GHAE boost per dimension
        if hasattr(self, 'grad_tracker'):
            ret_dict["grad_health_steer"] = self.grad_tracker.grad_health[0].item()
            ret_dict["grad_health_gas"] = self.grad_tracker.grad_health[1].item()
            ret_dict["grad_health_brake"] = self.grad_tracker.grad_health[2].item()
            ghae = self.grad_tracker.get_entropy_boost()
            ret_dict["ghae_boost_steer"] = ghae[0].item()
            ret_dict["ghae_boost_gas"] = ghae[1].item()
            ret_dict["ghae_boost_brake"] = ghae[2].item()
        if effective_target_entropy_diag is not None:
            ret_dict["debug_target_entropy_steer"] = effective_target_entropy_diag[0].item()
            ret_dict["debug_target_entropy_gas"] = effective_target_entropy_diag[1].item()
            ret_dict["debug_target_entropy_brake"] = effective_target_entropy_diag[2].item()

        if self.learn_entropy_coef and loss_alpha is not None:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.mean().item()

        # ── World Model training (RSSM) ─────────────────────────────────
        if self.wm_enabled:
            # Use z_critic (always computed) as context for the world model
            # z_actor is only available during policy updates, but WM trains every step
            wm_context_z = z_critic.detach() if z_critic is not None else None
            wm_metrics = self._train_dynamics(o, a, r, o2, context_z=wm_context_z)
            ret_dict.update(wm_metrics)
            ret_dict["dynamics_loss"] = wm_metrics.get("wm_total_loss", 0.0)
            ret_dict["wm_train_steps"] = self.wm_train_steps
            ret_dict["wm_kl"] = wm_metrics.get("wm_kl", 0.0)
            ret_dict["wm_recon_state"] = wm_metrics.get("wm_recon_state", 0.0)

            # ── Train ImaginationActor to mimic real policy ─────────────
            critic_state_for_imag = self._extract_critic_state(o).detach()
            real_action = a.detach()
            pred_action = self.imag_actor(critic_state_for_imag)
            imag_actor_loss = F.mse_loss(pred_action, real_action)
            self.imag_actor_optimizer.zero_grad()
            imag_actor_loss.backward()
            self.imag_actor_optimizer.step()
            ret_dict["imag_actor_loss"] = imag_actor_loss.item()

            hgi_actor_health_cache = getattr(self, "_last_hgi_actor_health_metrics", {})
            ret_dict.update(
                _compute_hgi_trust_metrics(
                    ret_dict,
                    cfg.TMRL_CONFIG.get("ALG", {}),
                    actor_health_cache=hgi_actor_health_cache,
                )
            )
            if self.wm_train_steps > self.wm_warmup:
                hgi_gate = _compute_hgi_imagination_gate(
                    ret_dict,
                    cfg.TMRL_CONFIG.get("ALG", {}),
                    base_horizon=self.wm_horizon,
                    post_warmup_steps=max(0.0, self.wm_train_steps - self.wm_warmup),
                )
                hgi_gate["hgi/warmup_active"] = 0.0
                hgi_gate["hgi/skip_reason_warmup"] = 0.0
                hgi_gate["hgi/warmup_remaining_steps"] = 0.0
                ret_dict.update(hgi_gate)
                effective_horizon = int(hgi_gate["hgi/effective_horizon"])
                if effective_horizon >= 2:
                    imag_metrics = self._imagined_critic_update(
                        o,
                        ctx=ctx if uses_context else None,
                        context_z=wm_context_z,
                        real_action=a.detach(),
                        horizon=effective_horizon,
                    )
                    ret_dict.update(imag_metrics)
                    if "bridge/q_imag_mean" in imag_metrics and "bridge/q_real_mean" in ret_dict:
                        ret_dict["bridge/q_imag_minus_real"] = (
                            imag_metrics["bridge/q_imag_mean"] - ret_dict["bridge/q_real_mean"]
                        )
                    self._last_verifier_trust = imag_metrics.get("verifier_trust_mt_mean", 1.0)
                    ret_dict.update(
                        _compute_hgi_trust_metrics(
                            ret_dict,
                            cfg.TMRL_CONFIG.get("ALG", {}),
                            actor_health_cache=hgi_actor_health_cache,
                        )
                    )
                else:
                    ret_dict["wm_imagined_steps"] = 0.0
                    ret_dict["wm_imagined_q_loss"] = 0.0

                # Log curiosity bonus stats (uses normalized surprise)
                state_cur = self._extract_critic_state(o).detach()
                state_nxt = self._extract_critic_state(o2).detach()
                with torch.no_grad():
                    surprise = self.dynamics.compute_surprise(state_cur, a, state_nxt, context_z=wm_context_z)
                surprise_norm = self.curiosity_rms.normalize(surprise)
                clip_val = getattr(self, 'curiosity_reward_clip', 5.0)
                surprise_norm_clipped = surprise_norm.clamp(-clip_val, clip_val)
                ret_dict["curiosity_surprise_raw_mean"] = surprise.mean().item()
                self._last_surprise_mean = surprise.mean().item()  # track for diagnostics
                ret_dict["curiosity_surprise_norm_mean"] = surprise_norm_clipped.mean().item()
                ret_dict["curiosity_bonus_mean"] = (self.curiosity_scale * surprise_norm_clipped).mean().item()
                ret_dict["curiosity_rms_mean"] = self.curiosity_rms.mean
                ret_dict["curiosity_rms_std"] = self.curiosity_rms.var ** 0.5
                # Log alpha floors (decoupled from curiosity)
                floor_mults = torch.tensor([1.5, 0.8, 0.5])
                trust_mod = 1.0 + (0.5 - getattr(self, '_last_verifier_trust', 1.0))
                trust_mod = max(0.2, min(2.0, trust_mod))
                # Removed dynamic_alpha_floor from logs as the metric is now organic alpha.
            else:
                ret_dict.update(
                    _compute_hgi_warmup_gate(
                        cfg.TMRL_CONFIG.get("ALG", {}),
                        base_horizon=self.wm_horizon,
                        warmup_remaining_steps=self.wm_warmup - self.wm_train_steps,
                    )
                )
                ret_dict["wm_imagined_steps"] = 0.0
                ret_dict["wm_imagined_q_loss"] = 0.0

        return ret_dict
