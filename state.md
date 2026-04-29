STATUS: REVIEW

# TMRL WM + Bridge Gradient-Pass Logging Plan

## Objective
Add lightweight diagnostics for each training gradient pass so we can see whether the world model, context bridge, imagined rollout bridge, critic bridge, and actor-gradient bridge are healthy.

The goal is observability only. Do not change the core Meta-RL agent structure, training loops, model architecture, optimizer behavior, or loss definitions unless a checklist item explicitly says so.

## Phase 1: Shared Logging Helpers

_Primary file: `tmrl/custom/custom_algorithms.py`_

### [x] 1. Add tensor-stat and gradient-norm helpers

Add small local helpers near the existing algorithm utilities:

```python
def _add_tensor_stats(logs, prefix, x, dead_threshold=1e-3):
    ...

def _module_grad_norm(module):
    ...

def _parameter_grad_norm(parameters):
    ...
```

Requirements:

- Helpers must be safe under `torch.no_grad()`.
- They must tolerate `None` tensors/modules.
- Tensor stats should include mean, std, abs mean, min, max.
- For tensors with batch dimension and feature dimension, also log per-dimension std min/mean/max and dead-dimension ratio.
- Gradient helpers should return `0.0` if no gradients exist.

## Phase 2: World Model Training Diagnostics

_Primary file: `tmrl/custom/custom_algorithms.py`_

### [x] 2. Log replay inputs before each WM gradient pass

Inside the dynamics/world-model training path, before the WM train step, log:

```text
wm_input/state_mean
wm_input/state_std
wm_input/state_abs_mean
wm_input/next_state_mean
wm_input/next_state_std
wm_input/action_mean
wm_input/action_std
wm_input/reward_mean
wm_input/reward_std
wm_input/context_z_mean
wm_input/context_z_std
wm_input/context_z_dead_dim_ratio_1e3
```

Requirements:

- Use the shared tensor-stat helper.
- Only log `context_z` stats when `context_z` is present.
- Do not detach tensors in a way that changes training behavior.

### [x] 3. Log latent posterior, prior, KL, reconstruction, and reward metrics

Inside `LatentWorldModel.train_step(...)`, extend the returned metrics with:

```text
wm/z_t_mean
wm/z_t_std
wm/z_t_abs_mean
wm/z_t_dead_dim_ratio_1e3
wm/z_t_dim_std_min
wm/z_t_dim_std_mean
wm/z_t_dim_std_max
wm/prior_mu_mean
wm/prior_mu_std
wm/prior_logvar_mean
wm/prior_logvar_std
wm/post_mu_mean
wm/post_mu_std
wm/post_logvar_mean
wm/post_logvar_std
wm/kl_mean
wm/kl_std
wm/kl_min
wm/kl_max
wm/state_recon_loss
wm/reward_loss
wm/reward_pred_mean
wm/reward_pred_std
wm/reward_target_mean
wm/reward_target_std
wm/reward_error_abs_mean
```

Requirements:

- Prefer existing loss tensors/variables where available.
- Do not add extra forward passes for this checklist item.
- Keep metric names stable and scalar.

### [x] 4. Log WM gradient norms after backward and before optimizer step

In the WM training step, after `loss.backward()` and before optimizer stepping/clipping, log:

```text
wm_grad/encoder
wm_grad/prior
wm_grad/posterior
wm_grad/decoder
wm_grad/reward_head
wm_grad/total
```

Requirements:

- Match names to actual submodules in `LatentWorldModel`.
- If a submodule does not exist under that exact role, skip it or map it to the closest actual module name.
- Do not change clipping or optimizer behavior.

## Phase 3: Imagination Rollout Diagnostics

_Primary file: `tmrl/custom/custom_algorithms.py`_

### [x] 5. Log imagined rollout health by horizon

Inside `LatentWorldModel.imagine(...)` or the caller that receives imagined rollout tensors, log per horizon:

```text
wm_imag/h{h}_state_mean
wm_imag/h{h}_state_std
wm_imag/h{h}_state_abs_mean
wm_imag/h{h}_action_mean
wm_imag/h{h}_action_std
wm_imag/h{h}_reward_mean
wm_imag/h{h}_reward_std
wm_imag/h{h}_uncertainty_mean
wm_imag/h{h}_uncertainty_std
```

Requirements:

- Log only tensors already produced by imagination.
- If uncertainty is unavailable, skip uncertainty keys.
- Keep the logging cheap enough to run every gradient pass.

## Phase 4: Bridge Diagnostics

_Primary file: `tmrl/custom/custom_algorithms.py`_

### [x] 6. Log real-vs-imagined state and action bridge statistics

In `_imagined_critic_update(...)`, log:

```text
bridge/state_real_mean
bridge/state_real_std
bridge/state_imag_mean
bridge/state_imag_std
bridge/state_distribution_gap
bridge/action_real_mean
bridge/action_real_std
bridge/action_imag_mean
bridge/action_imag_std
bridge/act1_real_mean
bridge/act2_real_mean
bridge/act1_vs_imag_action_abs_diff
bridge/act2_vs_act1_abs_diff
```

Requirements:

- `state_distribution_gap` should compare feature means between real and imagined states when shapes are compatible.
- Action-history diagnostics should verify that critic history slots are not accidentally duplicated current actions.
- Do not change imagined critic target construction.

### [x] 7. Log context bridge health

Where `context_z` enters WM training, imagination, surprise, and critic-facing logic, log:

```text
bridge/context_z_mean
bridge/context_z_std
bridge/context_z_abs_mean
bridge/context_z_dead_dim_ratio_1e3
bridge/context_z_requires_grad
bridge/context_z_to_wm_present
bridge/context_z_to_critic_present
```

Requirements:

- Presence flags should be `1.0` or `0.0`.
- `requires_grad` should be logged as `1.0` or `0.0`.
- Do not force gradients on or off.

### [x] 8. Log critic value bridge metrics for real vs imagined batches

In critic update paths that have access to both real and imagined data, log:

```text
bridge/q_real_mean
bridge/q_real_std
bridge/q_imag_mean
bridge/q_imag_std
bridge/q_imag_minus_real
bridge/target_q_imag_mean
bridge/target_q_imag_std
bridge/imag_td_error_mean
bridge/imag_td_error_abs_mean
```

Requirements:

- Use existing Q predictions and targets when available.
- Avoid adding expensive duplicate critic forward passes unless there is no existing value to log.
- If real Q is not available in the imagined update function, log imagined-only values there and leave real-vs-imag comparison for the real critic update path.

### [x] 9. Log actor-critic gradient bridge health

In the actor update path, add a cheap probe for:

```text
bridge/dqda_norm
bridge/dqda_abs_mean
bridge/actor_action_requires_grad
bridge/critic_input_requires_grad
```

Requirements:

- The probe must not alter the actual actor loss or optimizer state.
- Use `torch.autograd.grad` carefully and tolerate unavailable gradients.
- Do not retain graphs longer than needed.

## Phase 5: Trust / Uncertainty Diagnostics

_Primary file: `tmrl/custom/custom_algorithms.py`_

### [x] 10. Log verifier uncertainty and trust saturation

Where verifier uncertainty or trust is computed, log:

```text
wm/verifier_uncertainty_mean
wm/verifier_uncertainty_std
wm/verifier_trust_mean
wm/verifier_trust_std
wm/verifier_trust_min
wm/verifier_trust_max
wm/verifier_trust_saturation_low
wm/verifier_trust_saturation_high
bridge/trust_mean
bridge/trust_saturation_low
bridge/trust_saturation_high
```

Requirements:

- Saturation low is the fraction of trust values below `0.05`.
- Saturation high is the fraction of trust values above `0.95`.
- Do not change trust computation in this task.

## Phase 6: Verification

### [x] 11. Run a syntax/import check

Run the lightest practical check available for the edited files, such as:

```powershell
python -m py_compile tmrl/custom/custom_algorithms.py
```

If project imports require unavailable runtime dependencies, document the blocker in the final response.

### [x] 12. Handoff for review

When all checklist items are complete:

- Change the top of this file to `STATUS: REVIEW`.
- Stop and report the implemented logging groups and verification result.
