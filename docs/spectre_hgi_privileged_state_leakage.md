# SPECTRE-HGI Privileged State Leakage Postmortem

Date: 2026-05-04

## Executive Summary

`SPECTRE_HGI_test_2` showed a painful but useful failure pattern: the agent
could sometimes discover strong deterministic behavior, then lose it again.
The world model and critic often looked numerically healthy, while the policy
still churned and evaluation returns oscillated.

The main architecture issue we found is privileged state leakage.

In the `TM2020HYBRID` setup, the environment exposes absolute map information:

```text
xyz absolute position
absolute track progress
```

The actor did not directly use these fields in its normal float branch, but the
critic and RSSM world model did. That means the actor was trained by teacher
signals that could know where the car was on this exact track, even when the
actor itself was supposed to drive from local perception.

This can create brittle behavior. The critic or world model may learn that a
state is valuable because of map location instead of because of ego-local
driving facts such as lidar, speed, and action history. The actor then receives
gradients shaped by a map-aware teacher and can converge toward track-specific
exploits or unstable policies.

## Symptoms

The live run showed:

- `return_test_det` sometimes spiking high, including around `82`.
- Later deterministic evaluation falling back down.
- Stochastic and deterministic evaluation frequently disagreeing.
- Training return sometimes looking much better than eval stability.
- World-model metrics looking healthy while policy behavior remained unstable.
- HGI staying disabled during the long `50000` WM warmup, so most of the early
  run behaved as model-free actor-critic despite the WM learning in the
  background.

The important interpretation is:

```text
not simply "the actor is dead"
not simply "the WM is bad"
not simply "reward hacking only"

more likely:
privileged-state leakage + brittle critic/WM guidance + policy churn
```

## Evidence From Code

The active config uses:

```text
ENV.RTGYM_INTERFACE = TM2020HYBRID
ALG.ALGORITHM = DROQSAC
WORLD_MODEL.ENABLED = true
```

The hybrid environment builds observations as:

```text
speed, gear, rpm, images, lidar, xyz, progress, crash, progress_gain
```

The old RSSM/world-model state was:

```text
speed(1)
gear(1)
rpm(1)
lidar(19)
xyz(3)
absolute progress(1)
crash(1)
progress_gain(1)
= 28 dims
```

The old critic float input was:

```text
speed(1)
gear(1)
rpm(1)
lidar(19)
xyz(3)
absolute progress(1)
crash(1)
progress_gain(1)
act1(3)
act2(3)
= 34 dims
```

The actor float branch used:

```text
speed
gear
rpm
lidar
act1
act2
```

So the actor did not directly consume `xyz` and `progress`, but its training
signal came from a critic and world model that did consume them.

That is the leak.

## Why This Is Dangerous

Absolute position and absolute progress can be useful for a critic, but they
are dangerous in this setup because the goal is robust driving behavior, not
track-coordinate lookup.

With absolute state, the critic can learn:

```text
At this exact track coordinate, this action is high value.
```

Instead of:

```text
Given this speed, lidar shape, action history, and recent reward/progress,
this action is high value.
```

The first version is brittle. It can work on one trajectory, one spawn, one
track, or one local exploit, then collapse when small conditions change.

The second version is closer to the intended behavior: a policy that reacts to
what the car can locally sense.

## Fix 1: Scrub Absolute State From Critic And World Model

Status: implemented.

The RSSM/WM state is now ego-local:

```text
speed(1)
gear(1)
rpm(1)
lidar(19)
crash(1)
progress_gain(1)
= 24 dims
```

The critic float state is now:

```text
speed(1)
gear(1)
rpm(1)
lidar(19)
crash(1)
progress_gain(1)
act1(3)
act2(3)
= 30 dims
```

Removed from critic and WM:

```text
xyz
absolute progress
```

Kept:

```text
speed
gear
rpm
lidar
crash flag
progress_gain
action history
image features
context z
```

Important nuance: the environment and replay memory can still store `xyz` and
`progress`. The model now ignores them for critic/WM state construction. This
keeps the replay pipeline compatible while removing privileged information from
the teacher signal.

Main files changed:

- `tmrl/custom/custom_models.py`
  - added `EGO_WM_STATE_DIM = 24`
  - added `EGO_CRITIC_FLOAT_DIM = 30`
  - changed contextual DroQ critic heads from `34 + CONTEXT_Z_DIM` to
    `30 + CONTEXT_Z_DIM`
  - changed critic float construction to exclude `xyz` and absolute `progress`
  - changed `ImaginationActor` default state dim to `EGO_WM_STATE_DIM`

- `tmrl/custom/custom_algorithms.py`
  - changed WM/RSSM state dim to `core.EGO_WM_STATE_DIM`
  - changed `_extract_critic_state(...)` to return only ego-local state
  - changed lazy init paths to use the same constant

## Fix 2: Preserve Best Evaluation Checkpoints

Status: implemented.

Policy churn was one of the clearest symptoms. The agent could find good
deterministic behavior and then lose it. Normal checkpointing can overwrite the
best policy with a later worse policy.

The preservation fix adds a separate best-eval checkpoint. It does not change
learning behavior. It only saves an additional checkpoint when deterministic
evaluation improves enough.

Default behavior:

```text
BEST_CHECKPOINT_ENABLED = true
BEST_CHECKPOINT_METRIC = return_test_det
BEST_CHECKPOINT_MIN_RETURN = 50.0
tie-breaker = longer episode_length_test_det
```

Logged metrics:

```text
best_checkpoint/triggered
best_checkpoint/best_return_test_det
best_checkpoint/best_epoch
best_checkpoint/best_round
best_checkpoint/save_failed
```

Why this matters:

```text
good policy discovered -> preserve it
later actor churn -> normal checkpoint may degrade, best checkpoint survives
```

## Fix 3: Conservative HGI Post-Warmup Ramp

Status: implemented.

The WM should not jump from no imagination directly to full `15`-step
imagination. The ramp adds a probation period after WM warmup.

Behavior:

```text
during WM warmup:
  effective_horizon = 0

first 5000 WM train steps after warmup:
  if HGI gate is healthy, cap imagination to HGI_SHORT_HORIZON

after probation:
  full HGI_FULL_HORIZON is allowed only if existing trust/health gates pass
```

Default knobs:

```text
HGI_POST_WARMUP_RAMP_ENABLED = true
HGI_POST_WARMUP_SHORT_STEPS = 5000
HGI_SHORT_HORIZON = 3
HGI_FULL_HORIZON = 15
```

Logged metrics:

```text
hgi/ramp_active
hgi/post_warmup_steps
hgi/ramp_remaining_steps
hgi/post_warmup_short_steps
```

Why this matters:

```text
WM usable but imperfect -> short imagination first
WM plus critic healthy after probation -> full horizon can engage
WM/critic unhealthy -> imagination still skips
```

## Fix 4: Reduce WM Warmup For Next Run

Status: recommended config change.

The old config used:

```text
WORLD_MODEL.WARMUP_STEPS = 50000
```

In the live logs, the WM already looked usable around `18000` to `23000` WM
train steps:

```text
hgi/model_trust around 0.97
wm_recon_state around 0.001 to 0.002
wm/reward_error_abs_mean around 0.012 to 0.014
wm_val/recon_prior_post_ratio near 1.0
```

But because warmup was `50000`, HGI stayed off:

```text
hgi/effective_horizon = 0
hgi/skipped_ratio = 1
hgi/warmup_active = 1
```

For the next scrubbed run, use one of:

```json
"WORLD_MODEL": {
  "WARMUP_STEPS": 20000
}
```

or the more conservative:

```json
"WORLD_MODEL": {
  "WARMUP_STEPS": 25000
}
```

With the post-warmup ramp, this is safer than it sounds:

```text
0 to 20k/25k:
  WM trains, no imagination

next 5k:
  short-horizon imagination only

after that:
  full horizon only if HGI trust/health gate passes
```

## What This Does Not Fix By Itself

This does not automatically solve every failure mode.

Remaining possible issues:

- Reward can still encourage oscillation if `progress_gain` or survival terms
  are exploitable.
- Spawn determinism can still encourage memorization through images/lidar.
- The actor can still churn if critic gradients remain noisy.
- Context `z` can still encode track-specific history through images and
  reward sequence, although it no longer directly receives `xyz/progress`.
- Old checkpoints with the previous tensor shapes are incompatible with the
  scrubbed critic/WM modules.

## Required Run Protocol After Scrub

Because tensor shapes changed:

```text
WM state: 28 -> 24
critic floats: 34 -> 30
```

Use a fresh trainer reset or new run name. Do not continue an old trainer
checkpoint into the scrubbed architecture unless the checkpoint updater
explicitly rebuilds the agent.

Recommended next-run checklist:

1. Set a new `RUN_NAME`.
2. Ensure trainer reset/new checkpoint.
3. Lower `WORLD_MODEL.WARMUP_STEPS` to `20000` or `25000`.
4. Keep best-eval checkpoint enabled.
5. Keep HGI post-warmup ramp enabled.
6. Watch deterministic eval, not only stochastic eval.

## Metrics To Watch

Good signs:

```text
return_test_det spikes become less temporary
return_test_det and return_test_stoch gap shrinks
best_checkpoint/triggered becomes 1 when a real high-det policy appears
hgi/effective_horizon becomes 3 after warmup
hgi/ramp_active becomes 1 during probation
hgi/effective_horizon reaches 15 only when trust/health gates pass
hgi/model_trust remains healthy
hgi/critic_health does not collapse
wm_val/recon_prior_post_ratio stays near 1.0
bridge/q_action_sensitivity remains above its health floor
bridge/q_pi_std remains above its health floor
```

Bad signs:

```text
return_train rises while return_test_det stays low
return_test_stoch is healthy but return_test_det is dead
best_checkpoint/triggered never fires
hgi/critic_health collapses after imagination starts
wm_val/recon_prior_post_ratio drifts far above 1.0
bridge/dqda_norm falls into starvation
guard/actor_mu_step_blocked becomes persistent
policy becomes high-frequency steering jitter
```

## If Oscillation Continues

If the scrubbed run still shows oscillation, the next fixes should target the
environment/reward directly:

1. Add or strengthen progress-only reward:
   - primary reward should be forward centerline/checkpoint progress, not
     survival or raw speed.

2. Add anti-idle termination:
   - terminate if progress gain stays below a threshold for a fixed window.

3. Add action derivative penalty:
   - penalize large `||a_t - a_{t-1}||`, especially steering jitter.

4. Add spawn perturbation:
   - randomize start pose, start progress, angle, and speed if the simulator
     allows it.

5. Audit image/lidar determinism:
   - if the visual observation is too map-specific and spawn is fixed, the
     actor can still learn a visual macro.

## Short Diagnosis

The old architecture let the critic and world model act like a map-aware
teacher. That teacher could guide the actor using absolute track coordinates
and absolute progress, even though the actor itself mostly consumed local
sensors.

The implemented fix forces the teacher networks to reason from ego-local
driving signals:

```text
speed
gear
rpm
lidar
crash
progress_gain
action history
context
```

Then best-eval checkpointing preserves good policies when they appear, and the
post-warmup ramp prevents the WM from suddenly influencing the agent with full
horizon imagination.

The next run should test whether removing map-aware teacher leakage makes the
good behavior stick instead of appearing briefly and then vanishing.
