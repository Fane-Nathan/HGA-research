STATUS: REVIEW

# SPECTRE-HGI: Health-Gated Imagination Actor-Critic

## Summary
Build the novel direction around **Health-Gated Imagination**: imagination is allowed to train the actor/critic only when both the **world model is trustworthy** and the **actor-critic learning signal is healthy**.

This differs from DreamerV3, MBPO, M2AC/MACURA, and PAVE:
- DreamerV3 trains heavily from imagined latent rollouts: [DreamerV3](https://arxiv.org/abs/2301.04104)
- M2AC/MACURA gate model rollouts by model uncertainty: [M2AC](https://papers.nips.cc/paper_files/paper/2020/hash/77133be2e96a577bd4794928976d2ae2-Abstract.html), [MACURA](https://arxiv.org/abs/2405.19014)
- PAVE regularizes critic action-gradient geometry: [PAVE](https://arxiv.org/abs/2601.22970)
- SPECTRE-HGI’s novelty target is: **model-based imagination gated by actor-critic health signals such as `dQ/dA`, `q_pi_std`, critic action sensitivity, gradient health, and deterministic/stochastic divergence.**

## Key Changes
- Add an HGI trust score:
  - `model_trust`: based on world-model uncertainty, surprise KL, reward prediction error, and prior/post reconstruction agreement.
  - `critic_health`: based on `bridge/dqda_norm`, `bridge/q_pi_std`, `bridge/q_action_sensitivity`, `grad_health_mean`, guard tier, and deterministic/stochastic return gap.
  - `imag_trust = model_trust * critic_health`.

- Use `imag_trust` to control imagination:
  - below threshold: skip imagined actor/critic update for that batch;
  - medium trust: use short-horizon imagination;
  - high trust: use full configured imagination horizon;
  - always keep real replay critic learning active.

- Keep PAVE-style pieces as support, not the main novelty:
  - keep the current `q_action_sensitivity` floor;
  - optionally add a small Q-gradient consistency loss later;
  - treat PAVE as a baseline/prior-art comparison, not the identity of the algorithm.

- Fix current guard behavior before adding more complexity:
  - `grad_health_low` alone should not force `lr_scale=0.1` when `dqda_norm`, `q_pi_std`, and deterministic return are healthy;
  - reserve hard throttling for explosion, severe gradient collapse, or deterministic failure;
  - log `guard/throttle_reason`, `guard/grad_health_only`, and `guard/healthy_signal_override`.

## Config And Logs
- Add config knobs:
  - `HGI_ENABLED`
  - `HGI_MODEL_TRUST_MIN`
  - `HGI_CRITIC_HEALTH_MIN`
  - `HGI_SHORT_HORIZON`
  - `HGI_FULL_HORIZON`
  - `HGI_SKIP_UNHEALTHY_IMAGINATION`
  - `HGI_DET_STOCH_GAP_WARN`

- Add logs:
  - `hgi/model_trust`
  - `hgi/critic_health`
  - `hgi/imag_trust`
  - `hgi/effective_horizon`
  - `hgi/skipped_ratio`
  - `hgi/skip_reason_model`
  - `hgi/skip_reason_critic`
  - `hgi/det_stoch_gap_health`

## Test Plan
- Unit/synthetic checks:
  - high model trust + healthy critic gives full horizon;
  - high model trust + flat critic skips or shortens imagination;
  - low model trust + healthy critic skips imagination;
  - `det=0.05`, `stoch=16.0` marks deterministic failure;
  - low `grad_health_mean` alone does not force `lr_scale=0.1` when `dqda_norm` and `q_pi_std` are healthy.

- Training ablations:
  - current SPECTRE test as baseline;
  - model-trust-only gating, similar to M2AC/MACURA;
  - PAVE-lite only;
  - health-gate only;
  - full SPECTRE-HGI.

- Success criteria:
  - `return_test_det` tracks stochastic return instead of dying;
  - `bridge/dqda_norm` sustains above `1e-4`;
  - `bridge/q_pi_std` stays above `0.03`;
  - guard is not permanently stuck at `lr_scale=0.1`;
  - imagined updates are skipped when critic health collapses.

## Assumptions
- Do not change the world-model architecture yet.
- Do not unfreeze actor encoder/context gradients in this pass.
- Treat this as a research algorithm direction, not a guaranteed publication claim.
- Current live run should continue as evidence; the next implementation should start a clean named run such as `SPECTRE_HGI_test_10`.

## Checklist

### [x] 1. Fix guard grad-health-only throttling

Make `grad_health_low` alone a warning/override path when `dqda_norm`,
`q_pi_std`, and critic action sensitivity are healthy. Reserve hard throttling
for dQ/dA starvation/explosion, flat critic action signal, or severe gradient
health collapse. Log `guard/throttle_reason`, `guard/grad_health_only`, and
`guard/healthy_signal_override`.

### [x] 2. Add HGI trust metrics

Compute and log `hgi/model_trust`, `hgi/critic_health`, and `hgi/imag_trust`
from world-model uncertainty and actor-critic health.

### [x] 3. Gate imagination updates

Use `imag_trust` to skip, shorten, or allow imagined actor/critic updates while
leaving real replay critic learning active.

### [x] 4. Add HGI config knobs and run naming

Add the HGI config fields and prepare a clean `SPECTRE_HGI_test_10` run.

### [x] 5. Add HGI synthetic checks

Cover trust gating, deterministic/stochastic failure detection, and guard
override behavior.

### [x] 6. Add conservative HGI post-warmup ramp

After world-model warmup, cap healthy imagination to `HGI_SHORT_HORIZON` for
the first `HGI_POST_WARMUP_SHORT_STEPS` WM train steps before allowing full
`HGI_FULL_HORIZON`. Log ramp state and keep unhealthy/disabled gates skipping
as before.

### [x] 7. Add best evaluation checkpoint preservation

Preserve a separate best-eval trainer checkpoint when deterministic evaluation
sets a new high score above the configured threshold. Keep the normal scheduled
checkpoint flow unchanged and log `best_checkpoint/*` round metrics.

### [x] 8. Scrub absolute position from critic and world model

Remove `xyz` and absolute `progress` from privileged critic/RSSM state so the
training signal cannot directly memorize track coordinates. Keep ego-local
speed, gear, rpm, lidar, crash, progress gain, action history, images, and
context features.

### [x] 9. Document privileged-state leakage and fixes

Add a dedicated postmortem/design note explaining the map-aware teacher leak,
the critic/RSSM scrub, best-eval checkpointing, HGI post-warmup ramp, warmup
recommendation, required restart/reset protocol, and next-run watch metrics.

### [x] 10. Add actor churn regularization

Add a small EMA actor-head anchor that penalizes rapid deterministic pre-tanh
policy drift on real replay actor updates. Keep the anchor configurable, log
`loss_churn` and `churn/*`, and leave real replay, HGI, WM, and guard logic
otherwise unchanged.

### [x] 11. Add critic CHAIN value-churn regularization

Add a slow EMA Q-head anchor that penalizes rapid critic output drift on replay
states/actions. Keep TD learning, WM, HGI, actor/critic architectures, and guard
thresholds unchanged. Log `loss_q_churn`, `loss_q_churn_weighted`, and
`churn/q_*` so value churn is visible during runs.

### [x] 12. Add health-gated entropy floor controller

Replace fixed alpha-floor pressure with a per-action sidecar controller:
consolidate by lowering forced entropy when actor churn is high and critic
signal is healthy, but restore the baseline floor when dQ/dA, `q_pi_std`, Q
drift, model trust, gradient health, or guard state indicates starving/unsafe
actor-critic signal. Keep SAC temperature learning intact and log each
`entropy_health/*` gate and floor.

### [x] 13. Fix metrics logging schema

Replace the dynamic all-metrics CSV append path with two per-run files:
`RUN_NAME.metrics.jsonl` for full-fidelity metrics and
`RUN_NAME.stable.csv` for a fixed-width dashboard schema. Preserve dynamic
diagnostics such as `wm_imag/h*` in JSONL only, keep missing dashboard values
blank, and cover the 298/509/941-key schema expansion regression.

### [x] 14. Fix RSSM posterior-collapse pressure

Align the world-model KL objective with Dreamer-style dynamics/representation
balancing: train the prior toward a stopped posterior, apply a smaller
posterior-to-stopped-prior representation loss, restore nonzero free nats in
the live config, and add z-only latent-probe diagnostics so HGI can distinguish
true latent usefulness from decoder reconstruction through the deterministic
state path.

### [x] 15. Tighten latent bypass detection

Make HGI latent trust depend on the posterior beating zero/shuffled latent
ablations, not just KL or a single favorable baseline. Add a small decoder
latent-use loss so the actual RSSM decoder is pressured to reconstruct from
posterior `z`, and log the stricter ablation margin metrics.

### [x] 16. Add portable critic false-oracle safety

Add scale-free actor safety gates for policy-Q overconfidence, relative actor
churn spikes, and reward-energy liveness collapse. Feed the same health signals
into HGI critic health, and switch WM KL trust to a free-nats band so alive
latents are no longer scored as untrusted merely because raw KL is nonzero.

### [x] 17. Add deterministic skill-transfer feedback

### [x] 18. Fix entropy collapse feedback loops

Gate stagnation override and recovery drive by log_std sign so they never
boost exploration when the policy is already too noisy. Ungate log_std ceiling
penalty from det_skill_drive so it always pushes log_std below the ceiling.
Gate AWDB min_weight by positive mean Q advantage so the bridge only distills
from stochastic samples the critic actually ranks as better on average, and log
target vs applied AWDB minimum weight.

Latch the signed deterministic/stochastic eval gap back into the trainer so the
next actor updates know when stochastic samples are carrying skill that the
deterministic mean has not absorbed. Use that signal to raise AWDB distillation,
increase the minimum deterministic bridge weight, apply a soft log-std ceiling
penalty only during stochastic-advantage regimes, and expose dashboard metrics
for the controller.
