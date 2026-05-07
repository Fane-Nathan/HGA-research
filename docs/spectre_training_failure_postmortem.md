# SPECTRE Training Failure Postmortem

## Executive Summary

The major failures were not one single bad hyperparameter. They were a sequence
of different failure modes:

- test_6 failed by instability: dQ/dA explosion, entropy runaway, and guard blind spots.
- test_7 exposed analysis and evaluation problems: deterministic return did not reliably track stochastic return, and CSV rows became malformed after the metric schema changed.
- test_8 failed by starvation: stochastic policy found useful behavior, but deterministic policy stayed dead because the critic became action-flat and actor updates were permanently throttled.
- The code audit found a pipeline bug behind test_8: context was declared as reward-aware but replay and online actor context omitted reward, hybrid replay dropped crash/progress_gain, and dual eval could mark runs healthy while deterministic return was near zero.

The test_10 fix focuses on the pipeline: preserve privileged fields, add reward
to context, make deterministic eval the primary health signal, and keep the
critic/guard starvation safeguards.

## Test 6: Explosion And Entropy Runaway

Symptoms:

- dQ/dA exploded far beyond the normal range.
- Entropy coefficient increased after the actor/critic shock.
- Existing actor guard watched low gradient health, but not high dQ/dA.
- Q-shield threshold was too large for the observed Q-value scale.

Evidence:

- Logs showed dQ/dA moving thousands of times above the intended band.
- Entropy coefficient rose after collapse instead of stabilizing exploration.
- Q values were small, so an absolute Q-drop threshold around 2.0 could not trigger.

Cause:

- No explicit dQ/dA explosion guard.
- Entropy auto-tuning overreacted after policy shock.
- Q-shield threshold was scale-mismatched to Q values near 0.03 to 0.2.

Fix applied:

- Added dQ/dA gradient clipping.
- Added alpha ceiling.
- Added high-dQ/dA guard detection.
- Lowered Q-shield threshold and made it relative to current Q scale.

Future warning thresholds:

- `bridge/dqda_norm > 0.01`: explosion warning.
- `bridge/dqda_norm > 0.1`: critical explosion.
- `guard/alpha_ceiling_hit = 1`: entropy pressure is at the ceiling.
- Large Q drop relative to current Q mean: Q-shield should trigger.

## Test 7: Evaluation Split And CSV Drift

Symptoms:

- Stochastic and deterministic returns diverged.
- Later CSV rows had more fields than the original header.
- Analysis became risky because metrics after schema drift could be misaligned.

Evidence:

- Deterministic return stayed much lower than stochastic return.
- CSV header had fewer columns than later rows after new diagnostics were added.
- Rows widened when additional bridge/world-model metrics appeared after the
  header had already been written.

Cause:

- Dual eval logged both modes, but the alias `return_test` followed stochastic
  return in dual mode.
- CSV writer wrote the header only once using the first metric set, then appended
  later rows with a larger metric set.

Fix planned/applied:

- Keep explicit `return_test_det` and `return_test_stoch`.
- Add `return_test_det_stoch_gap`.
- Use deterministic return for health in dual mode.
- Treat CSV schema drift as an analysis hazard when comparing older runs.

Future warning thresholds:

- CSV row width differing from header width: invalidate automated aggregate analysis.
- `return_test_stoch - return_test_det > 0.5` while stochastic is healthy: deterministic bridge warning.

## Test 8: Actor Starvation

Symptoms:

- Stochastic return could be high, but deterministic return stayed near zero.
- `bridge/dqda_norm` collapsed to tiny values.
- `bridge/q_pi_std` and `bridge/q_real_std` fell, meaning Q became nearly action-flat.
- Guard tier stayed active and actor LR was repeatedly throttled to `0.1x`.

Evidence:

- test_8 early rows showed dQ/dA falling toward or below `1e-4`.
- Q variation across policy actions shrank toward roughly `0.02`.
- Live pre-fix epoch 3 to 4 logs showed stochastic return reaching strong values
  while deterministic return remained around `0.05` to `0.13`.
- Guard logs showed `guard/tier = 1` and `guard/lr_scale = 0.1` while explosion
  flags were false.

Cause:

- The critic learned a value surface with too little local action sensitivity.
- The actor then received almost no Q-gradient.
- AWDB was too weak when advantage normalization became flat.
- The guard treated ordinary starvation like a risky actor update and throttled
  recovery too hard.

Fix applied:

- Added critic action-sensitivity regularizer.
- Added minimum AWDB weight.
- Increased deterministic regularization.
- Changed starvation LR recovery to `STARVATION_LR_SCALE`.
- Kept explosion protection, dQ/dA clipping, and alpha ceiling.

Future warning thresholds:

- `bridge/dqda_norm < 1e-4` sustained: starvation warning.
- `bridge/q_pi_std < 0.03` sustained: action-flat critic warning.
- `guard/lr_scale = 0.1` with `guard/dqda_explosion = 0`: guard is likely blocking recovery.
- `return_test_det < 0.5` while `return_test_stoch` is healthy: deterministic policy is not learning.

## Code Audit Findings

Missing reward in context:

- `CONTEXT_INPUT_DIM` declared a 24-dim context containing reward.
- Replay actually built 23 dims: speed, lidar, action.
- Online actor stored `_online_prev_reward` but did not append it to context.
- Result: context could not directly encode which recent actions produced reward.

Dropped privileged fields:

- The hybrid environment produced `crash` and `progress_gain`.
- The preprocessor preserved them.
- The hybrid sample compressor dropped them, and replay reconstructed old 9-field
  observations with zeros for those fields.
- Result: critic/world-model privileged signals were weaker than intended.

Dual eval masked deterministic failure:

- In dual mode, `return_test` followed stochastic return.
- Stall diagnostics averaged deterministic and stochastic return.
- Result: a run could be reported as healthy even when deterministic policy was dead.

Actor representation detachment:

- Actor update detaches fused features and context z.
- This is intentional for stability, but it means the actor head cannot reshape
  the encoder/context representation when Q becomes flat.
- Result: deterministic recovery depends heavily on Q action sensitivity and AWDB.

## Test 10 Fix Strategy

Implementation:

- Preserve `crash` and `progress_gain` in compressed hybrid samples and replay.
- Return 11-field hybrid observations from replay.
- Build replay and online context as 24 dims: speed, lidar, action, reward.
- Update the context encoder split so reward and reward delta feed the state stream.
- Log context reward presence and statistics.
- Use deterministic return as the health signal in dual eval mode.
- Keep stochastic return as a separate exploration signal.

Success criteria:

- `return_test_det` repeatedly exceeds `0.5`.
- Deterministic return starts tracking stochastic return.
- `bridge/dqda_norm` sustains above `1e-4`.
- `bridge/q_pi_std` stabilizes above `0.03`.
- Guard LR is not permanently stuck at `0.1`.

Operational note:

- Start test_10 fresh with `RESET_TRAINING = true`. Old replay is only kept
  compatible enough to avoid crashes; it should not be used as evidence that the
  new context/replay path is learning correctly.
