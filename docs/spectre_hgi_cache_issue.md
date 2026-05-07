# SPECTRE-HGI Actor-Health Cache Incident

Date: 2026-05-03

## Summary

During the first `SPECTRE_HGI_test_2` run, HGI logging appeared to work, but
the new `critic_health` score looked much worse than the raw bridge metrics.
The run showed healthy actor-critic signals in the `BRIDGE` and
`STABILITY GUARDS` sections, while `TRUST / UNCERTAINTY` reported
`hgi/critic_health` near `0.1`.

This was not a real critic collapse. It was a logging and trust-computation
bug caused by mixing per-step critic metrics with actor metrics that only exist
on policy-update steps.

## Symptoms

- HGI logs were present, but critic health was artificially low.
- `hgi/critic_health` hovered near `0.1`.
- `hgi/critic_health_dqda` and `hgi/critic_health_q_pi_std` were often `0.1`
  even when the live bridge section showed healthy values.
- HGI remained in warmup as intended:
  - `hgi/effective_horizon = 0`
  - `hgi/skipped_ratio = 1`
  - `hgi/skip_reason_warmup = 1`
  - `wm_imagined_steps = 0`
- Guard over-throttling was mostly fixed:
  - early `guard/lr_scale` stayed near `1.0`
  - small starvation tiers used mild scaling such as `0.975` or `0.95`, not
    permanent `0.1`

## Evidence

Raw bridge metrics from the live run were not dead:

- `bridge/dqda_norm` was around `1e-4` to `2.8e-4`.
- `bridge/q_pi_std` was around `0.045` to `0.096`.
- `bridge/q_action_sensitivity` was around `0.00285` to `0.00767`.
- `guard/lr_scale` was `1.0`, then only mildly reduced in early starvation
  cases.

But HGI reported:

- `hgi/critic_health = 0.1`
- `hgi/critic_health_dqda = 0.1`
- `hgi/critic_health_q_pi_std = 0.1`

The mismatch was the clue: raw `BRIDGE` health was usable, but HGI was scoring
many rows as if `dQ/dA` and `q_pi_std` were missing or zero.

## Root Cause

The trainer performs critic updates more often than policy updates.

Metrics such as:

- `bridge/q_action_sensitivity`
- critic loss
- world-model loss

are available on every training update.

But actor-health metrics such as:

- `bridge/dqda_norm`
- `bridge/q_pi_std`
- `bridge/q_pi_action_corr_d*`
- guard tier and actor guard details

are only produced on policy-update steps.

HGI trust was computed every update. On critic-only updates, missing
`bridge/dqda_norm` and `bridge/q_pi_std` were treated as `0.0`. If policy
updates happen roughly once every ten critic updates, the averaged HGI health
naturally falls toward `0.1` even when the actor signal is actually healthy on
the policy-update rows.

In short:

`missing actor-health metric` was accidentally interpreted as
`dead actor-health metric`.

## Fix Applied

The HGI trust path now caches the latest valid actor-health snapshot from a real
policy update and reuses it during critic-only updates.

Changed file:

- `tmrl/custom/custom_algorithms.py`

New diagnostics:

- `hgi/actor_health_cache_used`
- `hgi/actor_health_cache_fill_count`
- `hgi/actor_health_fresh_present`

Expected behavior:

- On fresh policy-update rows:
  - `hgi/actor_health_fresh_present = 1`
  - `hgi/actor_health_cache_used = 0`
- On critic-only rows after at least one policy update:
  - `hgi/actor_health_fresh_present = 0`
  - `hgi/actor_health_cache_used = 1`
- `hgi/critic_health_dqda` and `hgi/critic_health_q_pi_std` should no longer
  collapse to `0.1` just because actor metrics were not emitted on that row.

## Tests Added

Changed file:

- `tests/test_hgi_guard.py`

Added coverage for:

- fresh actor-health rows reporting no cache use;
- critic-only rows reusing cached `dqda_norm` and `q_pi_std`;
- cached health preventing fake HGI critic-health collapse.

Verification:

```text
py_compile: OK
unittest: Ran 22 tests in 0.002s OK
```

## What To Watch Next

In the next live `SPECTRE_HGI_test_2` logs, check:

- `hgi/actor_health_cache_used`
- `hgi/actor_health_fresh_present`
- `hgi/critic_health_dqda`
- `hgi/critic_health_q_pi_std`
- `hgi/critic_health`
- `bridge/dqda_norm`
- `bridge/q_pi_std`
- `bridge/q_action_sensitivity`
- `guard/lr_scale`

Healthy expected pattern during warmup:

- `hgi/effective_horizon = 0`
- `hgi/skipped_ratio = 1`
- `hgi/skip_reason_warmup = 1`
- `wm_imagined_steps = 0`
- `hgi/model_trust` should improve as WM KL/reconstruction/reward error improve.
- `hgi/critic_health` should reflect the latest real actor-health snapshot,
  not collapse just because the current row is critic-only.

## Conclusion

The problem was not that HGI itself was conceptually wrong. The HGI gate was
reading sparse actor-health metrics as if they were dense per-update metrics.
That made the trust score pessimistic and misleading.

The cache fix makes HGI match the training loop reality: critic updates are
dense, actor-health measurements are sparse, and trust scoring needs to carry
forward the latest valid actor-health evidence until the next policy update.
