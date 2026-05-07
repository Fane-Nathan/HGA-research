# SPECTRE Training Collapse — Debug Report

**Files analysed:** `SPECTRE_new_policy_test_4.csv`, `..._test_5.csv`, `..._test_6.csv`
**Approach:** the three runs are sibling seeds of the same policy. I mined for shared collapse signals and run-specific divergences. Every numeric claim below was re-verified against the raw CSVs in `verify_claims.py`.

---

## TL;DR — Ranked hypotheses

1. **Root cause (high confidence):** the policy's entropy is being pushed *upward* over training, not downward. `debug_log_std_mean` drifts up in every run (e.g. test_5: −0.294 → −0.134 in 5 rounds → σ≈0.88), while `debug_alpha_steer` is **clamped at its floor 0.08** for 100% of all rows in all three runs. SAC wants to lower α to make the policy more deterministic; the floor prevents that, so the action distribution stays noisy and the car crashes.
2. **Amplifier #1 (high confidence):** the guard system has *no hysteresis*. In test_6 `guard/lr_scale` drops 1.0 → 0.73 → 0.1 in three rounds and stays at 0.1 for 24 of 27 rounds. In test_4 `guard/actor_mu_step_blocked = 1` for 20 of the last 21 rounds. Once the guard fires, the policy is permanently kneecapped.
3. **Amplifier #2 (high confidence):** `wm_kl_anti_collapse = 0.0` for **every row of test_6** (vs always >0 in test_5, mostly >0 in test_4). The world-model anti-collapse term appears disabled in test_6, and that run's `wm_total_loss` starts at 0.416 — 18× bigger than test_4's 0.023 in round 0.
4. **Suspicious silent regression:** `debug_target_entropy_steer` is a constant **−0.300** in test_5 and test_6 but **adapts** in test_4 (−0.300 → −0.369). Either the target-entropy schedule was disabled, or it never ran enough in 5/27 rounds to move.
5. **Architectural concern:** `bridge/dqda_norm` lives at ~1e-4 across all runs. If that's the raw critic-to-actor gradient signal, the actor is essentially un-driven by Q, and it drifts on entropy noise alone.
6. **Logging integrity:** test_4's CSV switches from 152 columns to 797 columns at line 52 (epoch 5, round 0). 30 rows had to be dropped from analysis. Either the logger schema changed mid-run, or two runs were concatenated.

---

## Diagnostic plots

`spectre_p1_returns_losses.png`, `spectre_p2_world_model.png`, `spectre_p3_actor_entropy.png`, `spectre_p4_guard.png` — all in this folder.

---

## 1. Per-run collapse signature

| run | rounds | wall-clock | peak `return_train` | final `return_train` | failure mode |
|-----|--------|-----------|---------------------|----------------------|---------------|
| test_4 | 50 (kept of 80) | 11 962 s | **26.12** at idx 22 | 9.52 | climbs, then plateaus around 8–10 once `actor_mu_step_blocked` engages at idx 29 |
| test_5 |  5 | 833 s | 22.81 at idx 0 | **0.71** | running great (1000-step episodes), then **catastrophic drop at round-idx 3**: 21.81 → 0.21, episode length 1000 → 111 |
| test_6 | 27 | 1 220 s | 2.42 at idx 0 | 1.85 | crashes within 3 rounds; `lr_scale` collapses to 0.1 and never recovers |

### Detail — test_5 (round 2 → round 3)

```
return_train          22.81  22.81  21.81  0.21  0.71
episode_length_train    987    987   1000   111   154   <- car crashes
debug_log_std_mean   −0.29  −0.10  −0.16  −0.09  −0.13  <- σ rises to 0.92
debug_alpha_steer     0.08   0.08   0.08   0.08   0.08  <- pinned at floor
wm_kl, wm_total_loss  fine and falling throughout
loss_critic, dqda_norm  no spike before collapse
```
Nothing in the loss/world-model/gradient signals warns of the crash; the policy just becomes too noisy to drive.

### Detail — test_6 (the lr_scale trap)

```
idx round return_train  guard/tier  guard/lr_scale  actor_stab  q_drop
  0    0       2.42        0.0          1.00          0.0       0.0
  1    1       2.42        0.0          1.00          0.0       0.0
  2    2       0.63        0.3          0.73          0.3       0.0
  3    3       0.02        1.0          0.10          1.0       0.0
  4    4       0.03        1.0          0.10          1.0       0.0
  ...                       (1.0 / 0.10 / 1.0 / 0.0 forever)
```
The guard reacts to falling `grad_health_mean` (0.90 → 0.75 → 0.63 → 0.52). Tier never resets even when return improves later (return reaches 2.34 at idx 25 and tier is still 1.0, lr_scale still 0.1). `q_shield_triggered` and `tier3_*` never fire at all — only the LR-scale lever pulled, and it's stuck.

### Detail — test_4 (actor_mu_step_blocked)

```
idx 28: return 3.81, actor_stab=0.0, mu_blocked=0.0
idx 29: return 15.84, actor_stab=1.0, mu_blocked=1.0  <- engages
... mu_blocked stays 1 for 20 of remaining 21 rounds
```
Mean actor updates frozen → policy cannot improve past its plateau (8–10) and never re-touches the earlier 26.12 peak.

---

## 2. Cross-run common precursor: rising log_std

| run | log_std_mean start | log_std_mean end | slope per round | σ implied at end |
|-----|-------------------:|-----------------:|----------------:|-----------------:|
| test_4 | −0.266 | −0.030 | +0.0048 | 0.97 |
| test_5 | −0.294 | −0.134 | +0.0401 | 0.88 |
| test_6 | −0.945 | −0.189 | +0.0291 | 0.83 |

Every run drifts toward σ ≈ 1 on each action dim. For a precision driving task, σ ≈ 0.9 on the steering action means the policy is essentially being slapped with random ±1-ish steering perturbations every step — incompatible with completing a lap. The slope is steeper in test_5/6, which is why they crash quickly; test_4 slopes slowly and so survives longer before guards engage.

This is consistent with `α` being clamped at the floor: SAC wants α↓ to encourage determinism, but `α=α_floor=0.08` forever, so the entropy term keeps pushing log_std up.

---

## 3. The α floor is inconsistent across action dims

```
debug_alpha_floor_steer = 0.08    (steering — most sensitive)
debug_alpha_floor_gas   = 0.05
debug_alpha_floor_brake = 0.05
```
And the actuals:
```
debug_alpha_steer = 0.0800 in 100% of rows of all 3 runs   (pegged)
debug_alpha_gas   = 0.0500–0.0515  (slightly above floor, regulating)
debug_alpha_brake = 0.0500–0.0516  (slightly above floor, regulating)
```
Only steer is permanently saturating. Gas/brake α's are tracking. **Lowering `α_floor_steer` (or removing it) is the single most leveraged fix** — it lets the SAC update reach the deterministic regime steering needs.

---

## 4. The world-model anti-collapse term is missing in test_6

```
wm_kl_anti_collapse:
  test_4: zero for 13 of 50 rows, otherwise up to 8.2e-4
  test_5: always > 0 (1.7e-5 ... 7.8e-4)
  test_6: exactly 0.0 for ALL 27 rows
```
And the WM is visibly less healthy in test_6's first round: `wm_total_loss = 0.4161` vs test_4's 0.0229 and test_5's 0.0343. By round 3 it converges, but the spike feeds noisy gradients into the bridge → into the actor, exactly while the guard is deciding whether to fire. This may be why the guard fires so early in test_6 specifically.

Action item: confirm whether the anti-collapse coefficient was set to 0 deliberately in the test_6 config.

---

## 5. Target-entropy schedule may be silently off

```
debug_target_entropy_steer:
  test_4: 30 distinct values, evolves −0.300 → −0.369
  test_5: 1 value (−0.300)        ← never adapts
  test_6: 1 value (−0.300)        ← never adapts
```
If a target-entropy decay schedule was meant to push the policy toward determinism over time, it's not running in test_5/6. (test_5 only has 5 rounds so the absence is ambiguous, but test_6 has 27 rounds and the schedule still hasn't moved.) Combined with the α floor, the policy has *no mechanism* to become more deterministic.

---

## 6. The actor receives a vanishingly small gradient signal

`bridge/dqda_norm` (the ‖∂Q/∂a‖ flowing into the actor):

| run | min | median | max |
|-----|-----|--------|-----|
| test_4 | 3.7e-5 | 4.4e-5 | 2.2e-4 |
| test_5 | 8.2e-5 | 9.0e-5 | 2.4e-4 |
| test_6 | 5.7e-5 | 7.1e-5 | 2.7e-4 |

These are extremely small numbers. If this is the raw gradient (not normalized) feeding the deterministic actor update, the actor is almost not being trained by the critic — only by the entropy term. Worth checking the bridge code: is dqda being scaled, clipped, or zeroed somewhere upstream? `guard/dqda_min_norm = 7e-5` is the same fixed value across all runs, hinting at a hard floor that the gradient is already at.

---

## 7. Logging integrity — test_4 schema bump

```
test_4 CSV: 81 lines
  rows 1..51: 152 columns (matches header)
  rows 52..81: 797 columns  (header still has 152 names)
```
Row 52 is `wall_clock_seconds=12334.9, epoch=5, round=0` — only 372 s after row 51. The bump aligns with an epoch boundary, suggesting the logger schema was updated mid-run (likely a per-dimension dump was added). 30 rows were unusable for cross-run comparison. Flag: ensure logger version is fixed for a run before starting it; or, if you bumped the schema deliberately, write a separate file rather than continue appending.

---

## 8. Other observations (lower priority)

- `kl_div_loss` always starts at ~19 in round 0 and falls to ~0.05 by round 1. This is an initialization artifact — first-step target distribution far from the policy. Not a bug, but consider warm-starting it.
- `memory_len` is identical for round 0 and round 1 in every run (no replay grew yet). Expected.
- `idle_time` reaches 71 s in test_4 round 0 and 9 s in test_5 round 0 — not present in test_6. Likely first-round env warmup; not a bug.
- `guard/grad_health_min = 0.01` is a fixed value in every row of every run — it's the floor of a clipped quantity, not a measurement.

---

## Recommended next experiments

In priority order (each is small):

1. **Lower `α_floor_steer`** to 0.02 (matching gas/brake) and re-run the test_5 seed. If the policy stops collapsing, the α-floor hypothesis is confirmed.
2. **Add hysteresis to the tier guard** (test_6): require N consecutive healthy rounds before stepping `lr_scale` back up. Currently it's a one-way ratchet.
3. **Verify `wm_kl_anti_collapse` is enabled in the test_6 config** — re-run with it on; expect WM round-0 loss to drop from 0.42 toward test_4's 0.02.
4. **Restore the target-entropy schedule** in test_5/6 (or confirm it's intentionally off).
5. **Audit `bridge/dqda_norm`** path: print raw vs scaled values; if it's already on the order of 1e-4, the actor is essentially un-driven by the critic.
6. **Pin the logger schema** for the duration of a run, or rotate to a new file when changing it.

---

*Report generated 2026-05-01. Underlying scripts: `load_data.py`, `summary.py`, `collapse_dive.py`, `verify_claims.py`, `make_plots.py` in the outputs folder.*
