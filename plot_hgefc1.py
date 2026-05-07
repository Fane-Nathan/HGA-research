"""Focused plot for SPECTRE_HGI_HGEFC_1: pinpoint the brittle-attractor failure mode.

Data source: hgefc1_history.csv (pulled fresh from W&B by fetch_hgefc1_history.py
to bypass the local CSV's multi-schema concatenation).

Four panels on a shared x-axis (wall-clock hours since run start):
  1. log_std_min vs log_std_mean        -> asymmetric entropy collapse
  2. q_pi_std vs Q_PI_STD_HEALTH (0.03) -> action-flat critic signal
  3. q_action_sensitivity vs floor 0.05 -> is the regularizer biting?
  4. return_test_det / _stoch / gap     -> brittle policy
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV = "hgefc1_history.csv"
OUT = "hgefc1_diagnosis.png"

df = pd.read_csv(CSV).sort_values("_step").reset_index(drop=True)
t = df["_runtime"].to_numpy() / 3600.0  # hours since run start

fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

# --- 1. log_std collapse ---
ax = axes[0]
ax.plot(t, df["debug_log_std_mean"], label="log_std_mean", color="tab:blue", lw=1.4)
ax.plot(t, df["debug_log_std_min"],  label="log_std_min",  color="tab:red",  lw=1.4)
ax.axhline(-20, color="grey", ls=":", lw=0.8, label="code clamp (-20)")
ax.set_ylabel("log std")
ax.set_title("Entropy is NOT collapsing — log_std_min ~-1 to -2.3, log_std_mean ~-0.2  (σ_min≈0.1, σ_mean≈0.8). Earlier 'σ_min→1e-4' read was a column-misalignment artifact.")
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.3)

# --- 2. q_pi_std vs health threshold ---
ax = axes[1]
ax.plot(t, df["bridge/q_pi_std"], color="tab:purple", lw=1.4, label="q_pi_std")
ax.axhline(0.03, color="black", ls="--", lw=1.0, label="health threshold 0.03 (Q_PI_STD_HEALTH_THRESHOLD)")
ax.set_ylabel("q_pi_std")
ax.set_yscale("log")
ax.set_title("q_pi_std is HEALTHY — sits at ~0.047 above the 0.03 threshold. Critic ensemble disagreement is fine.")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3, which="both")

# --- 3. q_action_sensitivity vs regularizer floor ---
ax = axes[2]
ax.plot(t, df["bridge/q_action_sensitivity"], color="tab:green", lw=1.4,
        label="q_action_sensitivity")
ax.axhline(0.05, color="black", ls="--", lw=1.0,
           label="Q_ACTION_STD_FLOOR = 0.05 (regularizer target)")
ax.set_ylabel("q_action_sensitivity")
ax.set_title("q_action_sensitivity stuck at ~0.008  —  6× below the 0.05 target. The regularizer is firing but losing badly.")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# --- 4. det vs stoch returns ---
ax = axes[3]
det = df["return_test_det"].copy()
stoch = df["return_test_stoch"].copy()
# wandb may log 0.0 on rounds where det wasn't re-evaluated. Mark those nan.
det_skipped = (det == 0.0)
det_plot = det.where(~det_skipped, np.nan)
ax.plot(t, det_plot, "o-", color="tab:orange", lw=1.0, ms=3,
        label="return_test_det (eval rounds only)")
ax.plot(t, stoch, "x-", color="tab:cyan", lw=0.8, ms=4, alpha=0.7,
        label="return_test_stoch")
best = df["best_checkpoint/best_return_test_det"].max()
ax.axhline(best, color="tab:orange", ls=":", lw=0.8, alpha=0.7,
           label=f"best_return_test_det = {best:.1f} (locked at epoch 1, round 1)")
ax.set_ylabel("return")
ax.set_xlabel("wall-clock hours since run start")
ax.set_title("Plateau at ~82: det and stoch BOTH reach 60–82 regularly, but best is locked from step 11 onward. No upward push.")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("SPECTRE_HGI_HGEFC_1 — diagnosis: Q-landscape too flat in action space; regularizer underweighted",
             fontsize=12, y=0.995)
plt.tight_layout()
plt.savefig(OUT, dpi=130)
print(f"saved {OUT}")

# --- numeric summary ---
def stat(col):
    s = df[col]
    return f"min={s.min():.4g}  median={s.median():.4g}  max={s.max():.4g}  last={s.iloc[-1]:.4g}"

print()
print("=== numeric summary ===")
print(f"rows                       : {len(df)}")
print(f"wall-clock                 : {t[-1]:.2f} h")
print(f"best_return_test_det       : {df['best_checkpoint/best_return_test_det'].max():.3f}")
best_idx = df["best_checkpoint/best_return_test_det"].idxmax()
ctx = []
for k in ("epoch", "round"):
    if k in df.columns:
        ctx.append(f"{k}={int(df.loc[best_idx, k])}")
print(f"first hit at step          : {int(df.loc[best_idx, '_step'])}  ({', '.join(ctx) if ctx else 'epoch/round not in wandb log'})")
print()
print(f"debug_log_std_mean         : {stat('debug_log_std_mean')}")
print(f"debug_log_std_min          : {stat('debug_log_std_min')}")
print(f"bridge/q_pi_std            : {stat('bridge/q_pi_std')}")
print(f"bridge/q_action_sensitivity: {stat('bridge/q_action_sensitivity')}")
print(f"bridge/dqda_norm           : {stat('bridge/dqda_norm')}")
print(f"debug_alpha_steer          : {stat('debug_alpha_steer')}")
print()
det_nz = df.loc[det != 0.0, "return_test_det"]
print("return_test_det (eval rounds only):")
print(det_nz.describe().to_string())
print("\nreturn_test_stoch:")
print(df["return_test_stoch"].describe().to_string())
gap = det_nz - df.loc[det_nz.index, "return_test_stoch"]
print("\ndet - stoch gap (when det evaluated):")
print(gap.describe().to_string())
