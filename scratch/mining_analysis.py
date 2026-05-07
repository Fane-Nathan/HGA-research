"""
Advanced data-mining of SPECTRE HGI training logs.
Goal: surface NOVEL phenomena, not retell metrics.

Strategy:
  1. Detect regime changes (PELT-style on multivariate series)
  2. Latent-space collapse fingerprints (RSSM posterior)
  3. Causal lead/lag between guard intervention and downstream metrics
  4. Reward distribution drift (target vs predicted)
  5. Anomaly: where do "111-step" episodes cluster?
  6. Q-value pathology vs entropy floor escalation
"""
import numpy as np
import pandas as pd
from pathlib import Path

# --- load ---
ROOT = Path(r"c:/Users/felix/OneDrive/Documents/tmrl-test")
df = pd.read_csv(ROOT / "hgefc1_history.csv")
df = df.sort_values("_step").reset_index(drop=True)
df["t"] = np.arange(len(df))
print(f"Rows: {len(df)}, columns: {df.shape[1]}")
print(f"_step span: {df['_step'].iloc[0]} → {df['_step'].iloc[-1]}, "
      f"runtime: {df['_runtime'].iloc[-1]:.0f}s")

# ----- 1. Posterior-collapse fingerprint -----
print("\n=== 1. POSTERIOR / LATENT COLLAPSE ===")
collapse = df[["t", "wm/kl_mean", "wm/z_t_dim_std_min", "wm/z_t_dim_std_mean",
               "wm/z_t_dead_dim_ratio_1e3", "wm/post_logvar_mean",
               "wm/prior_logvar_mean"]].copy()
collapse["kl_log"] = np.log10(collapse["wm/kl_mean"].clip(lower=1e-9))
collapse["dim_std_min_log"] = np.log10(collapse["wm/z_t_dim_std_min"].clip(lower=1e-9))
snap = [0, 5, 15, 30, 45, len(df)-1]
print(collapse.iloc[snap].to_string(index=False))
print("KL halving time (rounds):",
      np.argmax(collapse["wm/kl_mean"].values <
                collapse["wm/kl_mean"].iloc[0] / 2))
print("Final KL / initial KL:",
      collapse["wm/kl_mean"].iloc[-1] / collapse["wm/kl_mean"].iloc[0])

# ----- 2. Q-value runaway vs reward signal -----
print("\n=== 2. Q vs REWARD DISTRIBUTION ===")
qr = df[["t", "bridge/q_real_mean", "bridge/q_pi_mean", "bridge/q_pi_std",
         "wm/reward_pred_mean", "wm/reward_target_mean",
         "wm/reward_target_std", "loss_critic"]].copy()
qr["q_to_r_ratio"] = qr["bridge/q_real_mean"] / qr["wm/reward_target_mean"].replace(0, np.nan)
print(qr.iloc[snap].to_string(index=False))

# Reward target std shrinkage = signal collapse
print(f"\nreward_target_std: first={qr['wm/reward_target_std'].iloc[0]:.4f} "
      f"last={qr['wm/reward_target_std'].iloc[-1]:.4f}")

# ----- 3. Guard intervention lead/lag -----
print("\n=== 3. GUARD CAUSAL LEAD/LAG ===")
g = df[["t", "guard/tier", "guard/q_shield_triggered",
        "guard/dqda_critical", "guard/grad_health_severe",
        "entropy_health/floor_target_steer",
        "entropy_health/preserve_active",
        "entropy_health/consolidate_active",
        "return_test_det", "loss_actor"]].copy()

# Lead/lag corr: does guard tier predict future returns?
def lagcorr(a, b, max_lag=5):
    out = {}
    for k in range(-max_lag, max_lag + 1):
        if k < 0:
            x, y = a.iloc[:k], b.iloc[-k:]
        elif k > 0:
            x, y = a.iloc[k:], b.iloc[:-k]
        else:
            x, y = a, b
        x, y = x.reset_index(drop=True), y.reset_index(drop=True)
        valid = (~x.isna()) & (~y.isna())
        if valid.sum() < 10:
            out[k] = np.nan; continue
        out[k] = x[valid].corr(y[valid])
    return out

print("Cross-corr guard/tier ↔ return_test_det (lag k = guard leads by k rounds):")
for k, v in lagcorr(g["guard/tier"], g["return_test_det"]).items():
    print(f"  k={k:+d}  r={v: .3f}")

print("\nCross-corr floor_target_steer ↔ return_test_det:")
for k, v in lagcorr(g["entropy_health/floor_target_steer"],
                    g["return_test_det"]).items():
    print(f"  k={k:+d}  r={v: .3f}")

# ----- 4. Episode-length bimodality ('111 vs 1000' truncation) -----
print("\n=== 4. EPISODE-LENGTH STRUCTURE ===")
el = df[["t", "episode_length_test", "episode_length_test_det",
         "episode_length_test_stoch", "return_test_det",
         "return_test_stoch", "hgi/skipped_ratio"]].copy()
el = el[el["episode_length_test"] > 0]
print(f"Length distribution (det):")
bins = [0, 110, 120, 200, 400, 700, 999, 1001]
print(pd.cut(el["episode_length_test_det"], bins).value_counts().sort_index())
# 111 = stall-out floor (likely min eval steps); 1000 = truncation
n_111 = (el["episode_length_test_det"].between(110, 112)).sum()
n_1000 = (el["episode_length_test_det"] >= 999).sum()
print(f"\n# eval det runs at floor (~111): {n_111}/{len(el)}")
print(f"# eval det runs at ceiling (1000): {n_1000}/{len(el)}")

# ----- 5. Det/stoch gap as policy-determinism proxy -----
print("\n=== 5. DET/STOCH GAP DRIFT ===")
gap = df["return_test_det"] - df["return_test_stoch"]
print(gap.describe())
print("Gap sign flips:", ((np.sign(gap).diff().fillna(0) != 0)).sum())

# ----- 6. Imagination quality vs trust gating -----
print("\n=== 6. IMAGINATION GATING DYNAMICS ===")
ig = df[["t", "hgi/model_trust", "hgi/critic_health", "hgi/imag_trust",
         "hgi/skipped_ratio", "hgi/effective_horizon",
         "wm_imagined_q_loss", "wm_imagined_steps",
         "imag_actor_loss"]].copy()
print(ig.iloc[snap].to_string(index=False))

# Find points where trust collapses
trust_drops = (ig["hgi/imag_trust"].diff() < -0.3)
print(f"\nLarge imag_trust drops (>0.3): rows {df.index[trust_drops].tolist()}")

# ----- 7. Anomaly: dimensions of input feature collapse -----
print("\n=== 7. INPUT FEATURE DIMENSION HEALTH ===")
inp = df[["t", "wm_input/state_dim_std_min", "wm_input/state_dim_std_mean",
          "wm_input/state_dead_dim_ratio_1e3",
          "wm_input/action_dim_std_min", "wm_input/action_dim_std_mean"]]
print(inp.iloc[[0, 30, 59]].to_string(index=False))

# ----- 8. Surprising correlations across the whole dataframe -----
print("\n=== 8. SURPRISE CORRELATIONS ===")
num = df.select_dtypes(include=[np.number])
# limit to columns with enough variance
num = num.loc[:, num.std() > 1e-6]
target = "return_test_det"
if target in num.columns:
    corrs = num.corrwith(num[target]).dropna().sort_values()
    interesting = pd.concat([corrs.head(10), corrs.tail(11).iloc[:-1]])
    print("Top negative / top positive correlates of return_test_det:")
    print(interesting.to_string())
