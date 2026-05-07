"""Pass 3: alpha behavior, regime clustering, predictive features for return drops."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"c:/Users/felix/OneDrive/Documents/tmrl-test")
df = pd.read_csv(ROOT / "hgefc1_history.csv").sort_values("_step").reset_index(drop=True)

# === H. ALPHA temperature behavior ===
print("=== H. ENTROPY TEMPERATURE (alpha) BEHAVIOR ===")
for c in ["debug_alpha_steer", "debug_alpha_gas", "debug_alpha_brake",
          "debug_alpha_floor_steer", "debug_alpha_floor_gas", "debug_alpha_floor_brake"]:
    if c in df.columns:
        v = df[c]
        print(f"{c:30s}  early={v.iloc[:5].mean():.5f}  mid={v.iloc[28:32].mean():.5f}  late={v.iloc[-5:].mean():.5f}  range=[{v.min():.5f},{v.max():.5f}]")

# Is alpha hitting its floor (== floor injection mode)?
for ax in ["steer", "gas", "brake"]:
    a = df[f"debug_alpha_{ax}"]
    fl = df[f"debug_alpha_floor_{ax}"]
    on_floor = ((a - fl).abs() < 1e-5)
    print(f"  alpha_{ax}: on-floor in {on_floor.sum()}/{len(df)} rounds ({on_floor.mean()*100:.0f}%)")

# entropy_coef itself
if "entropy_coef" in df.columns:
    ec = df["entropy_coef"]
    print(f"\nentropy_coef: early={ec.iloc[:5].mean():.4f}  late={ec.iloc[-5:].mean():.4f}  trajectory monotone? {ec.is_monotonic_increasing or ec.is_monotonic_decreasing}")

# === I. Hidden regimes via unsupervised clustering on key health metrics ===
print("\n=== I. HIDDEN REGIME CLUSTERS ===")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

feat_cols = [
    "wm/kl_mean", "bridge/q_real_mean", "bridge/q_pi_std",
    "hgi/imag_trust", "hgi/critic_health", "wm/reward_target_std",
    "entropy_health/floor_target_steer", "loss_actor", "loss_critic",
    "return_test_det",
]
feat_cols = [c for c in feat_cols if c in df.columns]
X = df[feat_cols].fillna(0).values
Xs = StandardScaler().fit_transform(X)
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    labels = km.labels_
    print(f"\nk={k}: cluster sizes {np.bincount(labels).tolist()}")
    # show mean of return per cluster
    for ci in range(k):
        sel = labels == ci
        ret = df.loc[sel, "return_test_det"].mean()
        kl = df.loc[sel, "wm/kl_mean"].mean()
        qstd = df.loc[sel, "bridge/q_pi_std"].mean()
        trust = df.loc[sel, "hgi/imag_trust"].mean()
        rounds = df.loc[sel, "_step"].tolist()
        contig = (np.diff(rounds) == 1).sum()
        print(f"  c{ci}: n={sel.sum():2d}  ret={ret:6.2f}  kl={kl:.4f}  q_pi_std={qstd:.4f}  imag_trust={trust:.2f}  contiguous_pairs={contig}")

# === J. Predictive features for return drops (regression-based) ===
print("\n=== J. WHAT PREDICTS A RETURN DROP NEXT ROUND? ===")
ret = df["return_test_det"].values
delta = np.diff(ret)  # return at t+1 minus return at t
# For each candidate metric at t, how well does it predict delta_{t+1}?
candidates = [
    "wm/kl_mean", "bridge/q_real_mean", "bridge/q_pi_std",
    "bridge/q_action_sensitivity", "wm/reward_target_std",
    "wm/reward_target_mean", "loss_critic", "loss_actor", "loss_q_churn",
    "hgi/imag_trust", "hgi/critic_health", "entropy_health/floor_target_steer",
    "entropy_health/det_gap_ema", "guard/dqda_norm", "guard/grad_health_min",
    "debug_log_std_mean", "return_test_det_stoch_gap",
    "debug_alpha_steer", "wm/z_t_dim_std_min",
]
results = []
for c in candidates:
    if c not in df.columns:
        continue
    x = df[c].values[:-1]  # at time t
    if np.std(x) < 1e-9:
        continue
    if len(x) != len(delta):
        continue
    r = np.corrcoef(x, delta)[0, 1]
    results.append((c, r))
results.sort(key=lambda kv: kv[1])
print("Predictors of NEXT-round return change (negative r = high-now → drop-next):")
for c, r in results:
    print(f"  {c:42s}  r={r:+.3f}")

# === K. Per-round wallclock — is training getting slower? ===
print("\n=== K. WALLCLOCK / COMPUTE DRIFT ===")
if "training_step_duration" in df.columns:
    d = df["training_step_duration"]
    print(f"training_step_duration: early={d.iloc[:5].mean():.4f}  late={d.iloc[-5:].mean():.4f}  ratio={d.iloc[-5:].mean()/d.iloc[:5].mean():.2f}x")
if "_runtime" in df.columns:
    rt = df["_runtime"].diff().dropna()
    print(f"per-round wallclock: early={rt.iloc[:5].mean():.1f}s  late={rt.iloc[-5:].mean():.1f}s  ratio={rt.iloc[-5:].mean()/rt.iloc[:5].mean():.2f}x")

# === L. Bonus: what fraction of total training time has imag been wasted? ===
print("\n=== L. WASTED IMAGINATION COMPUTE ===")
if "hgi/imag_trust" in df.columns and "_runtime" in df.columns:
    rt_diff = df["_runtime"].diff().fillna(0)
    on = df["hgi/imag_trust"] > 0.5
    on_time = rt_diff[on].sum()
    off_time = rt_diff[~on].sum()
    print(f"imagination ON wallclock: {on_time:.0f}s / {(on_time+off_time):.0f}s ({on_time/(on_time+off_time)*100:.1f}%)")

# === M. ACTOR ENTROPY: is it being held DOWN by the floor? ===
print("\n=== M. ACTOR ENTROPY VS TARGET ===")
for ax in ["steer", "gas", "brake"]:
    if f"debug_target_entropy_{ax}" in df.columns:
        te = df[f"debug_target_entropy_{ax}"]
        print(f"target_entropy_{ax}: early={te.iloc[:5].mean():.3f}  late={te.iloc[-5:].mean():.3f}  range=[{te.min():.3f},{te.max():.3f}]")
