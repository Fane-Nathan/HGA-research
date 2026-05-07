"""Deeper dive: confirm posterior collapse, oscillation, gap pathology."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"c:/Users/felix/OneDrive/Documents/tmrl-test")
df = pd.read_csv(ROOT / "hgefc1_history.csv").sort_values("_step").reset_index(drop=True)

# === A. Posterior-collapse formal test ===
print("=== A. POSTERIOR COLLAPSE — formal evidence ===")
post_mu = df["wm/post_mu_mean"]
prior_mu = df["wm/prior_mu_mean"]
post_lv = df["wm/post_logvar_mean"]
prior_lv = df["wm/prior_logvar_mean"]

mu_gap = (post_mu - prior_mu).abs()
lv_gap = (post_lv - prior_lv).abs()
print(f"|post_mu - prior_mu|  early={mu_gap.iloc[:5].mean():.6f}  late={mu_gap.iloc[-5:].mean():.6f}")
print(f"|post_lv - prior_lv|  early={lv_gap.iloc[:5].mean():.6f}  late={lv_gap.iloc[-5:].mean():.6f}")

# Does decoder rely on posterior or prior?
ratio = df["wm_val/recon_prior_post_ratio"]
print(f"\nrecon_prior/recon_post ratio (1.0 = posterior is useless):")
print(f"  early={ratio.iloc[:5].mean():.4f}  late={ratio.iloc[-5:].mean():.4f}  min={ratio.min():.3f}  max={ratio.max():.3f}")

# === B. Q vs reward ratio over time — bootstrap inflation diagnostic ===
print("\n=== B. Q-INFLATION DIAGNOSTIC ===")
gamma = 0.99  # standard
r_mean = df["wm/reward_target_mean"]
q_mean = df["bridge/q_real_mean"]
q_steady = r_mean / (1 - gamma)  # what Q should be at convergence
print("                  early    mid     late")
print(f"r_mean         {r_mean.iloc[:5].mean():8.4f} {r_mean.iloc[28:32].mean():8.4f} {r_mean.iloc[-5:].mean():8.4f}")
print(f"q_real_mean    {q_mean.iloc[:5].mean():8.4f} {q_mean.iloc[28:32].mean():8.4f} {q_mean.iloc[-5:].mean():8.4f}")
print(f"q_steady (r/(1-γ)) {q_steady.iloc[:5].mean():8.4f} {q_steady.iloc[28:32].mean():8.4f} {q_steady.iloc[-5:].mean():8.4f}")
print(f"q / q_steady   {(q_mean/q_steady).iloc[:5].mean():8.4f} {(q_mean/q_steady).iloc[28:32].mean():8.4f} {(q_mean/q_steady).iloc[-5:].mean():8.4f}")

# Q-action sensitivity: is critic action-aware?
print("\nQ action sensitivity (lower = critic ignores actions):")
print(df["bridge/q_action_sensitivity"].describe())

# === C. Entropy-floor controller oscillation ===
print("\n=== C. ENTROPY-FLOOR CONTROLLER (closed-loop) ===")
floor = df["entropy_health/floor_target_steer"]
ret = df["return_test_det"]
# is floor monotonically rising?
diffs = floor.diff().dropna()
print(f"floor_target_steer: mean Δ per step = {diffs.mean():.4f}, std Δ = {diffs.std():.4f}")
print(f"# strict rises:  {(diffs > 0).sum()}  # strict drops: {(diffs < 0).sum()}  # flat: {(diffs == 0).sum()}")
print(f"range: {floor.min():.3f} → {floor.max():.3f}")
print("Periodogram (FFT power) of floor signal:")
sig = floor.values - floor.values.mean()
psd = np.abs(np.fft.rfft(sig))**2
freqs = np.fft.rfftfreq(len(sig), d=1)
top = np.argsort(psd)[::-1][:5]
for k in top:
    if freqs[k] > 0:
        print(f"  period={1/freqs[k]:.1f} rounds  power={psd[k]:.2f}")

# === D. Det/stoch gap — when does it explode? ===
print("\n=== D. DET/STOCH GAP CONDITIONAL ===")
gap = df["return_test_det_stoch_gap"]
det = df["return_test_det"]
stoch = df["return_test_stoch"]
print(f"gap correlates with: q_pi_std r={gap.corr(df['bridge/q_pi_std']):.3f}")
print(f"                    debug_log_std_mean r={gap.corr(df['debug_log_std_mean']):.3f}")
print(f"                    grad_norm_actor r={gap.corr(df['grad_norm_actor']):.3f}")
print(f"                    alpha_steer (debug) r={gap.corr(df['debug_alpha_steer']):.3f}")
print(f"                    return_test_det r={gap.corr(det):.3f}")
print(f"                    |gap| ↔ |return_test_det| r={gap.abs().corr(det.abs()):.3f}")
print(f"\nWhen gap >  +20 (stoch >> det): n={int((gap > 20).sum())}, mean det return = {det[gap>20].mean():.2f}")
print(f"When gap <  -20 (det >> stoch): n={int((gap < -20).sum())}, mean det return = {det[gap<-20].mean():.2f}")
print(f"When |gap|< 5  (agreement)  : n={int((gap.abs()<5).sum())}, mean det return = {det[gap.abs()<5].mean():.2f}")

# === E. HGI imagination trust regime detection ===
print("\n=== E. HGI BISTABILITY ===")
trust = df["hgi/imag_trust"]
# binarize: trust > 0.5 = ON
on = trust > 0.5
runs = (on.values[1:] != on.values[:-1]).sum()
print(f"imag_trust ON (>0.5): {on.sum()}/{len(on)} rounds")
print(f"State transitions: {runs}")
# average run length per state
states = on.astype(int).values
boundaries = np.where(np.diff(states) != 0)[0]
run_lengths = np.diff(np.concatenate([[0], boundaries+1, [len(states)]]))
state_seq = states[np.concatenate([[0], boundaries+1])]
on_runs = run_lengths[state_seq == 1]
off_runs = run_lengths[state_seq == 0]
print(f"ON  run lengths: mean={on_runs.mean() if len(on_runs) else 0:.1f}, n={len(on_runs)}")
print(f"OFF run lengths: mean={off_runs.mean() if len(off_runs) else 0:.1f}, n={len(off_runs)}")

# === F. Reward signal shrinkage ===
print("\n=== F. REWARD SIGNAL DEGRADATION ===")
rstd = df["wm/reward_target_std"]
rmean = df["wm/reward_target_mean"]
cv = rstd / rmean.replace(0, np.nan)  # coefficient of variation
print(f"reward CV (std/mean): early={cv.iloc[:5].mean():.3f}  late={cv.iloc[-5:].mean():.3f}")
print(f"  -> {'shrinking signal' if cv.iloc[-5:].mean() < cv.iloc[:5].mean() else 'stable/growing'}")

# === G. Frontier metric: q_pi_std (action sensitivity of critic) ===
print("\n=== G. CRITIC ACTION-SENSITIVITY DECAY ===")
qstd = df["bridge/q_pi_std"]
print(f"q_pi_std: early={qstd.iloc[:5].mean():.4f}  late={qstd.iloc[-5:].mean():.4f}")
print(f"  -> critic responsiveness shrunk by {(1 - qstd.iloc[-5:].mean()/qstd.iloc[:5].mean())*100:.1f}%")
