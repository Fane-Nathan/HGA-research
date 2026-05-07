import csv

f = r'c:\Users\felix\TmrlData\ablation\SPECTRE_new_policy_test_4.csv'
with open(f, 'r') as fp:
    reader = csv.reader(fp)
    headers = list(next(reader))
    rows = [r for r in reader]

def col(name):
    """Get column values as floats"""
    idx = headers.index(name)
    vals = []
    for r in rows:
        try:
            vals.append(float(r[idx]))
        except:
            vals.append(None)
    return vals

def safe(v, fmt=".4f"):
    if v is None: return "-"
    return f"{v:{fmt}}"

print(f"SPECTRE_new_policy_test_4 Analysis")
print(f"Rows: {len(rows)} | Columns: {len(headers)}")
print(f"{'='*130}")

# Key metrics over time
key_cols = [
    ('epoch', '.0f'),
    ('round', '.0f'),
    ('loss_actor', '.4f'),
    ('loss_critic', '.4f'),
    ('bridge/q_real_mean', '.4f'),
    ('bridge/q_real_std', '.4f'),
    ('entropy_coef', '.4f'),
    ('debug_alpha_steer', '.4f'),
    ('debug_alpha_gas', '.4f'),
    ('debug_alpha_brake', '.4f'),
    ('debug_log_std_mean', '.4f'),
    ('grad_health_steer', '.4f'),
    ('grad_health_gas', '.4f'),
    ('grad_health_brake', '.4f'),
    ('guard/grad_health_mean', '.4f'),
    ('guard/grad_health_min', '.4f'),
    ('bridge/dqda_norm', '.4f'),
    ('bridge/dqda_abs_mean', '.4f'),
    ('wm_kl', '.4f'),
    ('wm/kl_mean', '.4f'),
    ('wm_kl_anti_collapse', '.6f'),
    ('wm_input/action_std', '.4f'),
    ('wm_input/action_mean', '.4f'),
    ('wm_input/context_z_dead_dim_ratio_1e3', '.4f'),
    ('wm_input/state_std', '.4f'),
    ('return_test', '.2f'),
    ('return_test_det', '.2f'),
    ('return_test_stoch', '.2f'),
    ('return_train', '.2f'),
]

# Print header
hdr = f"{'Rnd':>4}"
for name, _ in key_cols:
    short = name.split('/')[-1][:12]
    hdr += f" {short:>12}"
print(hdr)
print('-' * len(hdr))

# Print every row
for i, r in enumerate(rows):
    line = f"{i:>4}"
    for name, fmt in key_cols:
        idx = headers.index(name)
        try:
            v = float(r[idx])
            line += f" {v:>12{fmt}}"
        except:
            line += f" {'-':>12}"
    print(line)

print(f"\n{'='*130}")
print("SECTION-BY-SECTION ANALYSIS")
print(f"{'='*130}")

# 1. Entropy/Alpha analysis
print("\n--- 1. ENTROPY & ALPHA STABILITY ---")
ec = col('entropy_coef')
a_s = col('debug_alpha_steer')
a_g = col('debug_alpha_gas')
a_b = col('debug_alpha_brake')
af_s = col('debug_alpha_floor_steer')
af_g = col('debug_alpha_floor_gas')

valid_ec = [v for v in ec if v is not None]
print(f"  entropy_coef:  min={min(valid_ec):.4f}  max={max(valid_ec):.4f}  last={valid_ec[-1]:.4f}  range_pct={(max(valid_ec)-min(valid_ec))/max(valid_ec)*100:.1f}%")
valid_as = [v for v in a_s if v is not None]
valid_ag = [v for v in a_g if v is not None]
valid_ab = [v for v in a_b if v is not None]
print(f"  alpha_steer:   min={min(valid_as):.4f}  max={max(valid_as):.4f}  last={valid_as[-1]:.4f}")
print(f"  alpha_gas:     min={min(valid_ag):.4f}  max={max(valid_ag):.4f}  last={valid_ag[-1]:.4f}")
print(f"  alpha_brake:   min={min(valid_ab):.4f}  max={max(valid_ab):.4f}  last={valid_ab[-1]:.4f}")
print(f"  VERDICT: {'✅ STABLE' if min(valid_ec) > 0.03 else '⚠️ UNSTABLE'} (Gen 1-5 fix working)")

# 2. Q-value analysis
print("\n--- 2. Q-VALUE DYNAMICS ---")
q_mean = col('bridge/q_real_mean')
q_std = col('bridge/q_real_std')
valid_q = [v for v in q_mean if v is not None]
print(f"  Q mean:  start={valid_q[0]:.4f}  end={valid_q[-1]:.4f}  max={max(valid_q):.4f}  min={min(valid_q):.4f}")
print(f"  Q trajectory (every 5 rounds):")
for i in range(0, len(valid_q), 5):
    bar_len = int(max(0, valid_q[i]+1) * 20)
    bar = '█' * min(bar_len, 50)
    print(f"    Round {i:>3}: Q={valid_q[i]:>8.4f} {bar}")
# Check for unbounded ascent
q_diffs = [valid_q[i+1] - valid_q[i] for i in range(len(valid_q)-1)]
max_jump = max(q_diffs)
print(f"  Max single-round Q jump: {max_jump:.4f}")
print(f"  Total Q ascent: {valid_q[-1] - valid_q[0]:.4f}")
ascending_pct = sum(1 for d in q_diffs if d > 0) / len(q_diffs) * 100
print(f"  Ascending rounds: {ascending_pct:.1f}%")
print(f"  VERDICT: {'🔥 UNBOUNDED ASCENT' if valid_q[-1] > 0.5 and ascending_pct > 60 else '⚠️ INFLATING' if valid_q[-1] > 0.3 else '✅ STABLE'}")

# 3. Actor loss analysis
print("\n--- 3. ACTOR LOSS & GRADIENT HEALTH ---")
la = col('loss_actor')
valid_la = [v for v in la if v is not None]
print(f"  Actor loss: start={valid_la[0]:.4f}  end={valid_la[-1]:.4f}  min={min(valid_la):.4f}")
gh_mean = col('guard/grad_health_mean')
gh_min = col('guard/grad_health_min')
gh_s = col('grad_health_steer')
gh_g = col('grad_health_gas')
gh_b = col('grad_health_brake')
valid_ghm = [v for v in gh_mean if v is not None]
valid_ghs = [v for v in gh_s if v is not None]
valid_ghg = [v for v in gh_g if v is not None]
valid_ghb = [v for v in gh_b if v is not None]
print(f"  Grad health mean: start={valid_ghm[0]:.4f}  end={valid_ghm[-1]:.4f}  min={min(valid_ghm):.4f}")
print(f"  Grad health per-dim:")
print(f"    Steer: start={valid_ghs[0]:.4f}  end={valid_ghs[-1]:.4f}  min={min(valid_ghs):.4f}")
print(f"    Gas:   start={valid_ghg[0]:.4f}  end={valid_ghg[-1]:.4f}  min={min(valid_ghg):.4f}")
print(f"    Brake: start={valid_ghb[0]:.4f}  end={valid_ghb[-1]:.4f}  min={min(valid_ghb):.4f}")

# Gradient starvation trajectory
print(f"  Grad health trajectory (every 5 rounds):")
for i in range(0, len(valid_ghm), 5):
    bar_len = int(valid_ghm[i] * 200)
    bar = '█' * min(bar_len, 50)
    flag = "💀" if valid_ghm[i] < 0.01 else "⚡" if valid_ghm[i] < 0.05 else "✅"
    print(f"    Round {i:>3}: GH={valid_ghm[i]:>8.4f} {flag} {bar}")

print(f"  VERDICT: {'💀 GRADIENT STARVATION' if min(valid_ghm) < 0.01 else '⚡ DECLINING' if valid_ghm[-1] < valid_ghm[0]*0.5 else '✅ HEALTHY'}")

# 4. dQ/da analysis
print("\n--- 4. dQ/da (CRITIC-TO-ACTOR GRADIENT) ---")
dqda_norm = col('bridge/dqda_norm')
dqda_abs = col('bridge/dqda_abs_mean')
valid_dqn = [v for v in dqda_norm if v is not None]
valid_dqa = [v for v in dqda_abs if v is not None]
if valid_dqn:
    print(f"  dQ/da norm: start={valid_dqn[0]:.4f}  end={valid_dqn[-1]:.4f}  max={max(valid_dqn):.4f}")
if valid_dqa:
    print(f"  dQ/da abs:  start={valid_dqa[0]:.4f}  end={valid_dqa[-1]:.4f}  max={max(valid_dqa):.4f}")

# 5. Action distribution
print("\n--- 5. ACTION DISTRIBUTION ---")
a_std = col('wm_input/action_std')
a_mean = col('wm_input/action_mean')
log_std = col('debug_log_std_mean')
log_std_min = col('debug_log_std_min')
valid_astd = [v for v in a_std if v is not None]
valid_amean = [v for v in a_mean if v is not None]
valid_lstd = [v for v in log_std if v is not None]
valid_lstdm = [v for v in log_std_min if v is not None]
print(f"  action_std:     start={valid_astd[0]:.4f}  end={valid_astd[-1]:.4f}  min={min(valid_astd):.4f}")
print(f"  action_mean:    start={valid_amean[0]:.4f}  end={valid_amean[-1]:.4f}")
print(f"  log_std_mean:   start={valid_lstd[0]:.4f}  end={valid_lstd[-1]:.4f}  min={min(valid_lstd):.4f}")
print(f"  log_std_min:    start={valid_lstdm[0]:.4f}  end={valid_lstdm[-1]:.4f}  min={min(valid_lstdm):.4f}")
print(f"  VERDICT: {'✅ NOT COLLAPSED' if valid_astd[-1] > 0.3 else '⚠️ NARROWING'}")

# 6. World Model health
print("\n--- 6. WORLD MODEL HEALTH ---")
wm_kl = col('wm/kl_mean')
wm_kl_ac = col('wm_kl_anti_collapse')
wm_dead = col('wm_input/context_z_dead_dim_ratio_1e3')
wm_zt_dead = col('wm/z_t_dead_dim_ratio_1e3')
valid_kl = [v for v in wm_kl if v is not None]
valid_ac = [v for v in wm_kl_ac if v is not None]
valid_dead = [v for v in wm_dead if v is not None]
valid_ztd = [v for v in wm_zt_dead if v is not None]
print(f"  KL mean:         start={valid_kl[0]:.4f}  end={valid_kl[-1]:.4f}  min={min(valid_kl):.4f}")
print(f"  Anti-collapse:   start={valid_ac[0]:.6f}  end={valid_ac[-1]:.6f}")
print(f"  Context z dead:  all={set(valid_dead)}")
print(f"  WM z_t dead:     start={valid_ztd[0]:.4f}  end={valid_ztd[-1]:.4f}")
print(f"  VERDICT: {'✅ ALIVE' if max(valid_dead) == 0 else '💀 DEAD DIMS'}")

# 7. Returns
print("\n--- 7. PERFORMANCE (RETURNS) ---")
ret_test = col('return_test')
ret_det = col('return_test_det')
ret_stoch = col('return_test_stoch')
ret_train = col('return_train')
valid_rt = [v for v in ret_test if v is not None]
valid_rd = [v for v in ret_det if v is not None]
valid_rs = [v for v in ret_stoch if v is not None]
valid_rtr = [v for v in ret_train if v is not None]
if valid_rt:
    print(f"  test_return:  start={valid_rt[0]:.2f}  end={valid_rt[-1]:.2f}  max={max(valid_rt):.2f}")
if valid_rd:
    print(f"  test_det:     start={valid_rd[0]:.2f}  end={valid_rd[-1]:.2f}  max={max(valid_rd):.2f}")
if valid_rs:
    print(f"  test_stoch:   start={valid_rs[0]:.2f}  end={valid_rs[-1]:.2f}  max={max(valid_rs):.2f}")
if valid_rtr:
    print(f"  train_return: start={valid_rtr[0]:.2f}  end={valid_rtr[-1]:.2f}  max={max(valid_rtr):.2f}")
if valid_rd and valid_rs:
    gap = [s-d for s,d in zip(valid_rs, valid_rd) if s is not None and d is not None]
    print(f"  Stoch-Det gap: mean={sum(gap)/len(gap):.2f}  max={max(gap):.2f}")

# 8. Guard system
print("\n--- 8. ACTOR GUARD SYSTEM ---")
guard_active = col('guard/actor_stability_active')
guard_blocked = col('guard/actor_mu_step_blocked')
guard_std_allowed = col('guard/actor_std_step_allowed')
valid_ga = [v for v in guard_active if v is not None]
valid_gb = [v for v in guard_blocked if v is not None]
valid_gsa = [v for v in guard_std_allowed if v is not None]
if valid_ga:
    print(f"  Stability active: start={valid_ga[0]:.0f}  end={valid_ga[-1]:.0f}")
if valid_gb:
    print(f"  Mu steps blocked: start={valid_gb[0]:.0f}  end={valid_gb[-1]:.0f}  sum={sum(valid_gb):.0f}")
if valid_gsa:
    print(f"  Std steps allowed: start={valid_gsa[0]:.0f}  end={valid_gsa[-1]:.0f}")

# 9. Critic gradient norm
print("\n--- 9. CRITIC DIAGNOSTICS ---")
lc = col('loss_critic')
gn_c = col('grad_norm_critic')
gn_a = col('grad_norm_actor')
valid_lc = [v for v in lc if v is not None]
valid_gnc = [v for v in gn_c if v is not None]
valid_gna = [v for v in gn_a if v is not None]
print(f"  Critic loss:     start={valid_lc[0]:.4f}  end={valid_lc[-1]:.4f}  max={max(valid_lc):.4f}")
print(f"  Critic grad norm: start={valid_gnc[0]:.4f}  end={valid_gnc[-1]:.4f}  max={max(valid_gnc):.4f}")
if valid_gna:
    print(f"  Actor grad norm:  start={valid_gna[0]:.4f}  end={valid_gna[-1]:.4f}  max={max(valid_gna):.4f}")

# 10. Correlation: Q vs grad_health
print("\n--- 10. Q-VALUE vs GRADIENT HEALTH CORRELATION ---")
print(f"  {'Round':>5} {'Q_mean':>10} {'GH_mean':>10} {'Actor_Loss':>12} {'dQda_norm':>10} {'log_std':>10}")
for i in range(0, len(rows), 5):
    q = valid_q[i] if i < len(valid_q) else None
    gh = valid_ghm[i] if i < len(valid_ghm) else None
    al = valid_la[i] if i < len(valid_la) else None
    dq = valid_dqn[i] if i < len(valid_dqn) else None
    ls = valid_lstd[i] if i < len(valid_lstd) else None
    print(f"  {i:>5} {safe(q):>10} {safe(gh):>10} {safe(al):>12} {safe(dq):>10} {safe(ls):>10}")
