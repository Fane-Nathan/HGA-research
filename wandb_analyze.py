import csv, json

# Load the runs
runs = []
with open(r'c:\Users\felix\OneDrive\Documents\tmrl-test\wandb_all_runs.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        runs.append(row)

print(f"Total runs: {len(runs)}")
print()

# Classify into generations/phases
phases = {
    'Phase 0: Early Baselines (pre-SPECTRE)': [],
    'Phase 1: HP Sweeps & Stability': [],
    'Phase 2: DroQ Variants': [],
    'Phase 3: SPECTRE v0-v6 (Initial Architecture)': [],
    'Phase 4: SPECTRE v7-v16 (World Model Integration)': [],
    'Phase 5: SPECTRE v17-v25 (Tuning & Debugging)': [],
    'Phase 6: SPECTRE v26-v37 (PEARL+RSSM Full Pipeline)': [],
    'Phase 7: New Policy Series (Final Fixes)': [],
    'Other Projects': [],
}

for r in runs:
    name = r['name']
    proj = r['project']
    
    if proj != 'trackmania-rl':
        phases['Other Projects'].append(r)
    elif 'hpsweep' in name.lower() or 'trial' in name.lower():
        phases['Phase 1: HP Sweeps & Stability'].append(r)
    elif name.startswith('Test_') or name.startswith('One of'):
        phases['Phase 1: HP Sweeps & Stability'].append(r)
    elif name.startswith('DroQ'):
        phases['Phase 2: DroQ Variants'].append(r)
    elif name == 'SPECTRE' or any(name == f'SPECTRE_v{i}' for i in range(7)):
        phases['Phase 3: SPECTRE v0-v6 (Initial Architecture)'].append(r)
    elif any(name == f'SPECTRE_v{i}' for i in range(7, 17)) or name == 'SPECTRE_v10.5':
        phases['Phase 4: SPECTRE v7-v16 (World Model Integration)'].append(r)
    elif any(name == f'SPECTRE_v{i}' for i in range(17, 26)):
        phases['Phase 5: SPECTRE v17-v25 (Tuning & Debugging)'].append(r)
    elif any(name == f'SPECTRE_v{i}' for i in range(26, 38)):
        phases['Phase 6: SPECTRE v26-v37 (PEARL+RSSM Full Pipeline)'].append(r)
    elif 'new_policy' in name.lower():
        phases['Phase 7: New Policy Series (Final Fixes)'].append(r)
    else:
        phases['Phase 0: Early Baselines (pre-SPECTRE)'].append(r)

print("=" * 100)
print("GENERATIONAL ANALYSIS OF SPECTRE DEVELOPMENT")
print("=" * 100)

for phase_name, phase_runs in phases.items():
    if not phase_runs:
        continue
    
    print(f"\n{'='*80}")
    print(f"  {phase_name} ({len(phase_runs)} runs)")
    print(f"{'='*80}")
    
    # Date range
    dates = [r['created'][:10] for r in phase_runs if r['created']]
    if dates:
        print(f"  Period: {min(dates)} to {max(dates)}")
    
    # Duration stats
    durations = [float(r['duration_hrs']) for r in phase_runs if r['duration_hrs']]
    if durations:
        print(f"  Total compute: {sum(durations):.1f} hrs | Avg: {sum(durations)/len(durations):.1f} hrs | Max: {max(durations):.1f} hrs")
    
    # State breakdown
    states = {}
    for r in phase_runs:
        s = r['state']
        states[s] = states.get(s, 0) + 1
    print(f"  Outcomes: {dict(states)}")
    
    # Key metrics
    ent_coefs = []
    wm_kls = []
    for r in phase_runs:
        try:
            ec = float(r['entropy_coef'])
            ent_coefs.append((r['name'], ec))
        except:
            pass
        try:
            kl = float(r['wm_kl'])
            wm_kls.append((r['name'], kl))
        except:
            pass
    
    if ent_coefs:
        print(f"  Entropy coefs (final):")
        for name, ec in ent_coefs:
            flag = " ⚠️ CRASH" if ec < 0.005 else " ✅" if ec > 0.03 else " ⚡ LOW"
            print(f"    {name:<38} α={ec:.4f}{flag}")
    
    if wm_kls:
        print(f"  World model KL (final):")
        for name, kl in wm_kls:
            flag = " 💀 COLLAPSED" if kl < 0.1 else " ⚠️ LOW" if kl < 0.5 else " ✅"
            print(f"    {name:<38} KL={kl:.4f}{flag}")
    
    # Per-run table
    print(f"\n  {'Name':<38} {'State':<10} {'Hrs':>6} {'EntCoef':>10} {'WM_KL':>8}")
    print(f"  {'-'*75}")
    for r in phase_runs:
        ec = r['entropy_coef'] if r['entropy_coef'] else '-'
        kl = r['wm_kl'] if r['wm_kl'] else '-'
        try:
            ec = f"{float(ec):.4f}"
        except:
            pass
        try:
            kl = f"{float(kl):.4f}"
        except:
            pass
        print(f"  {r['name']:<38} {r['state']:<10} {float(r['duration_hrs']):>6.1f} {ec:>10} {kl:>8}")

print("\n\n" + "=" * 100)
print("PROBLEM PATTERN SUMMARY")
print("=" * 100)

# Entropy analysis
print("\n--- ENTROPY COEFFICIENT TRAJECTORY ---")
spectre_runs = [r for r in runs if r['name'].startswith('SPECTRE')]
for r in spectre_runs:
    try:
        ec = float(r['entropy_coef'])
        bar_len = int(max(0, ec) * 200)
        bar = '█' * min(bar_len, 50)
        flag = "💀" if ec < 0.005 else "⚡" if ec < 0.02 else "✅"
        print(f"  {r['name']:<38} {ec:>8.4f} {flag} {bar}")
    except:
        print(f"  {r['name']:<38}     -    ⬜")

# WM KL analysis
print("\n--- WORLD MODEL KL TRAJECTORY ---")
for r in spectre_runs:
    try:
        kl = float(r['wm_kl'])
        bar_len = int(kl * 50)
        bar = '█' * min(bar_len, 50)
        flag = "💀" if kl < 0.1 else "⚠️" if kl < 0.5 else "✅"
        print(f"  {r['name']:<38} {kl:>8.4f} {flag} {bar}")
    except:
        print(f"  {r['name']:<38}     -    ⬜")
