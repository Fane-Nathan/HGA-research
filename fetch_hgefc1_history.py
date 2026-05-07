"""Fetch the HGEFC_1 run history via the wandb Python client (clean per-metric data)."""
import os, sys
import pandas as pd

os.environ["WANDB_API_KEY"] = "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"

import wandb
api = wandb.Api()

ENTITY = "trackmania-rl"
PROJECT = "trackmania-rl"
RUN_NAME = "SPECTRE_HGI_HGEFC_1"

# Use display name to find the run
runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"displayName": RUN_NAME}))
if not runs:
    print(f"no run named {RUN_NAME!r} in {ENTITY}/{PROJECT}; listing recent runs:")
    for r in list(api.runs(f"{ENTITY}/{PROJECT}"))[:20]:
        print(f"  {r.name}  {r.state}  {r.created_at}")
    sys.exit(1)

run = runs[0]
print(f"found {run.name}  state={run.state}  created={run.created_at}  steps={run.lastHistoryStep+1 if run.lastHistoryStep is not None else '?'}")

# Pull full history (no key filter — scan_history with keys requires ALL keys present per row).
rows = list(run.scan_history())
print(f"pulled {len(rows)} history records")
df = pd.DataFrame(rows)
print(f"columns: {len(df.columns)}; sample: {list(df.columns)[:25]}")
print(f"sanity: wall_clock_seconds present? {'wall_clock_seconds' in df.columns}")

out = "hgefc1_history.csv"
df.to_csv(out, index=False)
print(f"saved {out}")
