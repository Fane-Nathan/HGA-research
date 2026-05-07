import pandas as pd
import json
import matplotlib.pyplot as plt
import os

csv_path = 'c:/Users/felix/TmrlData/ablation/SPECTRE_HGI_HGEFC_7.stable.csv'
jsonl_path = 'c:/Users/felix/TmrlData/ablation/SPECTRE_HGI_HGEFC_7.metrics.jsonl'
stall_json_path = 'c:/Users/felix/TmrlData/ablation/SPECTRE_HGI_HGEFC_7_stall.json'

os.makedirs('scratch', exist_ok=True)

print("Starting Advanced Analysis...")

df = pd.read_csv(csv_path)

# 1. Analyze Epoch 16, Round 8 Anomaly
anomaly_df = df[(df['epoch'] == 16) & (df['round'] == 8)]
print("Anomaly Row:")
print(anomaly_df[['epoch', 'round', 'return_test_det', 'return_train', 'best_checkpoint/triggered']])

# 2. Plot Training vs Testing returns
plt.figure(figsize=(12, 6))
plt.plot(df['return_train'], label='Return Train', alpha=0.7)
plt.plot(df['return_test'], label='Return Test (Stoch)', alpha=0.7)
plt.plot(df['return_test_det'], label='Return Test (Det)', alpha=0.9, linewidth=2)
plt.title('Agent Returns Over Time (SPECTRE_HGI_HGEFC_7)')
plt.xlabel('Global Step (Rounds)')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.savefig('scratch/returns_plot.png')
plt.close()

# 3. Read JSONL for deeper metrics
metrics_data = []
with open(jsonl_path, 'r') as f:
    for line in f:
        metrics_data.append(json.loads(line))

df_metrics = pd.DataFrame(metrics_data)

# 4. Plot Guard Metrics and Health
plt.figure(figsize=(12, 6))
if 'entropy_health/critic_health' in df_metrics.columns:
    plt.plot(df_metrics['entropy_health/critic_health'], label='Critic Health', alpha=0.8)
if 'entropy_health/model_trust' in df_metrics.columns:
    plt.plot(df_metrics['entropy_health/model_trust'], label='Model Trust', alpha=0.8)
if 'guard/actor_stability_active' in df_metrics.columns:
    plt.plot(df_metrics['guard/actor_stability_active'], label='Actor Stability Active', alpha=0.8)
if 'guard/lr_scale' in df_metrics.columns:
    plt.plot(df_metrics['guard/lr_scale'], label='LR Scale', alpha=0.8)

plt.title('Hybrid Guard Health Metrics')
plt.xlabel('Global Step (Rounds)')
plt.ylabel('Health / Value')
plt.legend()
plt.grid(True)
plt.savefig('scratch/guard_health_plot.png')
plt.close()

# 5. Extract specific stall info
with open(stall_json_path, 'r') as f:
    stall_data = json.load(f)

with open('scratch/analysis_report.txt', 'w') as f:
    f.write("STALL ANALYSIS REPORT\n=====================\n")
    f.write(f"Stall Epoch: {stall_data.get('epoch')}\n")
    f.write(f"Best Eval MA10: {stall_data.get('best_eval_ma10')}\n")
    f.write(f"Current Eval MA10: {stall_data.get('eval_return_ma10')}\n")
    f.write(f"Stall Epochs Count: {stall_data.get('stall_epochs')}\n")
    
    # Check what happened around Epoch 16
    f.write("\nDetails around Epoch 16:\n")
    ep16_data = df[df['epoch'] == 16]
    for _, row in ep16_data.iterrows():
        f.write(f"  Round {int(row['round'])}: Train={row['return_train']:.2f}, Det={row['return_test_det']:.2f}, Stoch={row['return_test']:.2f}, CriticLoss={row['loss_critic']:.4f}\n")

print("Analysis Complete. Output saved to scratch/")
