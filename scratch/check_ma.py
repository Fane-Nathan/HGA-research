import pandas as pd
csv_path = 'c:/Users/felix/TmrlData/ablation/SPECTRE_HGI_HGEFC_7.stable.csv'
df = pd.read_csv(csv_path)

# Calculate epoch means
epoch_stats = df.groupby('epoch').agg({
    'return_test': 'mean',
    'return_test_det': 'mean',
    'return_train': 'mean'
}).reset_index()

epoch_stats['ma10_test'] = epoch_stats['return_test'].rolling(window=10, min_periods=1).mean()
epoch_stats['ma10_test_det'] = epoch_stats['return_test_det'].rolling(window=10, min_periods=1).mean()

# How is eval_epoch_return calculated? Dual mode uses max of stoch and det or average?
# Let's see max over the epoch or mean?
print(epoch_stats)
