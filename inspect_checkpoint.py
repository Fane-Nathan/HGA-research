import cloudpickle

cp_path = r"C:\Users\felix\TmrlData\checkpoints\SPECTRE_v35_t.tcpt"
print(f"Loading {cp_path}...")
try:
    with open(cp_path, 'rb') as f:
        data = cloudpickle.load(f)
        
    print("Successfully loaded. Checking neural network weights...")
    if hasattr(data, 'agent') and hasattr(data.agent, 'model'):
        bias = data.agent.model.actor.mu_layer.bias
        print(f"\n[CHECKPOINT READ SUCCESSFUL]")
        print(f"Actor Hotwire Bias (mu_layer.bias) is currently: {bias}")
    else:
        print("\nCould not find the actor model inside the loaded checkpoint object.")
except Exception as e:
    print(f"\n[ERROR]: {e}")
