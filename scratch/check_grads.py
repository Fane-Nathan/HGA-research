import json

print("Checking Ep 16 & 17 Gradients")
with open('c:/Users/felix/TmrlData/ablation/SPECTRE_HGI_HGEFC_7.metrics.jsonl', 'r') as f:
    for line in f:
        d = json.loads(line)
        ep = d.get('epoch')
        rnd = d.get('round')
        if ep == 16 and rnd >= 7:
            print(f"Ep 16 Rnd {rnd}: ActorGrad={d.get('grad_norm_actor', 0):.2f}, CriticGrad={d.get('grad_norm_critic', 0):.2f}, LRScale={d.get('guard/lr_scale', 1.0):.2f}")
        if ep == 17 and rnd <= 2:
            print(f"Ep 17 Rnd {rnd}: ActorGrad={d.get('grad_norm_actor', 0):.2f}, CriticGrad={d.get('grad_norm_critic', 0):.2f}, LRScale={d.get('guard/lr_scale', 1.0):.2f}")
