"""Verify a trained residual loads and drives. Usage: python logs/check_inference.py [port]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from path_follower import PathFollower
from residual_policy import ResidualPolicy
from residual_env import ResidualDonkeyEnv

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9091

model = SAC.load("models/residual_sac")
base = PathFollower()
rp = ResidualPolicy(base)
rp.attach_sb3(model)                      # trained SAC becomes the residual actor
print("loaded model; residual actor attached")

# drive a few steps through the env using the trained residual
env = ResidualDonkeyEnv(base, port=port, max_steps=200)
obs, _ = env.reset()
for i in range(10):
    residual = model.predict(obs, deterministic=True)[0]      # what the agent commands
    combined = rp.act(obs, obs[0], obs[2])                     # base(cte,speed)+residual
    obs, r, term, trunc, info = env.step(residual)
    print("step %d  residual=%s applied=%s cte=%.3f r=%.3f"
          % (i, np.round(residual, 3), np.round(info["applied"], 3), info["cte"], r))
    if term:
        print("  terminated"); break
env.close()
print("INFERENCE OK")
