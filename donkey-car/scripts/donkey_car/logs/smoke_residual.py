"""Smoke test for ResidualDonkeyEnv against a live sim. Usage: python logs/smoke_residual.py [port]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_follower import PathFollower
from residual_env import ResidualDonkeyEnv

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9091

env = ResidualDonkeyEnv(PathFollower(), port=port, max_steps=200)
obs, _ = env.reset()
print("obs shape", obs.shape, "expected", env.observation_space.shape)

# zero residual -> should track like the tuned PD (low |cte|)
ctes = []
for i in range(25):
    obs, r, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
    ctes.append(abs(info["cte"]))
    if term:
        print("  terminated at zero-residual step", i); break
print("zero-residual: mean|cte|=%.3f  last_reward=%.3f  speed=%.2f"
      % (np.mean(ctes), r, info["speed"]))

# random residual -> exercise the action path
obs, _ = env.reset()
for i in range(8):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    print("  rand step %d  a=%s applied=%s cte=%.2f r=%.3f term=%s"
          % (i, np.round(a, 2), np.round(info["applied"], 3), info["cte"], r, term))
    if term:
        break
env.close()
print("SMOKE OK")
