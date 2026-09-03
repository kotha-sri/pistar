import numpy as np

# Method 1: Import f1tenth_gym first to register the env, then use its internal gym
import f1tenth_gym
env = f1tenth_gym.gym.make("f1tenth-v0",
    map="Spielberg",
    num_agents=1,
    timestep=0.01,
    integrator="rk4"
)

obs, info = env.reset()
print(f"Observation type: {type(obs)}")
if isinstance(obs, dict):
    print(f"Keys: {list(obs.keys())}")
else:
    print(f"Shape: {obs.shape}")