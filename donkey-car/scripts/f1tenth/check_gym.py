"""Run this to determine the correct import and env creation method."""
import importlib

for module_name in ["f1tenth_gym", "f110_gym", "f1tenth_gymnasium"]:
    try:
        mod = importlib.import_module(module_name)
        print(f"[OK] {module_name} found at: {mod.__file__}")
        print(f"     Contents: {[x for x in dir(mod) if not x.startswith('_')]}")
    except ImportError:
        print(f"[--] {module_name} not found")

import gymnasium as gym
print("\nRegistered f1tenth envs:")
for env_id in gym.envs.registry.keys():
    if "f1" in env_id.lower() or "f110" in env_id.lower():
        print(f"  {env_id}")