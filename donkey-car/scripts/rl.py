"""
Minimal working DonkeyCar RL training script (Gymnasium + SB3)

Works with:
- DonkeySim.exe running
- gym-donkeycar (installed)
- NumPy >= 2
- Stable-Baselines3 v2+
"""

import os
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gym_donkeycar.envs.donkey_env import DonkeyEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor


# =========================
# CONFIG
# =========================

CONFIG = {
    "host": "127.0.0.1",
    "port": 9091,
    "exe_path": "remote",
    "env_level": "circuit_launch",
    "throttle_fixed": 0.3,
}


# =========================
# WRAPPER (reward shaping)
# =========================

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, throttle_fixed=None):
        super().__init__(env)
        self.throttle_fixed = throttle_fixed

    def step(self, action):
        if self.throttle_fixed is not None:
            action = np.array([action[0], self.throttle_fixed], dtype=np.float32)

        result = self.env.step(action)

        # FORCE compatibility
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result

        done = terminated or truncated

        reward += info.get("speed", 0.0) * 0.5
        reward -= abs(float(action[0])) * 0.1

        if done:
            reward -= 10.0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        # legacy gym
        if isinstance(result, tuple) and len(result) == 2:
            return result

        return result, {}


# =========================
# ENV FACTORY
# =========================

def make_env():
    def _init():
        os.environ["DONKEY_SIM_PATH"] = CONFIG["exe_path"]
        os.environ["DONKEY_SIM_PORT"] = str(CONFIG["port"])
        os.environ["DONKEY_SIM_HOST"] = CONFIG["host"]

        env = DonkeyEnv(level=CONFIG["env_level"])
        env = RewardWrapper(env, CONFIG["throttle_fixed"])
        env = RecordEpisodeStatistics(env)
        env = Monitor(env)

        return env

    return _init


# =========================
# TRAIN
# =========================

def train():
    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./tb_logs",
    )

    print("\n🚗 Starting training... (make sure DonkeySim.exe is running)\n")

    model.learn(total_timesteps=200_000)

    model.save("donkey-car/scripts/models/donkey_ppo_model")
    print("\n✅ Model saved as donkey_ppo_model.zip")


# =========================
# INFERENCE
# =========================

def run():
    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    model = PPO.load(f"donkey-car/scripts/donkey_ppo_model", env=env)

    obs = env.reset()

    print("\n▶ Running inference...\n")

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    mode = input("train or run? ").strip().lower()

    if mode == "train":
        train()
    else:
        run()