"""
DonkeyCar SAC Training Script (Gymnasium + NumPy >= 2 compatible)

Run:
  1. Start simulator:
     .\donkey_sim.exe

  2. Train:
     python donkey_sac_train.py --mode train

  3. Run model:
     python donkey_sac_train.py --mode run --model donkey_sac_model
"""

import os
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gym_donkeycar.envs.donkey_env import DonkeyEnv

from stable_baselines3 import SAC
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

    "image_width": 64,
    "image_height": 64,
    "image_depth": 3,

    "throttle_fixed": None,  # SAC learns throttle automatically

    "total_timesteps": 300_000,
}


# =========================
# WRAPPER
# =========================

class ResidualPolicyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # tuned for circuit_launch
        self.k_cte = 0.6
        self.k_damp = 0.15

    def step(self, action):
        action = np.clip(action, -1, 1)

        info = getattr(self, "last_info", {"cte": 0.0})
        cte = info.get("cte", 0.0)

        # =========================
        # LANE CENTERING CONTROLLER
        # =========================

        # normalize cte (track width ~4 → scale to [-1,1])
        norm_cte = np.clip(cte / 2.0, -1.0, 1.0)

        # push toward center between lane lines
        steer_correction = -self.k_cte * norm_cte

        # damping to reduce oscillation
        steer_correction -= self.k_damp * action[0]

        # =========================
        # RESIDUAL POLICY
        # =========================

        steer = action[0] + steer_correction
        steer = float(np.clip(steer, -1.0, 1.0))

        throttle = (action[1] + 1) / 2
        throttle = min(throttle, 0.2)   # cap speed

        final_action = np.array([steer, throttle], dtype=np.float32)

        result = self.env.step(final_action)

        # --- Gym compatibility ---
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result

        self.last_info = info

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        self.last_info = info
        return obs, info


# =========================
# ENV FACTORY
# =========================

def make_env():
    def _init():
        os.environ["DONKEY_SIM_PATH"] = CONFIG["exe_path"]
        os.environ["DONKEY_SIM_PORT"] = str(CONFIG["port"])
        os.environ["DONKEY_SIM_HOST"] = CONFIG["host"]

        env = DonkeyEnv(level=CONFIG["env_level"])
        env = ResidualPolicyWrapper(env)
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

    model = SAC(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./tb_logs",
    )

    print("\n🚗 Starting SAC training...\n")

    model.learn(total_timesteps=CONFIG["total_timesteps"])

    model.save("donkey_sac_model")

    print("\n✅ Model saved: donkey_sac_model.zip")


# =========================
# RUN MODEL
# =========================

def run(model_path):
    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    model = SAC.load(model_path, env=env)

    obs = env.reset()

    print("\n▶ Running trained SAC model...\n")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "run"], default="train")
    parser.add_argument("--model", type=str, default="donkey_sac_model")

    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        run(args.model)