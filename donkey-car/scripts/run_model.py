from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from rl import make_env, CONFIG  # or copy make_env into this file

# create env (same as training)
env = DummyVecEnv([make_env()])
env = VecTransposeImage(env)

# load model
model = PPO.load("donkey_ppo_model", env=env)

obs = env.reset()

print("Running model... Ctrl+C to stop")

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # reset if episode ends
    if done:
        obs = env.reset()