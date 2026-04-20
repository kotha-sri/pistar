import gymnasium as gym
import gym_donkeycar

if "donkey-generated-track-v0" not in gym.envs.registry:
    gym.register(
        id="donkey-generated-track-v0",
        entry_point="gym_donkeycar.envs.donkey_env:DonkeyEnv",
        kwargs={"conf": {}} # You can pass default car settings here
    )