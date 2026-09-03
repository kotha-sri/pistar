"""
train_residual.py
-----------------
Train the residual policy (SAC) on top of the PathFollower base controller.

    python train_residual.py [total_timesteps] [port]

The agent learns ONLY the bounded residual; the PathFollower keeps the car on
the track the whole time (see residual_policy.py / residual_env.py). Observations
and rewards are normalized with running statistics, as in the paper.

Outputs:
    models/residual_sac.zip   — the trained SAC residual actor
    models/residual_vecnorm.pkl — the obs/reward normalizer (needed at inference)

Drive with the trained residual afterwards:
    base = PathFollower()
    model = SAC.load("models/residual_sac")
    rp = ResidualPolicy(base); rp.attach_sb3(model)
    # then rp.act(obs, cte, speed) -> combined [steering, throttle]
"""

import os
import sys

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from path_follower import PathFollower
from residual_env import ResidualDonkeyEnv


class RacerMetrics(BaseCallback):
    """Print + log per-episode mean speed and steering/yaw-jerk (RMS).

    Watch `speed` climb and `yaw_jerk`/`steer_jerk` fall = the policy is
    learning to go faster AND smoother (the "racer" goal).
    """

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "ep_speed" in info:
                self.logger.record("racer/mean_speed", info["ep_speed"])
                self.logger.record("racer/steer_jerk_rms", info["ep_steer_jerk"])
                self.logger.record("racer/yaw_jerk_rms", info["ep_yaw_jerk"])
                print(
                    f"[ep] len={info['ep_len']:4d}  speed={info['ep_speed']:.2f}  "
                    f"steer_jerk={info['ep_steer_jerk']:.4f}  "
                    f"yaw_jerk={info['ep_yaw_jerk']:.4f}"
                )
        return True

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
PORT            = int(sys.argv[2]) if len(sys.argv) > 2 else 9091

os.makedirs("models", exist_ok=True)


def make_env():
    # Use the robust, slow base as-tuned (target_speed 0.8). The residual is
    # responsible for adding speed (throttle) on top of this stable foundation.
    env = ResidualDonkeyEnv(PathFollower(), port=PORT, max_steps=1000)
    return Monitor(env)


def main():
    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = SAC(
        "MlpPolicy",
        venv,
        verbose=1,
        learning_starts=500,        # collect some experience before learning
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cpu",               # tiny MLP — CPU is fine, sim is the bottleneck
    )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False,
                    callback=RacerMetrics())
    except KeyboardInterrupt:
        print("\n[Train] Interrupted — saving partial model.")
    finally:
        model.save("models/residual_sac")
        venv.save("models/residual_vecnorm.pkl")
        venv.close()
        print("[Train] Saved models/residual_sac.zip and models/residual_vecnorm.pkl")


if __name__ == "__main__":
    main()
