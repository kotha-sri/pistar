"""
drive_residual.py
-----------------
Watch the trained residual policy race in the sim.

    python drive_residual.py [port]

Loads the trained SAC residual and the VecNormalize statistics (so the
observation is normalized exactly as during training), attaches it on top of
the PathFollower base, and drives until Ctrl-C. Prints speed / cte / the
residual it is applying, plus per-lap-ish episode metrics.
"""

import os
import sys

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from path_follower import PathFollower
from residual_env import ResidualDonkeyEnv

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9091

MODEL_PATH   = "models/residual_sac"
VECNORM_PATH = "models/residual_vecnorm.pkl"

if not (os.path.exists(MODEL_PATH + ".zip") and os.path.exists(VECNORM_PATH)):
    sys.exit("No trained model found. Run:  python train_residual.py 30000 9091")


def make_env():
    # same robust slow base used during training (target_speed 0.8 from CONFIG)
    # one long episode so it keeps driving; off-track still resets it
    return ResidualDonkeyEnv(PathFollower(), port=PORT, max_steps=10_000_000)


def main():
    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(VECNORM_PATH, venv)
    venv.training = False               # freeze the running stats
    venv.norm_reward = False

    model = SAC.load(MODEL_PATH, device="cpu")
    print("[Drive] Trained residual loaded. Racing — Ctrl-C to stop.\n")

    obs = venv.reset()
    step = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            info = infos[0]
            step += 1
            if step % 10 == 0:
                print(
                    f"speed={info['speed']:5.2f}  cte={info['cte']:6.2f}  "
                    f"residual_applied={np.round(info['applied'], 3)}"
                )
            if dones[0] and "ep_speed" in info:
                print(
                    f"  [episode end] len={info['ep_len']}  mean_speed={info['ep_speed']:.2f}  "
                    f"yaw_jerk={info['ep_yaw_jerk']:.4f}"
                )
    except KeyboardInterrupt:
        print("\n[Drive] Stopping.")
    finally:
        venv.close()


if __name__ == "__main__":
    main()
