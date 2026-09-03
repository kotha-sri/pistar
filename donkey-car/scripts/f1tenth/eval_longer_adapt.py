"""Test longer adaptation with higher action scaling.

The 10K results showed vel_s=2.0 works on Austin but DNFs on Monza/Silverstone.
Hypothesis: the meta-init needs more steps to recalibrate to the new scaling.
Test 20K and 40K adaptation steps.
"""

import os
import sys
import gc
import pickle
import time

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
import jax
import jax.numpy as jnp

from stable_baselines3.common.vec_env import DummyVecEnv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from residual_env import RLPPEnv
from train_reptile import (
    create_model, snapshot_params, load_params, make_env_fn,
)

TRACKS = ["Austin", "Monza", "Silverstone"]
N_LAPS = 10
ALPHA_RL = 0.85

CONFIGS = [
    # (label, action_scaling, vel_gain, adapt_steps)
    ("baseline_10K", (0.05, 1.0), 0.50, 10_000),
    ("vel2x_10K",    (0.05, 2.0), 0.50, 10_000),
    ("vel2x_20K",    (0.05, 2.0), 0.50, 20_000),
    ("vel2x_40K",    (0.05, 2.0), 0.50, 40_000),
    ("steer2x_20K",  (0.10, 1.0), 0.50, 20_000),
    ("steer2x_40K",  (0.10, 1.0), 0.50, 40_000),
    ("both2x_20K",   (0.10, 2.0), 0.50, 20_000),
    ("both2x_40K",   (0.10, 2.0), 0.50, 40_000),
]


class Args:
    seed = 42
    utd = 20
    inner_lr = 1e-3
    inner_steps = 50_000
    batch_size = 256


def evaluate(model, track, tracks_dir, alpha_rl, vel_gain, action_scaling):
    env = RLPPEnv(
        track_name=track, tracks_dir=tracks_dir,
        alpha_rl=alpha_rl, velocity_gain=vel_gain,
        action_scaling=action_scaling,
        mu_noise_std=0.0, velocity_curriculum=False, max_laps=N_LAPS,
    )
    real_rl_xy = np.ascontiguousarray(env.raceline[:, 1:3])
    real_rl_psi = env.raceline[:, 3]

    lap_times = []
    rl_devs = []
    for _ in range(N_LAPS):
        obs, _ = env.reset()
        env.alpha_rl = alpha_rl
        steps = 0
        cumul_s = 0.0
        prev_s = env.raceline_s[env._closest_idx]
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            px, py = env._last_pos[0], env._last_pos[1]
            rl_idx = int(np.argmin((real_rl_xy[:, 0] - px)**2 + (real_rl_xy[:, 1] - py)**2))
            ref_pos = real_rl_xy[rl_idx]
            ref_psi = real_rl_psi[rl_idx]
            err = np.array([px - ref_pos[0], py - ref_pos[1]])
            d_rl = abs(-np.sin(ref_psi) * err[0] + np.cos(ref_psi) * err[1])
            rl_devs.append(d_rl)

            cur_s = env.raceline_s[env._closest_idx]
            ds = cur_s - prev_s
            if ds < -env.total_s / 2:
                ds += env.total_s
            elif ds > env.total_s / 2:
                ds -= env.total_s
            cumul_s += ds
            prev_s = cur_s
            if cumul_s >= env.total_s and steps > 100:
                lap_times.append(steps * env.controller_dt)
                break
            if terminated or truncated or steps > 100_000:
                break
    env.close()
    return lap_times, float(np.mean(rl_devs)) if rl_devs else None


def main():
    tracks_dir = os.path.join(_SCRIPT_DIR, "f1tenth_racetracks")
    ckpt = os.path.join(_SCRIPT_DIR, "reptile", "reptile_v3", "meta_params_best.pkl")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(_SCRIPT_DIR, "reptile", "reptile_v3", "meta_params_latest.pkl")

    print(f"Checkpoint: {ckpt}")
    with open(ckpt, "rb") as f:
        meta_params = jax.tree.map(jnp.array, pickle.load(f))

    args = Args()

    print(f"\n{'Track':<14} {'Config':<16} {'Laps':>5} {'Time':>8} {'|d_rl|':>8} {'Wall':>6}")
    print("-" * 65)

    for track in TRACKS:
        for label, scaling, vel_g, adapt in CONFIGS:
            env_kwargs = dict(
                alpha_rl=1.0, velocity_gain=vel_g,
                action_scaling=scaling,
                mu_noise_std=0.15, velocity_curriculum=True,
            )
            env = DummyVecEnv([make_env_fn(track, tracks_dir, seed=args.seed + 200, **env_kwargs)])
            model = create_model(env, args)
            load_params(model, meta_params)
            t0 = time.time()
            model.learn(total_timesteps=adapt)
            adapted = snapshot_params(model)
            env.close()
            adapt_s = time.time() - t0

            eval_env = DummyVecEnv([make_env_fn(
                track, tracks_dir, seed=args.seed + 300,
                alpha_rl=ALPHA_RL, velocity_gain=vel_g, action_scaling=scaling,
                mu_noise_std=0.0, velocity_curriculum=False,
            )])
            eval_model = create_model(eval_env, args)
            load_params(eval_model, adapted)
            eval_env.close()

            laps, mean_rl = evaluate(eval_model, track, tracks_dir, ALPHA_RL, vel_g, scaling)
            completed = len(laps)
            mean_t = float(np.mean(laps)) if laps else None
            t_str = f"{mean_t:.2f}" if mean_t else "DNF"
            rl_str = f"{mean_rl:.4f}" if mean_rl else "N/A"
            print(f"{track:<14} {label:<16} {completed:>2}/{N_LAPS}  {t_str:>8} {rl_str:>8} {adapt_s:>5.0f}s",
                  flush=True)

            del model, eval_model
            gc.collect()


if __name__ == "__main__":
    main()
