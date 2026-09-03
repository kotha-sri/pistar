"""Quick A/B test: vanilla vs raceline-shaped adaptation on Austin only.

Runs 10K adapt steps with and without raceline reward, then evaluates.
Designed to run alongside training with minimal GPU footprint.
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

TRACK = "Austin"
ADAPT_STEPS = 10_000
N_LAPS = 10


class Args:
    seed = 42
    utd = 20
    inner_lr = 1e-3
    inner_steps = 50_000
    batch_size = 256


def evaluate(model, track, tracks_dir, alpha_rl):
    env = RLPPEnv(
        track_name=track, tracks_dir=tracks_dir,
        alpha_rl=alpha_rl, velocity_gain=0.5,
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


def adapt_and_eval(meta_params, label, reward_overrides, tracks_dir, alpha_rl):
    print(f"\n  [{label}] Adapting {ADAPT_STEPS} steps on {TRACK}...")
    t0 = time.time()
    env_kwargs = dict(
        alpha_rl=1.0, velocity_gain=0.5,
        mu_noise_std=0.15, velocity_curriculum=True,
        reward_overrides=reward_overrides if reward_overrides else None,
    )
    env = DummyVecEnv([make_env_fn(TRACK, tracks_dir, seed=42 + 200, **env_kwargs)])
    args = Args()
    model = create_model(env, args)
    load_params(model, meta_params)
    model.learn(total_timesteps=ADAPT_STEPS)
    adapted = snapshot_params(model)
    env.close()
    adapt_s = time.time() - t0

    eval_env = DummyVecEnv([make_env_fn(
        TRACK, tracks_dir, seed=42 + 300,
        alpha_rl=alpha_rl, velocity_gain=0.5,
        mu_noise_std=0.0, velocity_curriculum=False,
    )])
    eval_model = create_model(eval_env, args)
    load_params(eval_model, adapted)
    eval_env.close()

    laps, mean_rl = evaluate(eval_model, TRACK, tracks_dir, alpha_rl)
    del model, eval_model
    gc.collect()

    completed = len(laps)
    mean_t = float(np.mean(laps)) if laps else None
    t_str = f"{mean_t:.2f}s" if mean_t else "DNF"
    rl_str = f"{mean_rl:.4f}m" if mean_rl else "N/A"
    print(f"    laps={completed}/{N_LAPS}  time={t_str}  |d_rl|={rl_str}  [{adapt_s:.1f}s]")
    return {"completed": completed, "mean_time": mean_t, "mean_rl_dev": mean_rl}


def main():
    tracks_dir = os.path.join(_SCRIPT_DIR, "f1tenth_racetracks")
    ckpt = os.path.join(_SCRIPT_DIR, "reptile", "reptile_v3", "meta_params_best.pkl")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(_SCRIPT_DIR, "reptile", "reptile_v3", "meta_params_latest.pkl")

    print(f"Checkpoint: {ckpt}")
    with open(ckpt, "rb") as f:
        meta_params = jax.tree.map(jnp.array, pickle.load(f))

    configs = [
        ("vanilla", {}),
        ("raceline_0.5", {"alpha_raceline": 0.5}),
        ("raceline_1.0", {"alpha_raceline": 1.0}),
        ("raceline_0.5+prog", {"alpha_raceline": 0.5, "use_raceline_progress": True}),
    ]

    for alpha_rl in [0.55, 0.85]:
        print(f"\n{'='*60}")
        print(f"Eval α_rl = {alpha_rl}")
        print(f"{'='*60}")
        for label, overrides in configs:
            adapt_and_eval(meta_params, label, overrides, tracks_dir, alpha_rl)

    print("\nDone.")


if __name__ == "__main__":
    main()
