"""Evaluate raceline-shaped adaptation vs vanilla adaptation.

Takes a Reptile meta checkpoint and adapts on each held-out track twice:
  1. Vanilla (no raceline reward) — baseline
  2. Raceline-shaped (alpha_raceline > 0) — should hug the racing line

Measures lap times, completion rate, |d_cl|, and |d_rl|.
"""

import os
import sys
import gc
import pickle
import argparse
import time

import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp

from stable_baselines3.common.vec_env import DummyVecEnv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from residual_env import RLPPEnv
from train_reptile import (
    create_model, snapshot_params, load_params,
    make_env_fn, discover_valid_tracks,
)

HELD_OUT = ["Austin", "Monza", "Silverstone"]
ADAPT_STEPS = [4_000, 10_000, 20_000]
ALPHA_RL_EVAL = [0.55, 0.85]
N_LAPS = 10

REWARD_CONFIGS = {
    "vanilla": {},
    "raceline_0.3": {"alpha_raceline": 0.3},
    "raceline_0.5": {"alpha_raceline": 0.5},
    "raceline_1.0": {"alpha_raceline": 1.0},
    "raceline_0.5_prog": {"alpha_raceline": 0.5, "use_raceline_progress": True},
}


def evaluate_with_metrics(model, track_name, tracks_dir, n_laps, alpha_rl):
    env = RLPPEnv(
        track_name=track_name, tracks_dir=tracks_dir,
        alpha_rl=alpha_rl, velocity_gain=0.5,
        mu_noise_std=0.0, velocity_curriculum=False, max_laps=n_laps,
    )
    real_rl_xy = np.ascontiguousarray(env.raceline[:, 1:3])
    real_rl_psi = env.raceline[:, 3]

    lap_times = []
    all_cl_devs = []
    all_rl_devs = []
    for _ in range(n_laps):
        obs, _ = env.reset()
        env.alpha_rl = alpha_rl
        steps = 0
        cumul_s = 0.0
        prev_s = env.raceline_s[env._closest_idx]
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            px, py, theta = env._last_pos[0], env._last_pos[1], env._last_heading
            d_cl, _ = env.pp.get_frenet_state(px, py, theta, env._closest_idx)
            all_cl_devs.append(abs(d_cl))

            rl_idx = int(np.argmin((real_rl_xy[:, 0] - px)**2 + (real_rl_xy[:, 1] - py)**2))
            ref_pos = real_rl_xy[rl_idx]
            ref_psi = real_rl_psi[rl_idx]
            err = np.array([px - ref_pos[0], py - ref_pos[1]])
            d_rl = -np.sin(ref_psi) * err[0] + np.cos(ref_psi) * err[1]
            all_rl_devs.append(abs(d_rl))

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
    return (
        lap_times,
        float(np.mean(all_cl_devs)) if all_cl_devs else None,
        float(np.mean(all_rl_devs)) if all_rl_devs else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--utd", type=int, default=20)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--inner-steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    tracks_dir = os.path.join(_SCRIPT_DIR, "f1tenth_racetracks")

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(_SCRIPT_DIR, "reptile", "reptile_v3", "meta_params_best.pkl")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(_SCRIPT_DIR, "reptile", "reptile_v3", "meta_params_latest.pkl")

    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        meta_params_np = pickle.load(f)
    meta_params = jax.tree.map(jnp.array, meta_params_np)

    print(f"Reward configs: {list(REWARD_CONFIGS.keys())}")
    print(f"Adapt steps: {ADAPT_STEPS}")
    print(f"Eval α_rl: {ALPHA_RL_EVAL}")
    print()

    all_results = {}
    for track in HELD_OUT:
        print(f"{'='*70}")
        print(f"TRACK: {track}")
        print(f"{'='*70}")

        for rc_name, rc_overrides in REWARD_CONFIGS.items():
            for adapt in ADAPT_STEPS:
                print(f"\n  [{rc_name}] Adapting {adapt} steps...")
                t0 = time.time()

                env_kwargs = dict(
                    alpha_rl=1.0, velocity_gain=0.5,
                    mu_noise_std=0.15, velocity_curriculum=True,
                    reward_overrides=rc_overrides if rc_overrides else None,
                )
                env = DummyVecEnv([make_env_fn(
                    track, tracks_dir, seed=args.seed + 200, **env_kwargs,
                )])
                model = create_model(env, args)
                load_params(model, meta_params)
                model.learn(total_timesteps=adapt)
                adapted_params = snapshot_params(model)
                env.close()
                adapt_time = time.time() - t0

                for alpha in ALPHA_RL_EVAL:
                    eval_env = DummyVecEnv([make_env_fn(
                        track, tracks_dir, seed=args.seed + 300,
                        alpha_rl=alpha, velocity_gain=0.5,
                        mu_noise_std=0.0, velocity_curriculum=False,
                    )])
                    eval_model = create_model(eval_env, args)
                    load_params(eval_model, adapted_params)
                    eval_env.close()

                    laps, mean_cl, mean_rl = evaluate_with_metrics(
                        eval_model, track, tracks_dir, N_LAPS, alpha,
                    )
                    completed = len(laps)
                    mean_t = float(np.mean(laps)) if laps else None

                    key = (track, rc_name, adapt, alpha)
                    all_results[key] = {
                        "completed": completed,
                        "mean_time": mean_t,
                        "mean_cl_dev": mean_cl,
                        "mean_rl_dev": mean_rl,
                        "adapt_wall_s": adapt_time,
                    }
                    t_str = f"{mean_t:.2f}s" if mean_t else "DNF"
                    rl_str = f"{mean_rl:.4f}m" if mean_rl else "N/A"
                    print(f"    {rc_name:<20} adapt={adapt:>5d}  α={alpha:.2f}  "
                          f"laps={completed:>2}/{N_LAPS}  time={t_str:>8}  |d_rl|={rl_str}")

                    del eval_model
                    gc.collect()

                del model
                gc.collect()

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY — Best configs per track (by lap time, 10/10 completion required)")
    print(f"{'='*90}")
    for track in HELD_OUT:
        print(f"\n{track}:")
        track_rows = [(k, v) for k, v in all_results.items()
                      if k[0] == track and v["completed"] == N_LAPS]
        track_rows.sort(key=lambda x: x[1]["mean_time"])
        for (_, rc, adapt, alpha), r in track_rows[:8]:
            rl_str = f"{r['mean_rl_dev']:.4f}" if r['mean_rl_dev'] else "N/A"
            print(f"  {rc:<20} adapt={adapt:>5d}  α={alpha:.2f}  "
                  f"time={r['mean_time']:.2f}s  |d_rl|={rl_str}m")

    out_path = os.path.join(_SCRIPT_DIR, "reptile", "reptile_v3", "raceline_adapt_sweep.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
