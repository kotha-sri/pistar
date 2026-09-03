"""Sweep action scaling and velocity gain to unlock raceline proximity.

The current bottleneck is NOT the reward function — it's physical constraints:
  - Steering residual: ±0.05 rad (2.9°) — can't reach raceline 0.64m from centerline
  - Velocity: PP commands 4.0 m/s, residual adds ±1.0 — car tops at ~5 m/s vs raceline's 7+ m/s

This script adapts from the Reptile meta-init with different action_scaling and
velocity_gain values, then evaluates lap time and raceline deviation.
"""

import os
import sys
import gc
import pickle
import time
import itertools

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
ADAPT_STEPS = 10_000
N_LAPS = 10

STEERING_SCALES = [0.05, 0.10, 0.15]
VELOCITY_SCALES = [1.0, 2.0]
VELOCITY_GAINS = [0.5, 0.70]
ALPHA_RL = 0.85


class Args:
    seed = 42
    utd = 20
    inner_lr = 1e-3
    inner_steps = 50_000
    batch_size = 256


def evaluate(model, track, tracks_dir, alpha_rl, velocity_gain, action_scaling):
    env = RLPPEnv(
        track_name=track, tracks_dir=tracks_dir,
        alpha_rl=alpha_rl, velocity_gain=velocity_gain,
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
    all_results = {}

    # First pass: sweep steering and velocity scaling on Austin only to find best combo
    track = "Austin"
    print(f"\n{'='*70}")
    print(f"PHASE 1: Action scaling sweep on {track}")
    print(f"{'='*70}")

    for steer_s, vel_s, vel_g in itertools.product(STEERING_SCALES, VELOCITY_SCALES, VELOCITY_GAINS):
        action_scaling = (steer_s, vel_s)
        label = f"steer={steer_s:.2f} vel_s={vel_s:.1f} vel_g={vel_g:.2f}"
        print(f"\n  [{label}] Adapting {ADAPT_STEPS} steps...")
        t0 = time.time()

        env_kwargs = dict(
            alpha_rl=1.0, velocity_gain=vel_g,
            action_scaling=action_scaling,
            mu_noise_std=0.15, velocity_curriculum=True,
        )
        env = DummyVecEnv([make_env_fn(track, tracks_dir, seed=args.seed + 200, **env_kwargs)])
        model = create_model(env, args)
        load_params(model, meta_params)
        model.learn(total_timesteps=ADAPT_STEPS)
        adapted = snapshot_params(model)
        env.close()
        adapt_s = time.time() - t0

        eval_env = DummyVecEnv([make_env_fn(
            track, tracks_dir, seed=args.seed + 300,
            alpha_rl=ALPHA_RL, velocity_gain=vel_g, action_scaling=action_scaling,
            mu_noise_std=0.0, velocity_curriculum=False,
        )])
        eval_model = create_model(eval_env, args)
        load_params(eval_model, adapted)
        eval_env.close()

        laps, mean_rl = evaluate(eval_model, track, tracks_dir, ALPHA_RL, vel_g, action_scaling)
        completed = len(laps)
        mean_t = float(np.mean(laps)) if laps else None

        key = (track, steer_s, vel_s, vel_g)
        all_results[key] = {
            "completed": completed, "mean_time": mean_t,
            "mean_rl_dev": mean_rl, "adapt_wall_s": adapt_s,
        }
        t_str = f"{mean_t:.2f}s" if mean_t else "DNF"
        rl_str = f"{mean_rl:.4f}m" if mean_rl else "N/A"
        print(f"    laps={completed:>2}/{N_LAPS}  time={t_str:>8}  |d_rl|={rl_str}  [{adapt_s:.0f}s]")

        del model, eval_model
        gc.collect()

    # Find best config (lowest time with 10/10 completion)
    austin_results = {k: v for k, v in all_results.items() if k[0] == track and v["completed"] == N_LAPS}
    if austin_results:
        best_key = min(austin_results, key=lambda k: austin_results[k]["mean_time"])
        best = austin_results[best_key]
        _, best_steer, best_vel_s, best_vel_g = best_key
        print(f"\n  BEST: steer={best_steer:.2f} vel_s={best_vel_s:.1f} vel_g={best_vel_g:.2f}")
        print(f"    time={best['mean_time']:.2f}s  |d_rl|={best['mean_rl_dev']:.4f}m")

        # Phase 2: Test best config on all held-out tracks
        print(f"\n{'='*70}")
        print(f"PHASE 2: Best config on all held-out tracks")
        print(f"{'='*70}")
        best_scaling = (best_steer, best_vel_s)

        for test_track in TRACKS:
            if test_track == track:
                print(f"\n  {test_track}: (from Phase 1) time={best['mean_time']:.2f}s  |d_rl|={best['mean_rl_dev']:.4f}m")
                continue
            print(f"\n  {test_track}: Adapting {ADAPT_STEPS} steps...")
            t0 = time.time()
            env_kwargs = dict(
                alpha_rl=1.0, velocity_gain=best_vel_g,
                action_scaling=best_scaling,
                mu_noise_std=0.15, velocity_curriculum=True,
            )
            env = DummyVecEnv([make_env_fn(test_track, tracks_dir, seed=args.seed + 200, **env_kwargs)])
            model = create_model(env, args)
            load_params(model, meta_params)
            model.learn(total_timesteps=ADAPT_STEPS)
            adapted = snapshot_params(model)
            env.close()
            adapt_s = time.time() - t0

            eval_env = DummyVecEnv([make_env_fn(
                test_track, tracks_dir, seed=args.seed + 300,
                alpha_rl=ALPHA_RL, velocity_gain=best_vel_g, action_scaling=best_scaling,
                mu_noise_std=0.0, velocity_curriculum=False,
            )])
            eval_model = create_model(eval_env, args)
            load_params(eval_model, adapted)
            eval_env.close()

            laps, mean_rl = evaluate(eval_model, test_track, tracks_dir, ALPHA_RL, best_vel_g, best_scaling)
            completed = len(laps)
            mean_t = float(np.mean(laps)) if laps else None
            t_str = f"{mean_t:.2f}s" if mean_t else "DNF"
            rl_str = f"{mean_rl:.4f}m" if mean_rl else "N/A"
            print(f"    laps={completed:>2}/{N_LAPS}  time={t_str:>8}  |d_rl|={rl_str}  [{adapt_s:.0f}s]")

            del model, eval_model
            gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("FULL RESULTS")
    print(f"{'='*70}")
    print(f"{'Track':<14} {'Steer':>6} {'VelS':>5} {'VelG':>5} {'Laps':>5} {'Time':>8} {'|d_rl|':>8}")
    for key in sorted(all_results.keys()):
        tr, ss, vs, vg = key
        r = all_results[key]
        t_str = f"{r['mean_time']:.2f}" if r['mean_time'] else "DNF"
        rl_str = f"{r['mean_rl_dev']:.4f}" if r['mean_rl_dev'] else "N/A"
        print(f"{tr:<14} {ss:>6.2f} {vs:>5.1f} {vg:>5.2f} {r['completed']:>2}/{N_LAPS}  {t_str:>8} {rl_str:>8}")

    out_path = os.path.join(_SCRIPT_DIR, "reptile", "reptile_v3", "action_scaling_sweep.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
