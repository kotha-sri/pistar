"""Ablation: does GEOMETRY variation in the training distribution help or HURT
robustness to DYNAMICS (grip) variation, when a base controller (PP) is present?

Thesis under test:
  With a base controller absorbing track geometry, varying geometry during
  training adds little/negative value for dynamics robustness, while varying
  dynamics is what actually matters. Standard practice varies geometry (many
  tracks) and fixes dynamics -- the wrong axis.

Design (plain CrossQ residual, NO meta-learning -- cleanest test of the
mechanism; meta is just one way to consume the same distribution):

  Arms (each trained for the SAME number of env steps):
    A) SINGLE_DYN : one track (Spielberg),  dynamics VARIED   (mu_noise=0.2)
    B) MULTI_DYN  : many tracks (pool),      dynamics VARIED   (mu_noise=0.2)
    C) SINGLE_FIX : one track (Spielberg),  dynamics FIXED    (mu_noise=0.0)
       (control: shows whether dynamics variation matters at all)

  Evaluation: each trained policy is tested at a GRID of fixed grip values mu,
  on two tracks:
    - Spielberg  (geometry KNOWN to A and C; seen ~1/N of the time by B)
    - Austin     (geometry HELD OUT for all arms)
  Metrics per (track, mu): laps completed, mean |cross-track deviation|, mean vx.

Reading the result:
  - A vs C on the mu grid  -> does dynamics variation buy robustness at all?
  - A vs B on Spielberg    -> does geometry variation help/hurt on a KNOWN track?
  - A vs B on Austin       -> does geometry variation help dynamics robustness
                              on a NEW track? (standard prior says B >> A;
                              thesis says ~equal because base absorbs geometry)

Outputs a JSON with all (arm, seed, track, mu) cells.
"""

import argparse
import gc
import json
import os
import sys
import time
import types

import numpy as np
from sbx import CrossQ
from stable_baselines3.common.vec_env import SubprocVecEnv

from residual_env import RLPPEnv
from multi_track_env import MultiTrackEnv


HELD_OUT_TRACKS = ["Austin", "Monza", "Silverstone"]
TRAIN_TRACK = "Spielberg"            # known track, in the MULTI pool too
EVAL_TRACKS = ["Spielberg", "Austin"]  # known-geometry, held-out-geometry
MU_GRID = [0.2, 0.35, 0.5, 0.65, 0.8]  # fixed grip values to probe robustness

# Env config shared with the real training regime (train_reptile v3 defaults)
BASE_ENV_KWARGS = dict(
    alpha_rl=1.0,
    velocity_gain=0.5,
    velocity_curriculum=True,
    action_scaling=(0.05, 1.0),
)


def discover_valid_tracks(tracks_dir):
    tracks = []
    if not os.path.isdir(tracks_dir):
        return tracks
    for name in sorted(os.listdir(tracks_dir)):
        td = os.path.join(tracks_dir, name)
        if not os.path.isdir(td):
            continue
        rp = os.path.join(td, f"{name}_raceline.csv")
        cp = os.path.join(td, f"{name}_centerline.csv")
        mp = os.path.join(td, f"{name}_map.png")
        if not (os.path.exists(rp) and os.path.exists(cp) and os.path.exists(mp)):
            continue
        try:
            np.loadtxt(rp, delimiter=";", skiprows=3)
            np.loadtxt(cp, delimiter=",", skiprows=1)
        except (ValueError, OSError):
            continue
        tracks.append(name)
    return tracks


def make_single_env_fn(track, tracks_dir, mu_noise, seed):
    def _init():
        env = RLPPEnv(track_name=track, tracks_dir=tracks_dir,
                      mu_noise_std=mu_noise, **BASE_ENV_KWARGS)
        env.reset(seed=seed)
        return env
    return _init


def make_multi_env_fn(pool, tracks_dir, mu_noise, seed):
    def _init():
        env = MultiTrackEnv(real_tracks=pool, tracks_dir=tracks_dir,
                            prob_generated=0.0, pregenerate=False,
                            mu_noise_std=mu_noise, **BASE_ENV_KWARGS)
        env.reset(seed=seed)
        return env
    return _init


def create_model(env, seed, utd, inner_lr, batch_size):
    return CrossQ(
        policy="MlpPolicy",
        env=env,
        learning_rate=inner_lr,
        buffer_size=100_000,
        batch_size=batch_size,
        gamma=0.99,
        gradient_steps=utd,
        learning_starts=batch_size,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        seed=seed,
    )


def _fixed_mu(self):
    """Replacement for _randomize_friction: pin grip to self._test_mu."""
    params = dict(self.params)
    params["mu"] = self._test_mu
    self.sim.update_params(params)


def eval_at_mu(model, track, tracks_dir, mu, n_laps=3, max_steps=20000):
    """Run n_laps at a FIXED grip mu; return completion, mean|d|, mean vx."""
    env = RLPPEnv(track_name=track, tracks_dir=tracks_dir,
                  mu_noise_std=0.0, max_laps=n_laps, **BASE_ENV_KWARGS)
    # Pin grip to the test value on every reset.
    env._test_mu = mu
    env._randomize_friction = types.MethodType(_fixed_mu, env)

    completed = 0
    devs = []
    vxs = []
    for _ in range(n_laps):
        obs, _ = env.reset()
        steps = 0
        cumul_s = 0.0
        prev_s = env.raceline_s[env._closest_idx]
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            devs.append(abs(float(obs[0])))     # obs[0] = cross-track deviation d
            vxs.append(float(info.get("vx", 0.0)))
            cur_s = env.raceline_s[env._closest_idx]
            ds = cur_s - prev_s
            if ds < -env.total_s / 2:
                ds += env.total_s
            elif ds > env.total_s / 2:
                ds -= env.total_s
            cumul_s += ds
            prev_s = cur_s
            if cumul_s >= env.total_s and steps > 100:
                completed += 1
                break
            if terminated or truncated or steps > max_steps:
                break
    env.close()
    return {
        "completed": completed,
        "n_laps": n_laps,
        "mean_abs_dev": float(np.mean(devs)) if devs else None,
        "mean_vx": float(np.mean(vxs)) if vxs else None,
    }


def run_arm(arm, seed, args, tracks_dir, pool):
    t0 = time.time()
    if arm == "SINGLE_DYN":
        mu_noise = args.mu_noise
        fns = [make_single_env_fn(TRAIN_TRACK, tracks_dir, mu_noise,
                                  seed * 100 + i) for i in range(args.envs)]
    elif arm == "MULTI_DYN":
        mu_noise = args.mu_noise
        fns = [make_multi_env_fn(pool, tracks_dir, mu_noise,
                                 seed * 100 + i) for i in range(args.envs)]
    elif arm == "SINGLE_FIX":
        mu_noise = 0.0
        fns = [make_single_env_fn(TRAIN_TRACK, tracks_dir, mu_noise,
                                  seed * 100 + i) for i in range(args.envs)]
    else:
        raise ValueError(arm)

    venv = SubprocVecEnv(fns)
    model = create_model(venv, seed, args.utd, args.inner_lr, args.batch_size)
    print(f"[{arm} seed={seed}] training {args.steps} steps "
          f"(mu_noise={mu_noise}, envs={args.envs})...", flush=True)
    model.learn(total_timesteps=args.steps)
    train_t = time.time() - t0
    venv.close()

    cells = {}
    for track in EVAL_TRACKS:
        for mu in MU_GRID:
            res = eval_at_mu(model, track, tracks_dir, mu, n_laps=args.eval_laps)
            cells[f"{track}|mu={mu}"] = res
            print(f"    {arm} s{seed} {track:<10s} mu={mu:.2f} -> "
                  f"{res['completed']}/{res['n_laps']} laps, "
                  f"|d|={res['mean_abs_dev']}, vx={res['mean_vx']}", flush=True)
    del model
    gc.collect()
    print(f"[{arm} seed={seed}] done in {(time.time()-t0)/60:.1f}m "
          f"(train {train_t/60:.1f}m)", flush=True)
    return {"arm": arm, "seed": seed, "mu_noise": mu_noise,
            "train_seconds": train_t, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--utd", type=int, default=20)
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--mu-noise", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+",
                    default=["SINGLE_DYN", "MULTI_DYN", "SINGLE_FIX"])
    ap.add_argument("--eval-laps", type=int, default=3)
    ap.add_argument("--out", default="ablation_geom_vs_dyn_results.json")
    ap.add_argument("--tag", default="abl1")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")
    all_tracks = discover_valid_tracks(tracks_dir)
    pool = [t for t in all_tracks if t not in HELD_OUT_TRACKS]

    print("=== Ablation: geometry vs dynamics variation ===")
    print(f"  steps/arm={args.steps} envs={args.envs} utd={args.utd} "
          f"mu_noise={args.mu_noise}")
    print(f"  arms={args.arms} seeds={args.seeds}")
    print(f"  train pool ({len(pool)}): {pool[:8]}...")
    print(f"  eval tracks={EVAL_TRACKS} mu_grid={MU_GRID} eval_laps={args.eval_laps}")
    print(f"  TRAIN_TRACK={TRAIN_TRACK} in pool: {TRAIN_TRACK in pool}", flush=True)

    results = []
    t0 = time.time()
    for arm in args.arms:
        for seed in args.seeds:
            r = run_arm(arm, seed, args, tracks_dir, pool)
            results.append(r)
            with open(args.out, "w") as f:
                json.dump({"args": vars(args), "results": results}, f, indent=2)
            print(f"  [checkpoint] wrote {args.out} "
                  f"({len(results)} runs, {(time.time()-t0)/60:.1f}m elapsed)\n",
                  flush=True)

    print(f"=== DONE in {(time.time()-t0)/60:.1f}m -> {args.out} ===")


if __name__ == "__main__":
    main()
