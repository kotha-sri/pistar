"""Train CrossQ on multi-track env for a generalizable base policy.

No dynamics model — pure CrossQ + multi-track. The resulting policy
should handle diverse track geometries, serving as the base for
online fine-tuning on unseen tracks.
"""

import argparse
import os
import time

import numpy as np
from sbx import CrossQ
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from multi_track_env import MultiTrackEnv
from residual_env import RLPPEnv


EVAL_TRACKS = ["Spielberg", "MoscowRaceway", "Silverstone"]


def make_multi_track_env(
    tracks_dir, generated_tracks_dir,
    n_generated, prob_generated, seed=None, **env_kwargs
):
    def _init():
        env = MultiTrackEnv(
            real_tracks=None,
            tracks_dir=tracks_dir,
            generated_tracks_dir=generated_tracks_dir,
            n_generated=n_generated,
            prob_generated=prob_generated,
            pregenerate=True,
            **env_kwargs,
        )
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)
    return _init


def make_eval_env(track_name, tracks_dir, seed=None, **env_kwargs):
    def _init():
        kw = dict(env_kwargs)
        kw["mu_noise_std"] = 0.0
        kw["velocity_curriculum"] = False
        env = RLPPEnv(track_name=track_name, tracks_dir=tracks_dir, **kw)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)
    return _init


def evaluate_on_track(model, track_name, tracks_dir, n_laps=10, alpha_rl=0.55):
    env = RLPPEnv(
        track_name=track_name, tracks_dir=tracks_dir,
        alpha_rl=alpha_rl, velocity_gain=0.75,
        mu_noise_std=0.0, velocity_curriculum=False, max_laps=n_laps,
    )
    lap_times = []
    for lap_i in range(n_laps):
        obs, _ = env.reset()
        env.alpha_rl = alpha_rl
        steps = 0
        cumul_s = 0.0
        prev_s = env.raceline_s[env._closest_idx]
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            cur_s = env.raceline_s[env._closest_idx]
            ds = cur_s - prev_s
            if ds < -env.total_s / 2: ds += env.total_s
            elif ds > env.total_s / 2: ds -= env.total_s
            cumul_s += ds
            prev_s = cur_s
            if cumul_s >= env.total_s and steps > 100:
                lap_times.append(steps * env.controller_dt)
                break
            if terminated or truncated or steps > 100000:
                break
    return lap_times


def main():
    parser = argparse.ArgumentParser(description="Train CrossQ on multi-track env")
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--utd", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--n-generated-tracks", type=int, default=50)
    parser.add_argument("--prob-generated", type=float, default=0.5)

    parser.add_argument("--tag", default="crossq_multitrack")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")
    gen_tracks_dir = os.path.join(script_dir, "generated_tracks")

    env_kwargs = dict(
        alpha_rl=1.0,
        velocity_gain=0.75,
        mu_noise_std=0.15,
        velocity_curriculum=True,
    )

    print(f"=== CrossQ Multi-Track Training: {args.tag} ===")
    print(f"  Steps: {args.steps}")
    print(f"  n_envs={args.n_envs}, UTD={args.utd}, lr={args.lr}")
    print(f"  Tracks: {args.n_generated_tracks} generated + real F1TENTH")
    print(f"  Eval tracks: {EVAL_TRACKS}")

    print("Creating multi-track environments...")
    train_env = SubprocVecEnv([
        make_multi_track_env(
            tracks_dir=tracks_dir,
            generated_tracks_dir=gen_tracks_dir,
            n_generated=args.n_generated_tracks,
            prob_generated=args.prob_generated,
            seed=args.seed + i, **env_kwargs,
        )
        for i in range(args.n_envs)
    ])

    eval_env = DummyVecEnv([
        make_eval_env(EVAL_TRACKS[0], tracks_dir, seed=args.seed + 100,
                      alpha_rl=0.55, velocity_gain=0.75)
    ])

    print("Initializing CrossQ policy...")
    model = CrossQ(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        buffer_size=1_000_000,
        batch_size=args.batch_size,
        gamma=0.99,
        gradient_steps=args.utd,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        tensorboard_log=f"./logs/crossq_multitrack/",
    )

    save_dir = f"./crossq/{args.tag}"
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=50_000,
            save_path=f"{save_dir}/checkpoints/",
            name_prefix="crossq",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"{save_dir}/best_model/",
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ])

    print(f"\n{'='*60}")
    print(f"Starting training ({args.steps} steps)...")
    print(f"{'='*60}\n")

    t0 = time.time()
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=args.tag,
    )
    wall_time = time.time() - t0

    model.save(f"{save_dir}/final_model")

    print(f"\n=== {args.tag} complete ===")
    print(f"  Wall clock: {wall_time:.1f}s ({wall_time/60:.1f}m)")
    print(f"  Steps: {args.steps}")
    print(f"  Effective FPS: {args.steps / wall_time:.0f}")

    # Evaluate on multiple tracks
    best_path = f"{save_dir}/best_model/best_model.zip"
    if os.path.exists(best_path):
        best_model = CrossQ.load(best_path)
        print("\nUsing best model checkpoint")
    else:
        best_model = model
        print("\nUsing final model")

    print(f"\n=== Evaluation (alpha_rl=0.55, 10 laps each) ===")
    for track in EVAL_TRACKS:
        print(f"\n--- {track} ---")
        lap_times = evaluate_on_track(best_model, track, tracks_dir)
        if lap_times:
            print(f"  {len(lap_times)}/10 completed")
            print(f"  Mean: {np.mean(lap_times):.2f} +/- {np.std(lap_times):.3f}s")
            print(f"  Best: {np.min(lap_times):.2f}s")
        else:
            print(f"  0/10 completed")


if __name__ == "__main__":
    main()
