"""Train RLPP with CrossQ: BatchNorm critic, no target networks, high UTD.

CrossQ (Bhatt et al., 2024) enables stable training at UTD=20 with
4x less gradient compute than REDQ ensembles. Combined with vectorized
envs, this is the first step toward few-lap convergence.

Uses SBX (Stable Baselines Jax) for the CrossQ implementation.
"""

import argparse
import time

from sbx import CrossQ
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from residual_env import RLPPEnv


def make_env_fn(track_name, alpha_rl=1.0, velocity_gain=0.75, mu_noise_std=0.15,
                velocity_curriculum=True, controller_dt=0.01, seed=None):
    def _init():
        env = RLPPEnv(
            track_name=track_name,
            alpha_rl=alpha_rl,
            velocity_gain=velocity_gain,
            mu_noise_std=mu_noise_std,
            velocity_curriculum=velocity_curriculum,
            controller_dt=controller_dt,
        )
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train RLPP with CrossQ")
    parser.add_argument("--track", default="Spielberg")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--utd", type=int, default=20,
                        help="Update-to-data ratio (gradient_steps)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (CrossQ default is 1e-3)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tag", default="crossq_utd20_vec16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"=== CrossQ Training: {args.tag} ===")
    print(f"  n_envs={args.n_envs}, UTD={args.utd}, lr={args.lr}, steps={args.steps}")

    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            make_env_fn(args.track, seed=args.seed + i)
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([
            make_env_fn(args.track, seed=args.seed)
        ])

    eval_env = DummyVecEnv([
        make_env_fn(args.track, mu_noise_std=0.0, velocity_curriculum=False,
                    seed=args.seed + 100)
    ])

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
        tensorboard_log=f"./logs/crossq_{args.track}/",
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
            eval_freq=25_000,
            n_eval_episodes=10,
            deterministic=True,
        ),
    ])

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
    print(f"  Effective FPS: {args.steps/wall_time:.0f}")

    # Quick inline eval
    print(f"\n=== Quick evaluation ===")
    import numpy as np
    eval_single = RLPPEnv(
        track_name=args.track, alpha_rl=0.55, velocity_gain=0.75,
        mu_noise_std=0.0, velocity_curriculum=False, max_laps=10,
    )
    best_path = f"{save_dir}/best_model/best_model.zip"
    import os
    if os.path.exists(best_path):
        best_model = CrossQ.load(best_path)
        print("Using best model checkpoint")
    else:
        best_model = model
        print("Using final model")

    lap_times = []
    for lap in range(10):
        obs, _ = eval_single.reset()
        eval_single.alpha_rl = 0.55
        steps = 0
        cumul_s = 0.0
        prev_s = eval_single.raceline_s[eval_single._closest_idx]
        while True:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_single.step(action)
            steps += 1
            cur_s = eval_single.raceline_s[eval_single._closest_idx]
            ds = cur_s - prev_s
            if ds < -eval_single.total_s / 2: ds += eval_single.total_s
            elif ds > eval_single.total_s / 2: ds -= eval_single.total_s
            cumul_s += ds
            prev_s = cur_s
            if cumul_s >= eval_single.total_s and steps > 100:
                lap_times.append(steps * eval_single.controller_dt)
                print(f"  Lap {lap+1}: {lap_times[-1]:.2f}s")
                break
            if terminated or truncated or steps > 100000:
                print(f"  Lap {lap+1}: DNF (steps={steps})")
                break

    if lap_times:
        print(f"\n  {len(lap_times)}/10 completed")
        print(f"  Mean: {np.mean(lap_times):.2f} +/- {np.std(lap_times):.3f}s")
        print(f"  Best: {np.min(lap_times):.2f}s")
    else:
        print("\n  0/10 completed")


if __name__ == "__main__":
    main()
