"""Train RLPP with ablation configs for wall-clock experiments.

Supports: vectorized envs, variable decision rate, higher UTD ratio.
"""

import argparse
import time

from stable_baselines3 import SAC
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
    parser = argparse.ArgumentParser(description="Train RLPP ablation")
    parser.add_argument("--track", default="Spielberg")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--tag", required=True, help="Experiment tag for log/save dirs")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs")
    parser.add_argument("--controller-dt", type=float, default=0.01,
                        help="Decision interval in seconds (0.01=100Hz, 0.05=20Hz)")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="Gradient updates per env step (UTD ratio)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    print(f"=== Experiment: {args.tag} ===")
    print(f"  n_envs={args.n_envs}, controller_dt={args.controller_dt}, "
          f"gradient_steps={args.gradient_steps}, steps={args.steps}")

    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            make_env_fn(args.track, controller_dt=args.controller_dt, seed=args.seed + i)
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([
            make_env_fn(args.track, controller_dt=args.controller_dt, seed=args.seed)
        ])

    eval_env = DummyVecEnv([
        make_env_fn(args.track, mu_noise_std=0.0, velocity_curriculum=False,
                    controller_dt=args.controller_dt, seed=args.seed + 100)
    ])

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        gradient_steps=args.gradient_steps,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=f"./logs/ablation_{args.track}/",
    )

    save_dir = f"./ablation/{args.tag}"
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=100_000,
            save_path=f"{save_dir}/checkpoints/",
            name_prefix="rlpp",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"{save_dir}/best_model/",
            eval_freq=50_000,
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


if __name__ == "__main__":
    main()
