"""Train RLPP: Residual RL + Pure Pursuit on F1TENTH.

Replicates the training procedure from:
  Ghignone et al., "RLPP: A Residual Method for Zero-Shot Real-World
  Autonomous Racing on Scaled Platforms," arXiv 2501.17311v2, 2025.

SAC with MLP [256, 256], lr=3e-4, buffer=1e6, 2M steps.
"""

import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from residual_env import RLPPEnv


def make_env(track_name, alpha_rl=1.0, velocity_gain=0.75, mu_noise_std=0.15,
             velocity_curriculum=True):
    env = RLPPEnv(
        track_name=track_name,
        alpha_rl=alpha_rl,
        velocity_gain=velocity_gain,
        mu_noise_std=mu_noise_std,
        velocity_curriculum=velocity_curriculum,
    )
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser(description="Train RLPP")
    parser.add_argument("--track", default="Spielberg", help="Track name")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--alpha-rl", type=float, default=1.0,
                        help="RL residual scaling during training")
    parser.add_argument("--velocity-gain", type=float, default=0.75,
                        help="PP velocity gain (alpha_v) for sim training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train_env = make_env(args.track, args.alpha_rl, args.velocity_gain)
    eval_env = make_env(args.track, args.alpha_rl, args.velocity_gain, mu_noise_std=0.0,
                        velocity_curriculum=False)

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=f"./logs/rlpp_{args.track}/",
    )

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=100_000,
            save_path=f"./checkpoints/{args.track}/",
            name_prefix="rlpp",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"./best_model/{args.track}/",
            eval_freq=50_000,
            n_eval_episodes=10,
            deterministic=True,
        ),
    ])

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(f"rlpp_{args.track}_final")
    print(f"Training complete. Model saved to rlpp_{args.track}_final.zip")


if __name__ == "__main__":
    main()
