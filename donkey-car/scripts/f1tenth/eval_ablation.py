"""Evaluate all ablation experiments."""

import argparse
import os
import numpy as np
from stable_baselines3 import SAC
from residual_env import RLPPEnv


def run_eval(env, model, n_laps=10, alpha_rl=0.55):
    lap_times = []
    for lap in range(n_laps):
        obs, _ = env.reset()
        env.alpha_rl = alpha_rl
        steps = 0
        cumul_s = 0.0
        prev_s = env.raceline_s[env._closest_idx]

        while True:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(2, dtype=np.float32)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="Spielberg")
    parser.add_argument("--n-laps", type=int, default=10)
    args = parser.parse_args()

    experiments = {
        "baseline": {"controller_dt": 0.01},
        "vec16": {"controller_dt": 0.01},
        "hz20": {"controller_dt": 0.05},
        "utd4": {"controller_dt": 0.01},
    }

    print(f"{'Experiment':<12} {'Laps':>6} {'Mean':>8} {'Best':>8} {'Std':>8}")
    print("-" * 50)

    for tag, cfg in experiments.items():
        best_path = f"ablation/{tag}/best_model/best_model.zip"
        final_path = f"ablation/{tag}/final_model.zip"
        model_path = best_path if os.path.exists(best_path) else final_path

        if not os.path.exists(model_path):
            print(f"{tag:<12} {'N/A':>6} (model not found)")
            continue

        env = RLPPEnv(
            track_name=args.track, alpha_rl=0.55, velocity_gain=0.75,
            mu_noise_std=0.0, velocity_curriculum=False, max_laps=10,
            controller_dt=cfg["controller_dt"],
        )
        model = SAC.load(model_path)
        laps = run_eval(env, model, n_laps=args.n_laps)

        if laps:
            print(f"{tag:<12} {len(laps):>4}/10 {np.mean(laps):>8.2f} {np.min(laps):>8.2f} {np.std(laps):>8.3f}")
        else:
            print(f"{tag:<12} {'0/10':>6}")


if __name__ == "__main__":
    main()
