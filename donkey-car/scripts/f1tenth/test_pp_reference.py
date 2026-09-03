"""PP-ALONE test (no RL, no training): does pointing Pure Pursuit at a
feasible raceline instead of the centerline cut the deviation-from-raceline
floor while still completing laps?

For each track, run PP alone (residual = 0) for a few laps with:
  - pp_reference="centerline" (current default)
  - pp_reference="raceline"   (new feasible raceline)
and report: laps completed, mean |d_rl| (distance to the TRUE optimal
raceline), and lap time.
"""
import os
import sys
import numpy as np
from residual_env import RLPPEnv

TRACKS = ["Spielberg", "Austin", "Monza", "Silverstone"]
N_LAPS = 3
MAX_STEPS = 20000


def d_rl(env, px, py):
    """Lateral distance from (px,py) to the TRUE optimal raceline."""
    i = int(np.argmin((env.real_rl_xy[:, 0] - px) ** 2
                      + (env.real_rl_xy[:, 1] - py) ** 2))
    rp = env.real_rl_xy[i]
    rpsi = env.real_rl_psi[i]
    ex, ey = px - rp[0], py - rp[1]
    return abs(-np.sin(rpsi) * ex + np.cos(rpsi) * ey)


def run(track, ref):
    env = RLPPEnv(track_name=track, pp_reference=ref,
                  alpha_rl=1.0, velocity_gain=0.5, mu_noise_std=0.0,
                  velocity_curriculum=False, action_scaling=(0.05, 1.0),
                  max_laps=N_LAPS)
    completed = 0
    devs = []
    vxs = []
    zero = np.zeros(2, dtype=np.float32)
    for _ in range(N_LAPS):
        obs, _ = env.reset()
        steps = 0
        cumul_s = 0.0
        prev_s = env.raceline_s[env._closest_idx]
        while True:
            obs, r, term, trunc, info = env.step(zero)   # PP alone
            steps += 1
            devs.append(d_rl(env, env._last_pos[0], env._last_pos[1]))
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
            if term or trunc or steps > MAX_STEPS:
                break
    env.close()
    return {
        "completed": completed,
        "mean_d_rl": float(np.mean(devs)) if devs else None,
        "mean_vx": float(np.mean(vxs)) if vxs else None,
    }


def main():
    print(f"=== PP-alone: centerline vs feasible-raceline reference ===")
    print(f"{'track':<12s} {'ref':<11s} {'laps':<6s} {'mean|d_rl|':<12s} {'mean_vx':<8s}")
    for track in TRACKS:
        for ref in ["centerline", "raceline"]:
            try:
                res = run(track, ref)
                print(f"{track:<12s} {ref:<11s} {res['completed']}/{N_LAPS}    "
                      f"{res['mean_d_rl']:<12.4f} {res['mean_vx']:<8.3f}", flush=True)
            except Exception as e:
                print(f"{track:<12s} {ref:<11s} ERROR: {e}", flush=True)
        print()


if __name__ == "__main__":
    main()
