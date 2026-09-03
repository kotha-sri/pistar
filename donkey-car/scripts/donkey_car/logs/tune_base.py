"""
tune_base.py — deterministic single tuning run for the BASE PathFollower,
including the speed-scheduled steering gains. Built for parallel gain sweeps:
all parameters are CLI flags (nothing is read from / written to path_follower.py
on disk), so many copies can run at once against different sim ports without
clobbering shared state.

Usage (all flags optional except they default to the current CONFIG):
    python logs/tune_base.py --port 9093 --target_speed 1.6 --duration 40 \
        --kp 2.5 --kd 1.6 --sched_kp_exp 1.0 --sched_kd_exp 0.0 --sched_ref_speed 0.8

Drives the base alone (no RL residual) for `duration` seconds at the requested
target_speed, then prints one machine-parsable metrics line. The point of this
tool is to find gains that give CLEAN, OSCILLATION-FREE tracking at the higher
speeds the RL residual will push the car to — speed itself is not the goal.

Headline oscillation metrics (LOWER = cleaner):
    cte_zc%    : % of samples where CTE flips sign  (weaving across centerline)
    steer_jerk : mean |Δsteering| per step          (how much the wheel jitters)
    max|cte|   : worst offset from center
    resets     : times the car left the track (>4.0). MUST be 0 for a usable base.
"""
import argparse, sys, os, time, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from donkey_sim_client import DonkeySimClient
from path_follower import PathFollower

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=9091)
p.add_argument("--duration", type=float, default=40.0)
p.add_argument("--kp", type=float, default=2.5)
p.add_argument("--ki", type=float, default=0.0)
p.add_argument("--kd", type=float, default=1.6)
p.add_argument("--target_speed", type=float, default=1.6)
p.add_argument("--throttle_kp", type=float, default=1.0)
p.add_argument("--sched_ref_speed", type=float, default=0.8)
p.add_argument("--sched_kp_exp", type=float, default=1.0)
p.add_argument("--sched_kd_exp", type=float, default=0.0)
p.add_argument("--label", type=str, default="")
a = p.parse_args()

ctrl = PathFollower()
ctrl.update_config({
    "kp": a.kp, "ki": a.ki, "kd": a.kd,
    "target_speed": a.target_speed, "throttle_kp": a.throttle_kp,
    "sched_ref_speed": a.sched_ref_speed,
    "sched_kp_exp": a.sched_kp_exp, "sched_kd_exp": a.sched_kd_exp,
})

client = DonkeySimClient("127.0.0.1", a.port)
samples = []          # (cte, speed, steering)
resets = [0]

def on_telemetry(msg):
    cte = float(msg.get("cte", 0.0))
    speed = float(msg.get("speed", 0.0))
    if abs(cte) > 4.0:
        client.send_control(0.0, 0.0)
        ctrl.reset()
        resets[0] += 1
        client.reset_car()
        return
    s, t = ctrl.compute_controls(cte, speed)
    client.send_control(s, t)
    samples.append((cte, speed, s))

client.connect()
client.start_listening()
time.sleep(1.0)
client.load_scene("generated_track")
time.sleep(2.0)
client.reset_car()
ctrl.reset()
time.sleep(0.5)
client.set_telemetry_callback(on_telemetry)

time.sleep(a.duration)
client.send_control(0.0, 0.0)
time.sleep(0.2)
client.disconnect()

# stats — skip launch-from-standstill transient
data = samples[8:]
if not data:
    print("NO TELEMETRY (sim not responding on this port?)"); sys.exit(1)

c  = [d[0] for d in data]
sp = [d[1] for d in data]
st = [d[2] for d in data]
absc = [abs(x) for x in c]

rms   = (sum(x*x for x in c) / len(c)) ** 0.5
zc    = sum(1 for i in range(1, len(c)) if (c[i] > 0) != (c[i-1] > 0))
sjerk = statistics.mean(abs(st[i] - st[i-1]) for i in range(1, len(st))) if len(st) > 1 else 0.0
szc   = sum(1 for i in range(1, len(st)) if (st[i] > 0) != (st[i-1] > 0))

label = f"[{a.label}] " if a.label else ""
print(
    f"{label}kp={a.kp} kd={a.kd} kpexp={a.sched_kp_exp} kdexp={a.sched_kd_exp} "
    f"tgt={a.target_speed} | n={len(c)} resets={resets[0]} | "
    f"mean|cte|={statistics.mean(absc):.3f} max|cte|={max(absc):.3f} "
    f"cte_std={statistics.pstdev(c):.3f} cte_zc%={100*zc/len(c):.1f} | "
    f"steer_jerk={sjerk:.4f} steer_zc%={100*szc/max(len(st)-1,1):.1f} | "
    f"spd={statistics.mean(sp):.2f}"
)
