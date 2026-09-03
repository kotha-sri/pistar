"""
trial.py — deterministic single tuning run for offline gain sweeps.

Usage:
    python logs/trial.py kp ki kd target_speed throttle_kp duration [port]

Connects, loads the scene, resets the car to start, drives with PathFollower
for `duration` seconds, then disconnects cleanly and prints CTE stats.
"""
import sys, os, time, threading, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from donkey_sim_client import DonkeySimClient
from path_follower import PathFollower

kp, ki, kd, tgt, thr, dur = (float(a) for a in sys.argv[1:7])
port = int(sys.argv[7]) if len(sys.argv) > 7 else 9091

client = DonkeySimClient("127.0.0.1", port)
ctrl = PathFollower()
ctrl.update_config({"kp": kp, "ki": ki, "kd": kd, "target_speed": tgt, "throttle_kp": thr})

samples = []          # (cte, speed)
resets = [0]
started = [False]
t0 = [0.0]

def on_telemetry(msg):
    cte = float(msg.get("cte", 0.0))
    speed = float(msg.get("speed", 0.0))
    if not started[0]:
        started[0] = True
        t0[0] = time.time()
    samples.append((cte, speed))
    if abs(cte) > 4.0:
        client.send_control(0.0, 0.0)
        ctrl.reset()
        resets[0] += 1
        client.reset_car()
        return
    s, t = ctrl.compute_controls(cte, speed)
    client.send_control(s, t)

client.connect()
client.start_listening()
time.sleep(1.0)
client.load_scene("generated_track")
time.sleep(2.0)
client.reset_car()
ctrl.reset()
time.sleep(0.5)
client.set_telemetry_callback(on_telemetry)

time.sleep(dur)
client.send_control(0.0, 0.0)
time.sleep(0.2)
client.disconnect()

# stats — skip launch transient
c = [s[0] for s in samples][8:]
sp = [s[1] for s in samples][8:]
if not c:
    print("NO TELEMETRY"); sys.exit()
absc = [abs(x) for x in c]
rms = (sum(x*x for x in c) / len(c)) ** 0.5
zc = sum(1 for i in range(1, len(c)) if (c[i] > 0) != (c[i-1] > 0))
print(f"kp={kp} ki={ki} kd={kd} spd={tgt} | "
      f"n={len(c)} resets={resets[0]} | "
      f"mean|cte|={statistics.mean(absc):.3f} rms={rms:.3f} max={max(absc):.3f} "
      f"std={statistics.pstdev(c):.3f} zc={100*zc/len(c):.1f}% spd={statistics.mean(sp):.2f}")
