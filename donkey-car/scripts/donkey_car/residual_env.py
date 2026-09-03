"""
residual_env.py
---------------
Gymnasium environment for training the residual policy on the Donkey Sim.

The agent's action IS the residual (a 2-vector in [-1, 1] = scaled steer/throttle
nudge). Each step the env:
    1. computes the BASE action from the current telemetry (PathFollower),
    2. combines base + agent residual via ResidualPolicy.combine(),
    3. sends the command to the sim,
    4. waits for the next telemetry packet,
    5. builds the next observation and the reward.

Reward — "fast + smooth like a racer" (after the paper's Eq. 9, adapted):
    r = w_speed · speed                  # go fast
      - w_steer_jerk · (Δsteering)²      # smooth steering inputs
      - w_yaw_jerk · (Δyaw_rate/norm)²   # no weaving — reward steady arcs
      - crash_penalty   if off-track     # stay on the track (also ends episode)

Penalising the *change* in yaw rate (not yaw rate itself) is the key to killing
wobble: a steady corner has near-constant yaw rate and is not penalised, while a
weaving line has rapidly flipping yaw rate and is punished hard.

The base controller keeps the car on the centerline, so the episode rarely
terminates once training gets going; the residual is free to optimise speed.

Observation (8-dim, all from telemetry — no pixels):
    [ cte, cte_rate, speed, yaw_rate, base_steer, base_throttle,
      prev_steer, prev_throttle ]
The base action and previous applied action are included so the residual knows
what it is amending (paper Eq. 8). Curvature-lookahead from the waypoint API can
be added later for anticipation.
"""

from __future__ import annotations

import time
import threading

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from donkey_sim_client import DonkeySimClient
from residual_policy   import ResidualPolicy

OBS_DIM = 8


class ResidualDonkeyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_policy,
        host: str = "127.0.0.1",
        port: int = 9091,
        scene_name: str = "circuit_launch",
        steer_bound: float = 0.10,
        throttle_bound: float = 0.50,
        # reward weights (tuned for "fast + smooth like a racer")
        w_speed: float = 0.30,
        w_steer_jerk: float = 2.0,
        w_yaw_jerk: float = 0.50,
        yaw_norm: float = 100.0,
        crash_penalty: float = 10.0,
        # episode / safety
        off_track_threshold: float = 3.0,
        max_steps: int = 1000,
        step_timeout: float = 1.0,
        scene_load_wait: float = 2.0,
    ):
        super().__init__()

        self.residual = ResidualPolicy(base_policy, steer_bound, throttle_bound)

        self._host = host
        self._port = port
        self.scene_name = scene_name
        self.w_speed = w_speed
        self.w_steer_jerk = w_steer_jerk
        self.w_yaw_jerk = w_yaw_jerk
        self.yaw_norm = yaw_norm
        self.crash_penalty = crash_penalty
        self.off_track_threshold = off_track_threshold
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.scene_load_wait = scene_load_wait

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # telemetry bridge (async receive thread -> sync step)
        self._latest: dict | None = None
        self._latest_t: float = 0.0
        self._telem_event = threading.Event()

        # per-step memory
        self._cur_msg: dict | None = None
        self._cur_base = np.zeros(2, dtype=np.float32)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_cte = 0.0
        self._prev_yaw = 0.0
        self._prev_yaw_rate = 0.0
        self._prev_t = 0.0
        self._steps = 0

        # per-episode racer metrics
        self._ep_n = 0
        self._ep_speed_sum = 0.0
        self._ep_sjerk_sum = 0.0
        self._ep_yjerk_sum = 0.0

        self._connect()

    # ── Sim connection ─────────────────────────────────────────────────────────

    def _connect(self):
        self.client = DonkeySimClient(self._host, self._port)
        self.client.connect()
        self.client.start_listening()
        time.sleep(1.0)                       # let scene_selection_ready arrive
        self.client.load_scene(self.scene_name)
        time.sleep(self.scene_load_wait)      # wait for scene + car to load
        self.client.set_telemetry_callback(self._on_telemetry)

    def _on_telemetry(self, msg: dict):
        self._latest = msg
        self._latest_t = time.time()
        self._telem_event.set()

    def _wait_next_telemetry(self) -> dict:
        self._telem_event.clear()
        if not self._telem_event.wait(timeout=self.step_timeout):
            # timed out — reuse last packet rather than crash
            return self._latest if self._latest is not None else {}
        return self._latest

    # ── Observation / reward ────────────────────────────────────────────────────

    @staticmethod
    def _yaw_delta(yaw: float, prev_yaw: float) -> float:
        """Shortest signed yaw difference in degrees, wrapped to [-180, 180]."""
        d = (yaw - prev_yaw + 180.0) % 360.0 - 180.0
        return d

    def _make_obs(self, msg: dict, prev_action: np.ndarray,
                  cte_rate: float, yaw_rate: float) -> np.ndarray:
        cte   = float(msg.get("cte", 0.0))
        speed = float(msg.get("speed", 0.0))

        base = self.residual.base_action(cte, speed)   # PathFollower(cte, speed)
        self._cur_base = base
        self._cur_msg = msg

        return np.array(
            [cte, cte_rate, speed, yaw_rate, base[0], base[1], prev_action[0], prev_action[1]],
            dtype=np.float32,
        )

    def _reward(self, msg: dict, applied_action: np.ndarray, yaw_rate: float):
        """
        Racer reward (after the paper's Eq. 9, adapted to Donkey telemetry):

            r =  w_speed     · speed              # go fast
               - w_steer_jerk· (Δsteer)²          # smooth steering inputs
               - w_yaw_jerk  · (Δyaw_rate/norm)²  # no weaving — steady arcs
               - crash_penalty   if off-track     # stay on the track (ends episode)

        Penalising yaw *jerk* (change in turn rate) rather than yaw rate itself
        lets the car hold a steady cornering arc without penalty, while heavily
        punishing the rapid back-and-forth that reads as wobble.
        """
        speed = float(msg.get("speed", 0.0))
        cte   = float(msg.get("cte", 0.0))
        hit   = str(msg.get("hit", "none")).lower()
        off_track = abs(cte) > self.off_track_threshold or hit != "none"

        steer_jerk = applied_action[0] - self._prev_action[0]
        yaw_jerk   = (yaw_rate - self._prev_yaw_rate) / self.yaw_norm

        r = (self.w_speed * speed
             - self.w_steer_jerk * (steer_jerk ** 2)
             - self.w_yaw_jerk * (yaw_jerk ** 2))
        if off_track:
            r -= self.crash_penalty
        return r, off_track

    # ── Gym API ─────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.client.send_control(0.0, 0.0)
        time.sleep(0.1)
        self.residual.reset()

        # Reset and VERIFY the car landed back on-track. The sim occasionally
        # spawns it off the (re)generated centerline; without this retry the
        # episode would terminate on step 0 and stall training.
        msg = {}
        for _ in range(6):
            self.client.reset_car()
            time.sleep(0.5)
            msg = self._wait_next_telemetry()
            if abs(float(msg.get("cte", 0.0))) <= self.off_track_threshold:
                break

        # init memory from this packet so the first rate/jerk terms are ~0
        self._prev_cte = float(msg.get("cte", 0.0))
        self._prev_yaw = float(msg.get("yaw", 0.0))
        self._prev_yaw_rate = 0.0
        self._prev_t = time.time()
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._steps = 0
        self._ep_n = 0
        self._ep_speed_sum = self._ep_sjerk_sum = self._ep_yjerk_sum = 0.0

        obs = self._make_obs(msg, self._prev_action, cte_rate=0.0, yaw_rate=0.0)
        self._prev_action = self._cur_base.copy()   # reference jerk against the base
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # combine base (from the telemetry the current obs came from) + residual
        applied = self.residual.combine(self._cur_base, action)

        self._telem_event.clear()
        self.client.send_control(float(applied[0]), float(applied[1]))
        msg = self._wait_next_telemetry()

        # derive kinematic rates once, consistently, for both reward and obs
        now = time.time()
        dt = max(now - self._prev_t, 1e-3)
        cte = float(msg.get("cte", 0.0))
        yaw = float(msg.get("yaw", 0.0))
        cte_rate = (cte - self._prev_cte) / dt
        yaw_rate = self._yaw_delta(yaw, self._prev_yaw) / dt

        reward, off_track = self._reward(msg, applied, yaw_rate)

        self._steps += 1
        terminated = bool(off_track)
        truncated = self._steps >= self.max_steps

        obs = self._make_obs(msg, applied, cte_rate, yaw_rate)

        # accumulate per-episode racer metrics (before advancing memory)
        self._ep_n += 1
        self._ep_speed_sum += float(msg.get("speed", 0.0))
        self._ep_sjerk_sum += (applied[0] - self._prev_action[0]) ** 2
        self._ep_yjerk_sum += ((yaw_rate - self._prev_yaw_rate) / self.yaw_norm) ** 2

        # advance memory
        self._prev_cte = cte
        self._prev_yaw = yaw
        self._prev_yaw_rate = yaw_rate
        self._prev_t = now
        self._prev_action = applied

        info = {"applied": applied, "cte": cte, "speed": float(msg.get("speed", 0.0))}
        if terminated or truncated:
            n = max(self._ep_n, 1)
            info["ep_speed"]      = self._ep_speed_sum / n
            info["ep_steer_jerk"] = (self._ep_sjerk_sum / n) ** 0.5   # RMS
            info["ep_yaw_jerk"]   = (self._ep_yjerk_sum / n) ** 0.5   # RMS
            info["ep_len"]        = self._ep_n
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.client.send_control(0.0, 0.0)
            time.sleep(0.1)
            self.client.disconnect()
        except Exception:
            pass
