"""RLPP environment: Residual RL + Pure Pursuit on F1TENTH.

Implements the environment from:
  Ghignone et al., "RLPP: A Residual Method for Zero-Shot Real-World
  Autonomous Racing on Scaled Platforms," arXiv 2501.17311v2, 2025.

Uses the vendored F1TENTH simulator (single-track dynamic model with
Pacejka tire forces, RK4 integration, laser-based collision detection).
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add vendored simulator to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.join(_SCRIPT_DIR, "rpl4f110", "simulator")
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from f110_gym.envs.base_classes import Simulator, Integrator
from pure_pursuit import PurePursuitController


# Vehicle params from Table II of the paper
VEHICLE_PARAMS = {
    "mu": 0.5,
    "C_Sf": 4.718,
    "C_Sr": 5.4562,
    "lf": 0.174,
    "lr": 0.151,
    "h": 0.074,
    "m": 3.56,
    "I": 0.0627,
    "s_min": -0.4189,
    "s_max": 0.4189,
    "sv_min": -3.2,
    "sv_max": 3.2,
    "v_switch": 7.319,
    "a_max": 9.51,
    "v_min": -5.0,
    "v_max": 8.0,
    "length": 0.58,
    "width": 0.31,
}

# Reward params from Table III
REWARD_PARAMS = {
    "alpha_dev": 1.0,
    "tau_dev": 0.1,
    "alpha_heading": 0.25,
    "tau_psi": 0.0,
}


def load_raceline(track_dir, track_name):
    path = os.path.join(track_dir, f"{track_name}_raceline.csv")
    data = np.loadtxt(path, delimiter=";", skiprows=3)
    return data


def load_centerline(track_dir, track_name):
    path = os.path.join(track_dir, f"{track_name}_centerline.csv")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data


class RLPPEnv(gym.Env):
    """RLPP residual racing environment.

    Observation (Eq. 18): [d, delta_psi, vx, vy, r, o_traj]
      where o_traj is 6*N dims (ref + left/right boundary in local frame).

    Action: normalized residual [steering, velocity] in [-1, 1],
      scaled by action_scaling and added to PP base action.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        track_name: str = "Spielberg",
        tracks_dir: str | None = None,
        alpha_rl: float = 1.0,
        n_waypoints: int = 20,
        sim_dt: float = 0.01,
        controller_dt: float = 0.01,
        lookahead_distance: float = 1.2,
        velocity_gain: float = 0.75,
        action_scaling: tuple[float, float] = (0.05, 1.0),
        mu_noise_std: float = 0.15,
        velocity_curriculum: bool = True,
        max_laps: int = 2,
        render_mode: str | None = None,
    ):
        super().__init__()

        if tracks_dir is None:
            tracks_dir = os.path.join(_SCRIPT_DIR, "f1tenth_racetracks")
        track_dir = os.path.join(tracks_dir, track_name)

        self.track_name = track_name
        self.track_dir = track_dir
        self.alpha_rl = alpha_rl
        self.n_waypoints = n_waypoints
        self.sim_dt = sim_dt
        self.controller_dt = controller_dt
        self.control_to_sim_ratio = max(1, int(controller_dt / sim_dt))
        self.action_scaling = np.array(action_scaling, dtype=np.float32)
        self.mu_noise_std = mu_noise_std
        self.velocity_curriculum = velocity_curriculum
        self.max_laps = max_laps

        self.raceline = load_raceline(track_dir, track_name)
        self.centerline = load_centerline(track_dir, track_name)

        self.raceline_s = self.raceline[:, 0]
        self.total_s = self.raceline_s[-1]

        self.params = dict(VEHICLE_PARAMS)
        wheelbase = self.params["lf"] + self.params["lr"]

        self.pp = PurePursuitController(
            self.raceline,
            self.centerline,
            lookahead_distance=lookahead_distance,
            wheelbase=wheelbase,
            velocity_gain=velocity_gain,
        )

        map_path = os.path.join(track_dir, f"{track_name}_map")
        map_ext = ".png"

        self.sim = Simulator(
            self.params, num_agents=1, seed=12345,
            time_step=sim_dt, integrator=Integrator.RK4,
        )
        self.sim.set_map(map_path + ".yaml", map_ext)

        self.vmax = self.params["v_max"]
        self.psi_max = np.pi

        obs_dim = 5 + 6 * n_waypoints
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._last_pos = np.zeros(2)
        self._last_heading = 0.0
        self._closest_idx = 0
        self._prev_s = 0.0
        self._cumulative_progress = 0.0
        self._lap_count = 0
        self._avg_velocity = 0.5
        self._episode_velocities = []

    def _randomize_friction(self):
        noise = self.np_random.normal(0.0, self.mu_noise_std)
        mu = VEHICLE_PARAMS["mu"] + noise
        mu = max(0.1, mu)
        params = dict(self.params)
        params["mu"] = mu
        self.sim.update_params(params)

    def _extract_state(self, obs):
        px = float(obs["poses_x"][0])
        py = float(obs["poses_y"][0])
        theta = float(obs["poses_theta"][0])
        vx = float(obs["linear_vels_x"][0])
        vy = float(obs["linear_vels_y"][0])
        yaw_rate = float(obs["ang_vels_z"][0])
        collision = bool(obs["collisions"][0])
        return px, py, theta, vx, vy, yaw_rate, collision

    def _build_obs(self, px, py, theta, vx, vy, yaw_rate, closest_idx):
        d, delta_psi = self.pp.get_frenet_state(px, py, theta, closest_idx)
        o_traj = self.pp.get_trajectory_obs(px, py, theta, closest_idx, self.n_waypoints)

        return np.concatenate(
            [np.array([d, delta_psi, vx, vy, yaw_rate], dtype=np.float32), o_traj]
        )

    def _compute_reward(self, px, py, theta, vx, closest_idx, collision):
        current_s = self.raceline_s[closest_idx]
        prev_s = self._prev_s
        delta_s = current_s - prev_s
        if delta_s < -self.total_s / 2:
            delta_s += self.total_s
        elif delta_s > self.total_s / 2:
            delta_s -= self.total_s

        self._cumulative_progress += delta_s
        if self._cumulative_progress >= self.total_s:
            self._cumulative_progress -= self.total_s
            self._lap_count += 1

        r_adv = delta_s / (self.vmax * self.sim_dt)
        r_speed = vx / self.vmax
        r_pos = r_adv + r_speed

        d, delta_psi = self.pp.get_frenet_state(px, py, theta, closest_idx)
        w_track = self.pp.track_width[closest_idx]

        r_dev = 0.0
        abs_d = abs(d)
        if abs_d > REWARD_PARAMS["tau_dev"]:
            r_dev = -REWARD_PARAMS["alpha_dev"] * abs_d / max(w_track, 0.1)

        r_heading = 0.0
        abs_dpsi = abs(delta_psi)
        if abs_dpsi > REWARD_PARAMS["tau_psi"]:
            r_heading = -REWARD_PARAMS["alpha_heading"] * abs_dpsi / self.psi_max

        r_coll = -1.0 if collision else 0.0
        r_tot = r_pos + r_pos * (r_dev + r_heading) + r_coll
        return float(r_tot)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._randomize_friction()

        if self.velocity_curriculum and len(self._episode_velocities) > 0:
            self._avg_velocity = float(np.mean(self._episode_velocities))

        start_idx = self.np_random.integers(0, len(self.raceline))
        start_x = self.raceline[start_idx, 1]
        start_y = self.raceline[start_idx, 2]
        start_psi = self.raceline[start_idx, 3]

        poses = np.array([[start_x, start_y, start_psi]])
        self.sim.reset(poses)

        if self.velocity_curriculum:
            v_init = self.np_random.normal(self._avg_velocity, 0.5)
            v_init = float(np.clip(v_init, 0.0, self.vmax))
            self.sim.agents[0].state[3] = v_init
            action = np.array([[0.0, v_init]])
        else:
            action = np.zeros((1, 2))
        obs_dict = self.sim.step(action)
        px, py, theta, vx, vy, yaw_rate, collision = self._extract_state(obs_dict)

        self._closest_idx = self.pp.find_closest_index(np.array([px, py]))
        self._prev_s = self.raceline_s[self._closest_idx]
        self._cumulative_progress = 0.0
        self._lap_count = 0
        self._last_pos = np.array([px, py])
        self._last_heading = theta
        self._episode_velocities = []

        return self._build_obs(px, py, theta, vx, vy, yaw_rate, self._closest_idx), {}

    def step(self, action: np.ndarray):
        pp_action, _ = self.pp.get_action(self._last_pos, self._last_heading)

        residual = action * self.action_scaling * self.alpha_rl
        combined = pp_action + residual
        combined[0] = np.clip(combined[0], self.params["s_min"], self.params["s_max"])
        combined[1] = np.clip(combined[1], 0.0, self.vmax)

        sim_action = np.array([[combined[0], combined[1]]])

        collision = False
        for _ in range(self.control_to_sim_ratio):
            obs_dict = self.sim.step(sim_action)
            px, py, theta, vx, vy, yaw_rate, col = self._extract_state(obs_dict)
            if col:
                collision = True
                break

        self._closest_idx = self.pp.find_closest_index(np.array([px, py]))
        reward = self._compute_reward(px, py, theta, vx, self._closest_idx, collision)

        self._last_pos = np.array([px, py])
        self._last_heading = theta
        self._prev_s = self.raceline_s[self._closest_idx]
        self._episode_velocities.append(vx)

        rl_obs = self._build_obs(px, py, theta, vx, vy, yaw_rate, self._closest_idx)

        terminated = collision
        truncated = self._lap_count >= self.max_laps
        info = {
            "collision": collision,
            "vx": vx,
            "closest_idx": self._closest_idx,
            "lap_count": self._lap_count,
        }

        return rl_obs, reward, terminated, truncated, info
