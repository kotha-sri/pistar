import numpy as np
from numba import njit


@njit(cache=True)
def _find_nearest(position, waypoints_xy):
    diffs = waypoints_xy - position
    dists_sq = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    return np.argmin(dists_sq)


@njit(cache=True)
def _find_lookahead(position, waypoints_xy, cumul_dist, closest_idx, dla):
    total_len = cumul_dist[-1]
    start_d = cumul_dist[closest_idx]
    n = len(waypoints_xy)
    for i in range(1, n):
        idx = (closest_idx + i) % n
        traveled = cumul_dist[idx] - start_d
        if traveled < 0:
            traveled += total_len
        if traveled >= dla:
            return idx
    return (closest_idx + 1) % n


class PurePursuitController:
    def __init__(
        self,
        raceline: np.ndarray,
        centerline: np.ndarray,
        lookahead_distance: float = 1.2,
        wheelbase: float = 0.325,
        velocity_gain: float = 0.75,
    ):
        """
        Args:
            raceline: (N, 7) array — columns: s, x, y, psi, kappa, vx, ax
            centerline: (M, 4) array — columns: x, y, w_right, w_left
            lookahead_distance: PP lookahead in meters
            wheelbase: lf + lr
            velocity_gain: alpha_v scaling on reference velocity
        """
        self.raceline = np.ascontiguousarray(raceline)
        self.centerline = np.ascontiguousarray(centerline)
        self.wpts_xy = np.ascontiguousarray(raceline[:, 1:3])
        self.dla = lookahead_distance
        self.lwb = wheelbase
        self.alpha_v = velocity_gain

        diffs = np.diff(self.wpts_xy, axis=0)
        seg_lens = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        self.cumul_dist = np.concatenate([np.array([0.0]), np.cumsum(seg_lens)])

        self._build_centerline_kdtree()
        self._build_frenet_data()

    def _build_centerline_kdtree(self):
        self.cl_xy = np.ascontiguousarray(self.centerline[:, :2])

    def _build_frenet_data(self):
        """Precompute reference heading and track width at each raceline point."""
        self.ref_psi = self.raceline[:, 3]
        self.ref_vx = self.raceline[:, 5]

        n_rl = len(self.raceline)
        n_cl = len(self.centerline)
        self.track_width = np.zeros(n_rl)
        for i in range(n_rl):
            pt = self.wpts_xy[i]
            cl_idx = _find_nearest(pt, self.cl_xy)
            w_r = self.centerline[cl_idx, 2]
            w_l = self.centerline[cl_idx, 3]
            self.track_width[i] = w_r + w_l

    def find_closest_index(self, position: np.ndarray) -> int:
        return int(_find_nearest(position, self.wpts_xy))

    def get_frenet_state(self, px, py, theta, closest_idx):
        ref_pos = self.wpts_xy[closest_idx]
        ref_psi = self.ref_psi[closest_idx]

        error_vec = np.array([px - ref_pos[0], py - ref_pos[1]])
        d = -np.sin(ref_psi) * error_vec[0] + np.cos(ref_psi) * error_vec[1]
        delta_psi = np.arctan2(np.sin(theta - ref_psi), np.cos(theta - ref_psi))
        return d, delta_psi

    def get_action(self, position: np.ndarray, heading: float):
        closest_idx = self.find_closest_index(position)
        la_idx = _find_lookahead(position, self.wpts_xy, self.cumul_dist, closest_idx, self.dla)
        la_point = self.wpts_xy[la_idx]

        dx = la_point[0] - position[0]
        dy = la_point[1] - position[1]
        local_y = -np.sin(heading) * dx + np.cos(heading) * dy

        steering = np.arctan2(2.0 * self.lwb * local_y, self.dla ** 2)
        velocity = self.ref_vx[la_idx] * self.alpha_v

        return np.array([steering, velocity], dtype=np.float32), closest_idx

    def get_trajectory_obs(self, px, py, theta, closest_idx, n_points=20):
        """Build the 6N-dim trajectory observation from the paper (Eq. 18).

        For each of N points ahead on the raceline, returns:
          - reference point (x, y) in local frame
          - left boundary point (x, y) in local frame
          - right boundary point (x, y) in local frame
        """
        n_rl = len(self.raceline)
        cos_h = np.cos(-theta)
        sin_h = np.sin(-theta)
        pos = np.array([px, py])

        o_traj = np.zeros(6 * n_points, dtype=np.float32)

        for i in range(n_points):
            idx_rl = (closest_idx + i) % n_rl
            ref_pt = self.wpts_xy[idx_rl]
            ref_psi = self.ref_psi[idx_rl]

            cl_idx = _find_nearest(ref_pt, self.cl_xy)
            w_r = self.centerline[cl_idx, 2]
            w_l = self.centerline[cl_idx, 3]

            normal = np.array([-np.sin(ref_psi), np.cos(ref_psi)])
            left_pt = ref_pt + w_l * normal
            right_pt = ref_pt - w_r * normal

            for j, pt in enumerate([ref_pt, left_pt, right_pt]):
                rel = pt - pos
                local_x = cos_h * rel[0] - sin_h * rel[1]
                local_y = sin_h * rel[0] + cos_h * rel[1]
                o_traj[i * 6 + j * 2] = local_x
                o_traj[i * 6 + j * 2 + 1] = local_y

        return o_traj
