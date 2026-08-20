"""Multi-track RLPP environment that samples a random track each episode.

Supports both pre-existing F1TENTH tracks and procedurally generated ones.
Each reset picks a new track, giving the agent diverse track geometry
experience for generalization.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from residual_env import RLPPEnv
from track_generator import generate_track, save_track


class MultiTrackEnv(gym.Env):
    """Wraps RLPPEnv to switch tracks each episode.

    On each reset, selects either a real track or generates a procedural one.
    The observation space is identical to RLPPEnv — the agent shouldn't
    need to know which track it's on; it should learn general racing.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        real_tracks=None,
        tracks_dir=None,
        generated_tracks_dir=None,
        n_generated=50,
        gen_seed_start=1000,
        prob_generated=0.5,
        pregenerate=True,
        **env_kwargs,
    ):
        """
        Args:
            real_tracks: list of track names to use (None = all available)
            tracks_dir: path to f1tenth_racetracks/
            generated_tracks_dir: where to save/load generated tracks
            n_generated: how many procedural tracks to pre-generate
            gen_seed_start: starting seed for generation
            prob_generated: probability of using a generated track per episode
            pregenerate: whether to generate tracks at init
            **env_kwargs: passed to RLPPEnv (alpha_rl, velocity_gain, etc.)
        """
        super().__init__()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        if tracks_dir is None:
            tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")
        if generated_tracks_dir is None:
            generated_tracks_dir = os.path.join(script_dir, "generated_tracks")

        self.tracks_dir = tracks_dir
        self.generated_tracks_dir = generated_tracks_dir
        self.prob_generated = prob_generated
        self.env_kwargs = env_kwargs
        self._gen_seed_counter = gen_seed_start + n_generated

        # Discover real tracks
        if real_tracks is None:
            real_tracks = self._discover_tracks(tracks_dir)
        self.real_tracks = real_tracks

        # Pre-generate procedural tracks or discover existing ones
        self.generated_tracks = []
        os.makedirs(generated_tracks_dir, exist_ok=True)
        if pregenerate:
            for i in range(n_generated):
                seed = gen_seed_start + i
                name = f"random_{seed}"
                track_dir = os.path.join(generated_tracks_dir, name)
                if not os.path.exists(track_dir):
                    track_data = generate_track(seed=seed)
                    save_track(track_data, generated_tracks_dir)
                self.generated_tracks.append(name)
        else:
            self.generated_tracks = self._discover_tracks(generated_tracks_dir)

        # Create initial env to set observation/action spaces
        self._current_env = None
        self._create_env(self.real_tracks[0], tracks_dir)

        self.observation_space = self._current_env.observation_space
        self.action_space = self._current_env.action_space

        self._episode_count = 0

    def _discover_tracks(self, tracks_dir):
        """Find all tracks with complete data files."""
        tracks = []
        if not os.path.isdir(tracks_dir):
            return tracks
        for name in sorted(os.listdir(tracks_dir)):
            td = os.path.join(tracks_dir, name)
            if not os.path.isdir(td):
                continue
            has_raceline = os.path.exists(os.path.join(td, f"{name}_raceline.csv"))
            has_centerline = os.path.exists(os.path.join(td, f"{name}_centerline.csv"))
            has_map = os.path.exists(os.path.join(td, f"{name}_map.png"))
            if has_raceline and has_centerline and has_map:
                tracks.append(name)
        return tracks

    def _create_env(self, track_name, tracks_dir):
        """Create a new RLPPEnv for the given track."""
        if self._current_env is not None:
            del self._current_env

        self._current_env = RLPPEnv(
            track_name=track_name,
            tracks_dir=tracks_dir,
            **self.env_kwargs,
        )

    def _pick_track(self):
        """Select a track for the next episode."""
        rng = self.np_random if hasattr(self, "np_random") else np.random.default_rng()

        use_generated = (
            len(self.generated_tracks) > 0
            and rng.random() < self.prob_generated
        )

        if use_generated:
            name = rng.choice(self.generated_tracks)
            return name, self.generated_tracks_dir
        else:
            name = rng.choice(self.real_tracks)
            return name, self.tracks_dir

    def generate_new_track(self):
        """Generate a fresh random track and add it to the pool."""
        seed = self._gen_seed_counter
        self._gen_seed_counter += 1
        track_data = generate_track(seed=seed)
        save_track(track_data, self.generated_tracks_dir)
        name = track_data["name"]
        self.generated_tracks.append(name)
        return name

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        track_name, tracks_dir = self._pick_track()
        self._create_env(track_name, tracks_dir)
        self._episode_count += 1
        obs, info = self._current_env.reset(seed=seed)
        info["track_name"] = track_name
        return obs, info

    def step(self, action):
        return self._current_env.step(action)

    @property
    def raceline_s(self):
        return self._current_env.raceline_s

    @property
    def total_s(self):
        return self._current_env.total_s

    @property
    def _closest_idx(self):
        return self._current_env._closest_idx

    @property
    def controller_dt(self):
        return self._current_env.controller_dt

    @property
    def alpha_rl(self):
        return self._current_env.alpha_rl

    @alpha_rl.setter
    def alpha_rl(self, value):
        if self._current_env is not None:
            self._current_env.alpha_rl = value
