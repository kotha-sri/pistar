"""Mixed replay buffer that keeps real and synthetic data separate.

CrossQ's BatchNorm critic is poisoned when synthetic transitions are mixed
into the replay buffer (critic loss explodes from ~1 to 1e9+). This wrapper
maintains two internal buffers and controls the per-batch sampling ratio,
keeping BatchNorm running statistics dominated by real data.
"""

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class MixedReplayBuffer(ReplayBuffer):
    """Replay buffer with separate real and synthetic storage.

    Inherits from ReplayBuffer so CrossQ's type checks pass.
    Real transitions go to the parent buffer via add().
    Synthetic transitions go to a separate internal buffer via add_synthetic().
    sample() draws from both with a controlled ratio.
    """

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="auto",
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        synthetic_buffer_size=None,
        real_fraction=0.8,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        if synthetic_buffer_size is None:
            synthetic_buffer_size = buffer_size

        self._synthetic_buffer = ReplayBuffer(
            buffer_size=synthetic_buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=1,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.real_fraction = real_fraction
        self._synthetic_count = 0

    @property
    def synthetic_size(self):
        if self._synthetic_buffer.full:
            return self._synthetic_buffer.buffer_size
        return self._synthetic_buffer.pos

    def add_synthetic(self, obs, next_obs, action, reward, done, infos):
        self._synthetic_buffer.add(obs, next_obs, action, reward, done, infos)
        self._synthetic_count += len(obs) if obs.ndim > 1 else 1

    def sample_real(self, batch_size, env=None):
        """Sample only from the real data buffer (for dynamics model training)."""
        return super().sample(batch_size, env=env)

    def sample(self, batch_size, env=None):
        syn_size = self.synthetic_size
        real_size = self.buffer_size if self.full else self.pos

        if syn_size == 0 or real_size == 0:
            return super().sample(batch_size, env=env)

        n_real = max(1, int(batch_size * self.real_fraction))
        n_synthetic = batch_size - n_real

        n_synthetic = min(n_synthetic, syn_size)
        n_real = batch_size - n_synthetic

        real_batch = super().sample(n_real, env=env)
        syn_batch = self._synthetic_buffer.sample(n_synthetic, env=env)

        return ReplayBufferSamples(
            observations=torch.cat([real_batch.observations, syn_batch.observations], dim=0),
            actions=torch.cat([real_batch.actions, syn_batch.actions], dim=0),
            next_observations=torch.cat([real_batch.next_observations, syn_batch.next_observations], dim=0),
            dones=torch.cat([real_batch.dones, syn_batch.dones], dim=0),
            rewards=torch.cat([real_batch.rewards, syn_batch.rewards], dim=0),
        )
