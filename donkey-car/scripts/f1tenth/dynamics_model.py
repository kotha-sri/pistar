"""Learned dynamics model ensemble for MBPO.

Ensemble of probabilistic MLPs that predict:
  (next_obs, reward, done) given (obs, action)

Each model outputs a Gaussian distribution over (delta_obs, reward)
and a Bernoulli for termination. The ensemble provides epistemic
uncertainty estimates for principled rollout truncation.
"""

import torch
import torch.nn as nn
import numpy as np


class ProbabilisticMLP(nn.Module):
    """Single dynamics model predicting Gaussian (delta_obs, reward) + done."""

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        layers = []
        in_dim = obs_dim + act_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.SiLU()])
            in_dim = h

        self.trunk = nn.Sequential(*layers)
        # Mean and log_var for delta_obs + reward
        out_dim = obs_dim + 1  # delta_obs + reward
        self.mean_head = nn.Linear(in_dim, out_dim)
        self.logvar_head = nn.Linear(in_dim, out_dim)
        # Termination logit
        self.done_head = nn.Linear(in_dim, 1)

        self._max_logvar = nn.Parameter(torch.ones(out_dim) * 0.5)
        self._min_logvar = nn.Parameter(torch.ones(out_dim) * -10.0)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)

        # Soft clamp log variance
        logvar = self._max_logvar - nn.functional.softplus(self._max_logvar - logvar)
        logvar = self._min_logvar + nn.functional.softplus(logvar - self._min_logvar)

        done_logit = self.done_head(h)
        return mean, logvar, done_logit

    def predict(self, obs, action, deterministic=False):
        """Predict next_obs, reward, done."""
        mean, logvar, done_logit = self.forward(obs, action)
        if deterministic:
            delta_obs_reward = mean
        else:
            std = torch.exp(0.5 * logvar)
            delta_obs_reward = mean + std * torch.randn_like(std)

        delta_obs = delta_obs_reward[:, :-1]
        reward = delta_obs_reward[:, -1]
        next_obs = obs + delta_obs
        done = torch.sigmoid(done_logit.squeeze(-1)) > 0.5

        return next_obs, reward, done


class DynamicsEnsemble(nn.Module):
    """Ensemble of probabilistic dynamics models."""

    def __init__(self, obs_dim, act_dim, n_models=5, hidden_dims=(256, 256, 256),
                 lr=1e-3, device="cuda"):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_models = n_models
        self.device = device

        self.models = nn.ModuleList([
            ProbabilisticMLP(obs_dim, act_dim, hidden_dims)
            for _ in range(n_models)
        ])
        self.to(device)

        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
            for m in self.models
        ]

        self._trained = False

    def train_on_batch(self, obs, action, next_obs, reward, done, n_epochs=5):
        """Train all models on a batch of real transitions."""
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.FloatTensor(action).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        reward_t = torch.FloatTensor(reward).to(self.device)
        done_t = torch.FloatTensor(done.astype(np.float32)).to(self.device)

        delta_obs_t = next_obs_t - obs_t
        target = torch.cat([delta_obs_t, reward_t.unsqueeze(-1)], dim=-1)

        losses = []
        for epoch in range(n_epochs):
            epoch_losses = []
            for i, (model, opt) in enumerate(zip(self.models, self.optimizers)):
                mean, logvar, done_logit = model(obs_t, act_t)

                # Gaussian NLL for (delta_obs, reward)
                inv_var = torch.exp(-logvar)
                mse = (mean - target) ** 2
                nll = (mse * inv_var + logvar).mean()

                # BCE for termination
                bce = nn.functional.binary_cross_entropy_with_logits(
                    done_logit.squeeze(-1), done_t
                )

                # Bounded variance regularization
                var_reg = 0.01 * (self.models[i]._max_logvar.sum() -
                                  self.models[i]._min_logvar.sum())

                loss = nll + bce + var_reg

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
                epoch_losses.append(loss.item())
            losses.append(np.mean(epoch_losses))

        self._trained = True
        return losses

    @torch.no_grad()
    def predict(self, obs, action, deterministic=False):
        """Predict using a randomly selected model from the ensemble."""
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.FloatTensor(action).to(self.device)

        model_idx = np.random.randint(self.n_models)
        next_obs, reward, done = self.models[model_idx].predict(
            obs_t, act_t, deterministic=deterministic
        )
        return (
            next_obs.cpu().numpy(),
            reward.cpu().numpy(),
            done.cpu().numpy(),
        )

    @torch.no_grad()
    def predict_with_disagreement(self, obs, action):
        """Predict using ensemble mean and return per-sample disagreement.

        Returns (next_obs, reward, done, disagreement) where disagreement
        is the mean std across ensemble predictions per sample — used for
        pessimistic reward penalization.
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.FloatTensor(action).to(self.device)

        all_next_obs, all_rewards, all_dones = [], [], []
        for model in self.models:
            next_obs, reward, done = model.predict(obs_t, act_t, deterministic=True)
            all_next_obs.append(next_obs)
            all_rewards.append(reward)
            all_dones.append(done.float())

        next_obs_stack = torch.stack(all_next_obs)
        reward_stack = torch.stack(all_rewards)
        done_stack = torch.stack(all_dones)

        mean_next_obs = next_obs_stack.mean(dim=0)
        mean_reward = reward_stack.mean(dim=0)
        mean_done = done_stack.mean(dim=0) > 0.5

        obs_disagreement = next_obs_stack.std(dim=0).mean(dim=-1)
        reward_disagreement = reward_stack.std(dim=0)
        disagreement = obs_disagreement + reward_disagreement

        return (
            mean_next_obs.cpu().numpy(),
            mean_reward.cpu().numpy(),
            mean_done.cpu().numpy(),
            disagreement.cpu().numpy(),
        )

    @torch.no_grad()
    def predict_ensemble(self, obs, action):
        """Predict with all models, return mean + disagreement."""
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.FloatTensor(action).to(self.device)

        means = []
        for model in self.models:
            mean, _, _ = model(obs_t, act_t)
            means.append(mean)

        means = torch.stack(means)
        ensemble_mean = means.mean(dim=0)
        disagreement = means.std(dim=0).mean(dim=-1)

        return ensemble_mean.cpu().numpy(), disagreement.cpu().numpy()

    def generate_synthetic_rollouts(self, policy, start_obs, horizon=5,
                                    n_rollouts=None):
        """Generate synthetic rollouts from real starting states.

        Args:
            policy: SB3-compatible policy with predict() method
            start_obs: (N, obs_dim) real observations to start from
            horizon: number of synthetic steps per rollout
            n_rollouts: if set, randomly sample this many start states

        Returns:
            dict with obs, action, next_obs, reward, done arrays
        """
        if n_rollouts is not None and n_rollouts < len(start_obs):
            indices = np.random.choice(len(start_obs), n_rollouts, replace=False)
            start_obs = start_obs[indices]

        all_obs, all_act, all_next, all_rew, all_done = [], [], [], [], []

        obs = start_obs.copy()
        for step in range(horizon):
            action, _ = policy.predict(obs, deterministic=False)
            next_obs, reward, done = self.predict(obs, action)

            all_obs.append(obs)
            all_act.append(action)
            all_next.append(next_obs)
            all_rew.append(reward)
            all_done.append(done)

            # Mask terminated trajectories
            alive = ~done
            if not alive.any():
                break
            obs = next_obs
            # Reset terminated ones to random start states
            dead_mask = done
            if dead_mask.any():
                replace_idx = np.random.choice(
                    len(start_obs), dead_mask.sum(), replace=True
                )
                obs[dead_mask] = start_obs[replace_idx]

        return {
            "obs": np.concatenate(all_obs, axis=0),
            "action": np.concatenate(all_act, axis=0),
            "next_obs": np.concatenate(all_next, axis=0),
            "reward": np.concatenate(all_rew, axis=0),
            "done": np.concatenate(all_done, axis=0),
        }

    def save(self, path):
        torch.save({
            "models": self.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "n_models": self.n_models,
        }, path)

    @classmethod
    def load(cls, path, device="cuda"):
        data = torch.load(path, map_location=device, weights_only=False)
        ensemble = cls(
            data["obs_dim"], data["act_dim"],
            n_models=data["n_models"], device=device
        )
        ensemble.load_state_dict(data["models"])
        ensemble._trained = True
        return ensemble
