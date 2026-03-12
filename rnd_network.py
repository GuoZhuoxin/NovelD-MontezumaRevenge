"""
RND (Random Network Distillation) for novelty estimation.

novel(s) = || predictor(s) - target(s) ||^2

- Target network:    fixed random weights, never updated
- Predictor network: trained to match target output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import RND_OUTPUT_DIM, RND_LR, OBS_NORM_CLIP


class RNDEncoder(nn.Module):
    """
    Shared CNN encoder for both target and predictor.
    Input:  (batch, 1, 84, 84)  — single frame, grayscale
    Output: (batch, 512) feature vector
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # → (32, 20, 20)
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # → (64, 9, 9)
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # → (64, 7, 7)
            nn.LeakyReLU(),
            nn.Flatten(),                                # → 3136
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TargetNetwork(nn.Module):
    """
    Fixed random network. Weights never updated.
    Input:  (batch, 512) encoder features
    Output: (batch, RND_OUTPUT_DIM)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, RND_OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


class PredictorNetwork(nn.Module):
    """
    Trainable network that learns to predict target output.
    Deeper than target to give it enough capacity.
    Input:  (batch, 512) encoder features
    Output: (batch, RND_OUTPUT_DIM)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, RND_OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


class RNDModel:
    """
    Full RND model: encoder + target + predictor.
    Provides novel(s) computation and predictor update.
    """
    def __init__(self, device: torch.device):
        self.device = device

        # Networks
        self.encoder   = RNDEncoder().to(device)
        self.target    = TargetNetwork().to(device)
        self.predictor = PredictorNetwork().to(device)

        # Freeze target completely
        for param in self.target.parameters():
            param.requires_grad = False

        # Only predictor (and encoder via predictor path) are trained
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=RND_LR,
        )

        # Running stats for observation normalization
        self.obs_mean = torch.zeros(1, 1, 84, 84).to(device)
        self.obs_var  = torch.ones(1, 1, 84, 84).to(device)
        self.obs_count = 1e-4

    # ------------------------------------------------------------------
    # Observation normalization (running mean/std)
    # ------------------------------------------------------------------
    def update_obs_stats(self, obs: torch.Tensor):
        """Update running mean/variance with a batch of observations."""
        batch_mean = obs.mean(dim=0, keepdim=True)
        batch_var  = obs.var(dim=0, keepdim=True)
        batch_count = obs.shape[0]

        total = self.obs_count + batch_count
        delta = batch_mean - self.obs_mean
        self.obs_mean  = self.obs_mean + delta * batch_count / total
        self.obs_var   = (
            self.obs_var * self.obs_count + batch_var * batch_count
            + delta ** 2 * self.obs_count * batch_count / total
        ) / total
        self.obs_count = total

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observation and clip."""
        normed = (obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)
        return normed.clamp(-OBS_NORM_CLIP, OBS_NORM_CLIP)

    # ------------------------------------------------------------------
    # Core: compute novelty
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_novelty(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute novel(s) for a batch of observations.

        Args:
            obs: np.ndarray of shape (batch, 1, 84, 84), float32, range [0,1]

        Returns:
            novelty: np.ndarray of shape (batch,)
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        obs_t = self.normalize_obs(obs_t)

        features = self.encoder(obs_t)
        target_feat    = self.target(features)
        predictor_feat = self.predictor(features)

        # MSE per sample: mean over output_dim
        novelty = F.mse_loss(predictor_feat, target_feat, reduction="none")
        novelty = novelty.mean(dim=1)  # (batch,)
        return novelty.cpu().numpy()

    # ------------------------------------------------------------------
    # Update predictor
    # ------------------------------------------------------------------
    def update(self, obs: np.ndarray) -> float:
        """
        Update predictor network on a batch of observations.

        Args:
            obs: np.ndarray of shape (batch, 1, 84, 84), float32, range [0,1]

        Returns:
            loss value (float)
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        self.update_obs_stats(obs_t)
        obs_t = self.normalize_obs(obs_t)

        features = self.encoder(obs_t)

        with torch.no_grad():
            target_feat = self.target(features.detach())

        predictor_feat = self.predictor(features)

        loss = F.mse_loss(predictor_feat, target_feat)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
