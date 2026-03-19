"""
NovelD intrinsic reward computation.

r_int_t = max[ novel(s_{t+1}) - alpha * novel(s_t), 0 ] * I{ N_e(s_{t+1}) = 1 }

Combines:
  - RNDModel      → computes novel(s)
  - EpisodeCounter → computes N_e (episode-level visit count)
  - Running reward normalization
"""

import numpy as np
from rnd_network import RNDModel
from episode_counter import EpisodeCounter
from config import ALPHA, BETA, REWARD_NORM_CLIP
import torch


class RunningMeanStd:
    """Tracks running mean and variance for normalization.

    Uses EMA (decay < 1) so old high-novelty values don't permanently
    suppress current r_int after the RND predictor has learned.
    decay=1.0 reverts to the original infinite-horizon accumulation.
    """
    def __init__(self, decay: float = 0.99):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 1e-4
        self.decay = decay

    def update(self, x: np.ndarray):
        batch_mean  = x.mean()
        batch_var   = x.var()
        batch_count = x.size

        if self.decay < 1.0:
            # EMA: tracks current distribution, forgets old scale
            self.mean = self.decay * self.mean + (1.0 - self.decay) * batch_mean
            self.var  = self.decay * self.var  + (1.0 - self.decay) * batch_var
        else:
            # Original infinite-horizon accumulation
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean  = self.mean + delta * batch_count / total
            self.var   = (
                self.var * self.count + batch_var * batch_count
                + delta ** 2 * self.count * batch_count / total
            ) / total
            self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class NovelDReward:
    """
    Computes NovelD intrinsic rewards for a batch of parallel environments.

    Usage per step:
        r_int = noveld.compute(obs_t, obs_t1, dones)
        noveld.update_rnd(obs_t1)   # update predictor
    """

    def __init__(self, n_envs: int, device: torch.device, reward_norm_clip: float = None):
        self.n_envs  = n_envs
        self.rnd     = RNDModel(device)
        self.counter = EpisodeCounter(n_envs)
        self.reward_rms = RunningMeanStd()
        self._reward_norm_clip = reward_norm_clip if reward_norm_clip is not None else REWARD_NORM_CLIP

        # Cache novelty of s_t to avoid recomputing
        self._prev_novelty = np.zeros(n_envs, dtype=np.float32)

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert stacked-frame obs to single-frame (n_envs, 1, 84, 84) float32 in [0, 1].

        SB3's VecTransposeImage wraps outside our wrapper, so obs inside is
        HWC format: (n_envs, 84, 84, 4). We detect and handle both formats.
        """
        if obs.ndim == 4:
            if obs.shape[-1] in (1, 4):
                # HWC format: (n_envs, 84, 84, 4) — take last channel, then transpose
                obs = obs[:, :, :, -1:]          # → (n_envs, 84, 84, 1)
                obs = obs.transpose(0, 3, 1, 2)  # → (n_envs, 1, 84, 84)
            elif obs.shape[1] == 4:
                # CHW format: (n_envs, 4, 84, 84) — take last frame
                obs = obs[:, -1:, :, :]          # → (n_envs, 1, 84, 84)
        obs = obs.astype(np.float32)
        if obs.max() > 1.0:
            obs = obs / 255.0
        return obs

    def reset(self, env_indices=None):
        """Call at episode start to reset episode counters."""
        self.counter.reset(env_indices)
        if env_indices is None:
            self._prev_novelty[:] = 0.0
        else:
            for i in env_indices:
                self._prev_novelty[i] = 0.0

    def compute(
        self,
        obs_t:  np.ndarray,   # (n_envs, 4, 84, 84) current states
        obs_t1: np.ndarray,   # (n_envs, 4, 84, 84) next states
        dones:  np.ndarray,   # (n_envs,) bool — episode ended?
    ) -> np.ndarray:
        """
        Compute NovelD intrinsic reward for one step.

        Returns:
            r_int: (n_envs,) float32 — scaled intrinsic rewards
        """
        obs_t_proc  = self._preprocess(obs_t)
        obs_t1_proc = self._preprocess(obs_t1)

        # novel(s_t): reuse cached value from previous step
        novel_t  = self._prev_novelty.copy()

        # novel(s_{t+1})
        novel_t1 = self.rnd.compute_novelty(obs_t1_proc)

        # Episode visit mask: I{ N_e(s_{t+1}) = 1 }
        visit_counts = self.counter.visit(obs_t1_proc)
        first_visit_mask = (visit_counts == 1).astype(np.float32)

        # NovelD formula
        raw_r_int = np.maximum(novel_t1 - ALPHA * novel_t, 0.0) * first_visit_mask

        # Normalize intrinsic reward (only update stats on non-zero rewards
        # to avoid zero-padding biasing the running mean/std)
        nonzero_mask = raw_r_int > 0
        nonzero = raw_r_int[nonzero_mask]
        if len(nonzero) > 0:
            self.reward_rms.update(nonzero)
        normalized = self.reward_rms.normalize(raw_r_int)
        normalized = np.where(nonzero_mask, normalized, 0.0)  # zero stays zero
        r_int = np.clip(normalized, 0.0, self._reward_norm_clip)    # r_int always >= 0
        r_int = (BETA * r_int).astype(np.float32)

        # Update cache for next step; reset on episode end
        self._prev_novelty = novel_t1.copy()
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            self.reset(done_indices.tolist())

        return r_int

    def update_rnd(self, obs_t1: np.ndarray) -> float:
        """
        Update RND predictor network.
        Call once per step after compute().

        Returns:
            RND loss (float)
        """
        obs_proc = self._preprocess(obs_t1)
        return self.rnd.update(obs_proc)
