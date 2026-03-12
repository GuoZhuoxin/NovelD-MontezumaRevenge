"""
Episode-level visit counter.

Uses SimHash to discretize continuous pixel observations into hashable keys.
"""

import numpy as np
from collections import defaultdict


class SimHashCounter:
    """
    SimHash-based state counter for a single environment.
    Projects high-dim obs to a low-dim binary vector, then uses as hash key.
    """
    def __init__(self, obs_shape=(1, 84, 84), hash_dim=64, seed=42):
        rng = np.random.RandomState(seed)
        # Random projection matrix: (hash_dim, obs_flat_dim)
        flat_dim = int(np.prod(obs_shape))
        self.A = rng.randn(hash_dim, flat_dim).astype(np.float32)
        self.counts: dict = defaultdict(int)

    def _hash(self, obs: np.ndarray) -> tuple:
        """Project obs to binary hash key."""
        flat = obs.flatten()
        proj = self.A @ flat          # (hash_dim,)
        bits = (proj > 0).astype(np.uint8)
        return tuple(bits.tolist())

    def visit(self, obs: np.ndarray) -> int:
        """
        Record a visit to state obs.
        Returns the visit count AFTER incrementing (1 = first visit).
        """
        key = self._hash(obs)
        self.counts[key] += 1
        return self.counts[key]

    def reset(self):
        """Call at the start of each episode."""
        self.counts.clear()


class EpisodeCounter:
    """
    Manages one SimHashCounter per parallel environment.
    """
    def __init__(self, n_envs: int, obs_shape=(1, 84, 84), hash_dim=64):
        self.counters = [
            SimHashCounter(obs_shape, hash_dim, seed=i)
            for i in range(n_envs)
        ]

    def visit(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        Args:
            obs_batch: (n_envs, 1, 84, 84)

        Returns:
            counts: (n_envs,) int array — 1 means first visit this episode
        """
        counts = np.array([
            self.counters[i].visit(obs_batch[i])
            for i in range(len(self.counters))
        ])
        return counts

    def reset(self, env_indices=None):
        """
        Reset counters for specified envs (or all if None).
        Call when an episode ends.
        """
        if env_indices is None:
            env_indices = range(len(self.counters))
        for i in env_indices:
            self.counters[i].reset()
