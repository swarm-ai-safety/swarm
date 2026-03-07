"""Deterministic seeding utilities for reproducible runs."""

from __future__ import annotations

import hashlib
import random

import numpy as np


def make_rng(seed: int) -> random.Random:
    """Create a deterministic random.Random instance."""
    return random.Random(seed)


def make_np_rng(seed: int) -> np.random.Generator:
    """Create a deterministic numpy Generator."""
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, label: str) -> int:
    """Derive a sub-seed from a base seed and label.

    Useful for giving each component (agents, governance, environment)
    its own independent RNG stream from a single master seed.
    """
    h = hashlib.sha256(f"{base_seed}:{label}".encode()).hexdigest()
    return int(h[:8], 16)


def seed_all(seed: int) -> None:
    """Seed stdlib random and numpy global RNG for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


class SeededRNG:
    """Wrapper providing both stdlib and numpy RNGs from a single seed."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = make_rng(seed)
        self.np_rng = make_np_rng(seed)

    def derive(self, label: str) -> "SeededRNG":
        """Derive a child RNG with a deterministic sub-seed."""
        return SeededRNG(derive_seed(self.seed, label))

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return self.rng.uniform(low, high)

    def randint(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)

    def choice(self, seq: list):
        return self.rng.choice(seq)

    def shuffle(self, seq: list) -> None:
        self.rng.shuffle(seq)

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self.rng.gauss(mu, sigma)
