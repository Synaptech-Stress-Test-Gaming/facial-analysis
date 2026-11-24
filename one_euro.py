"""1 Euro filter: smoothing for live input that hides jitter without adding lag.

Casiez, Roussel & Vogel, "1 Euro Filter: A Simple Speed-based Low-pass Filter for Noisy
Input in Interactive Systems", CHI 2012. The cutoff frequency rises with the speed of
the signal: while it holds still it is smoothed strongly (no jitter), while it moves
fast it is barely smoothed (no lag).
"""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """min_cutoff (Hz): smoothing at rest, lower = steadier. beta: how quickly the cutoff
    rises with speed (per unit of the signal per second), higher = less lag when moving."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.004, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t = self._x = None
        self._dx = 0.0

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))

    def __call__(self, t: float, x):
        """Filters a number or a numpy array; returns the same kind."""
        x = np.asarray(x, dtype=float)
        if self._t is None or t - self._t > 0.5:  # first value, or after a long gap: start over
            self._t, self._x, self._dx = t, x, np.zeros_like(x)
            return x if x.ndim else float(x)
        dt = max(t - self._t, 1e-3)
        a = self._alpha(dt, self.d_cutoff)
        self._dx = a * (x - self._x) / dt + (1 - a) * self._dx
        a = self._alpha(dt, self.min_cutoff + self.beta * float(np.linalg.norm(self._dx)))
        self._t, self._x = t, a * x + (1 - a) * self._x
        return self._x if self._x.ndim else float(self._x)
