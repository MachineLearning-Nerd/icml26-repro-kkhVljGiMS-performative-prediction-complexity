"""Shared concepts for performative prediction complexity reproduction.

Implements the core definitions from the paper (Section 2 / Preliminaries):
  - performative stability (Definition 2.2)
  - first-order epsilon-performative stability (Definition 2.4)
  - repeated risk minimization (RRM) map (Definition 2.1)
  - the hard class of problems (Section 3, Eq. 2-3)
  - variational inequality (VI) solution
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class PerformativeInstance:
    """A performative prediction instance of the hard class (Eq. 2-3).

    loss: ell(x; z) = 0.5 * ||x||^2 - x^T z   (1-strongly convex, 1-smooth)
    distribution shift: z = g(x),  g is L-Lipschitz
    constraint set: X (default [0,1]^d)
    So rho = L * beta / alpha = L * 1 / 1 = L.
    """

    g: Callable[[np.ndarray], np.ndarray]
    X_lo: np.ndarray
    X_hi: np.ndarray
    L: float  # Lipschitz constant of g => rho = L (since alpha=beta=1)

    @property
    def d(self) -> int:
        return len(self.X_lo)

    @property
    def rho(self) -> float:
        return self.L  # alpha = beta = 1 for this class

    def rrm_map(self, x: np.ndarray) -> np.ndarray:
        """G(x) = argmin_{x' in X} E_{z~D(x)}[ell(x';z)] = argmin 0.5||x'||^2 - x'^T g(x).

        Unconstrained minimizer is g(x); project onto X = [lo, hi].
        """
        z = self.g(x)
        return np.clip(z, self.X_lo, self.X_hi)

    def is_epsilon_stable(self, xstar: np.ndarray, eps: float, n_samples: int = 0) -> bool:
        """Check first-order epsilon-performative stability (Def 2.4).

        <x - x*, E_{z~D(x*)}[nabla ell(x*; z)]> >= -eps  for all x in X.

        For ell(x;z) = 0.5||x||^2 - x^T z: nabla_x ell(x;z) = x - z.
        So E[nabla ell(x*; z)] = x* - g(x*).
        Condition: <x - x*, x* - g(x*)> >= -eps for all x in X.
        """
        grad = xstar - self.g(xstar)  # expected gradient at x*
        # The worst-case x maximizes <x* - x, grad> negativity => picks the corner
        # opposite to grad sign. <x - x*, x* - g(x*)> = <x, grad> - <x*, grad>.
        # Minimize over x in X: x_i = lo_i if grad_i>0 else hi_i.
        worst_x = np.where(grad > 0, self.X_lo, self.X_hi).astype(float)
        val = np.dot(worst_x - xstar, grad)
        return val >= -eps - 1e-12

    def stability_gap(self, xstar: np.ndarray) -> float:
        """Return the first-order stability violation: max over x of -<x-x*, grad>.

        Positive => NOT stable; <=0 => stable (gap 0 = exactly stable).
        """
        grad = xstar - self.g(xstar)
        worst_x = np.where(grad > 0, self.X_lo, self.X_hi).astype(float)
        return -np.dot(worst_x - xstar, grad)


def vi_residual(A: np.ndarray, b: np.ndarray, x: np.ndarray,
                X_lo: np.ndarray | None = None, X_hi: np.ndarray | None = None) -> float:
    """Compute the VI residual: max_{x' in X} -<x' - x, Ax + b>.

    Returns the violation (0 = exact solution). For X = [0,1]^d by default.
    """
    d = len(x)
    if X_lo is None:
        X_lo = np.zeros(d)
    if X_hi is None:
        X_hi = np.ones(d)
    Fx = A @ x + b
    worst_x = np.where(Fx > 0, X_hi, X_lo).astype(float)
    return -np.dot(worst_x - x, Fx)
