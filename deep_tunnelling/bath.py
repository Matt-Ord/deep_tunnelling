"""Ohmic bath functions for the universal (UME) master equation, without a cutoff.

J(w) = eta w,  eta = m gamma
S(w) = 2 hbar J(w) / (1 - exp(-hbar w / kT))
Gamma(nu) = S(nu) / 2 + i P(nu)

The principal value P(nu) = (1 / 2 pi) P int S(w) / (nu - w) dw diverges for an
Ohmic bath, but only through (i) a constant, cancelled exactly by the
Caldeira-Leggett counterterm (thesis Eq. 7.77), and (ii) a term linear in nu,
which multiplies the double commutator [A^dag, [H_s, A]] = -hbar^2 / m, a
c-number for both the linear and the periodic couplings. Dropping both (thesis
Eq. 7.78) leaves the cutoff-free Lamb shift

    P(nu) = 2 eta kT x [Re psi(1 + i x) + gamma_E],  x = hbar nu / 2 pi kT,

from the Matsubara expansion of coth; P ~ (hbar eta nu / pi) ln(hbar nu / 2 pi kT)
at large nu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.special

from deep_tunnelling.units import HBAR, KB

Array = np.ndarray[Any, Any]


@dataclass(frozen=True)
class OhmicBath:
    """Ohmic bath. Units: meV, ps, A."""

    mass: float
    gamma: float  # ps^-1
    temperature: float  # K

    @property
    def eta(self) -> float:
        """Friction coefficient eta = m gamma (meV ps / A^2)."""
        return self.mass * self.gamma

    @property
    def kt(self) -> float:
        """Thermal energy (meV)."""
        return KB * self.temperature

    def s(self, w: Array) -> Array:
        """Bath spectrum S(w); S(0) = 2 eta kT, vanishes at large negative w."""
        w = np.asarray(w, dtype=float)
        x = HBAR * w / self.kt
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            out = 2 * HBAR * self.eta * w / (-np.expm1(-x))
        out = np.where(np.abs(x) < 1e-8, 2 * self.eta * self.kt, out)  # noqa: PLR2004
        return np.where(np.isfinite(out), out, 0.0)

    def principal_value(self, nu: Array) -> Array:
        """Cutoff-free principal value (constant and c-number linear term removed)."""
        x = HBAR * np.asarray(nu, dtype=float) / (2 * np.pi * self.kt)
        return (
            2
            * self.eta
            * self.kt
            * x
            * (scipy.special.digamma(1 + 1j * x).real + np.euler_gamma)
        )

    def gamma_function(self, nu: Array) -> Array:
        """Gamma(nu) = int_0^inf D(tau) e^{i nu tau} dtau, regularised as above."""
        nu = np.asarray(nu, dtype=float)
        return 0.5 * self.s(nu) + 1j * self.principal_value(nu)
