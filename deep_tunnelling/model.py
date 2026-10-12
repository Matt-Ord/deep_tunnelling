"""Hop rates of an adsorbate from the UME, for a set of temperatures and frictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from deep_tunnelling.ume import BandData, hop_rates

Array = np.ndarray[Any, Any]

# bands in the master equation: all bands whose mean energy lies within this
# window above the ground state, i.e. every band below the ~143 meV barrier.
# H: 11 bands, D: 17 bands (D's first vibrational band barely tunnels, so its
# activated hopping runs through the higher bands).
ENERGY_WINDOW = 150.0  # meV


@dataclass(frozen=True, kw_only=True)
class RateCurve:
    """Slow (out of fcc) and fast (out of hcp) escape rates (ps^-1) at one friction."""

    mass: float
    gamma: float  # ps^-1
    temperatures: Array  # K
    slow: Array
    fast: Array

    @property
    def eta(self) -> float:
        """Friction coefficient eta = m gamma, independent of the adsorbate mass for e-h friction."""
        return self.mass * self.gamma


def n_rate_for(bands: BandData, window: float = ENERGY_WINDOW) -> int:
    """Count the bands whose mean energy lies within window of the ground band."""
    mean = bands.energies.mean(axis=0)
    return int(np.sum(mean - mean[0] < window))


def rates(
    bands: BandData, mass: float, temperatures: Array, gammas: list[float]
) -> list[RateCurve]:
    """Escape-rate curves, one per friction. Slow and fast are equal for degenerate hollows."""
    n_rate = n_rate_for(bands)
    results = [
        hop_rates(bands, mass, t, gammas, n_rate) for t in temperatures
    ]  # (T, gamma)
    return [
        RateCurve(
            mass=mass,
            gamma=gamma,
            temperatures=np.asarray(temperatures),
            slow=np.array([row[j].fcc_escape_rate for row in results]),
            fast=np.array([row[j].hcp_escape_rate for row in results]),
        )
        for j, gamma in enumerate(gammas)
    ]
