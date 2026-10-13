"""Hop rates of an adsorbate from the UME, for a set of temperatures and frictions."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from threadpoolctl import threadpool_limits

from deep_tunnelling.ume import hop_rates

if TYPE_CHECKING:
    from deep_tunnelling.ume import BandData, Friction, RateResult

Array = np.ndarray[Any, Any]

# bands in the master equation: those whose mean energy lies below MAX_ENERGY
# above the ground band. Activated hopping runs through the bands at ~85-150 meV
# (just below the 143 meV barrier) at every temperature, so the cutoff is a fixed
# energy rather than a multiple of k_B T: 6 k_B T drops these bands and
# underestimates the H rate by 87% at 160 K (D: 88%). H: 10 bands, D: 16.
MAX_ENERGY = 150.0  # meV


@dataclass(frozen=True, kw_only=True)
class RateCurve:
    """Slow (out of fcc) and fast (out of hcp) escape rates (ps^-1) at one friction."""

    mass: float
    friction: Friction
    temperatures: Array  # K
    slow: Array
    fast: Array


def n_rate_for(bands: BandData) -> int:
    """Count the bands whose mean energy lies within MAX_ENERGY of the ground band."""
    mean = bands.energies.mean(axis=0)
    return int(np.sum(mean - mean[0] < MAX_ENERGY))


def _single_threaded_hop_rates(
    bands: BandData, mass: float, frictions: list[Friction], temperature: float
) -> list[RateResult]:
    # one BLAS thread per process, so parallel temperatures do not oversubscribe the cores
    with threadpool_limits(limits=1, user_api="blas"):
        return hop_rates(bands, mass, temperature, frictions, n_rate_for(bands))


def rates(
    bands: BandData, mass: float, temperatures: Array, frictions: list[Friction]
) -> list[RateCurve]:
    """Escape-rate curves, one per friction."""
    # one process per temperature
    with ProcessPoolExecutor() as pool:
        results = list(
            pool.map(
                partial(_single_threaded_hop_rates, bands, mass, frictions),
                temperatures,
            )
        )  # (T, friction)
    return [
        RateCurve(
            mass=mass,
            friction=friction,
            temperatures=np.asarray(temperatures),
            slow=np.array([row[j].fcc_escape_rate for row in results]),
            fast=np.array([row[j].hcp_escape_rate for row in results]),
        )
        for j, friction in enumerate(frictions)
    ]
