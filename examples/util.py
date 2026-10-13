"""Shared helpers for the examples: the DFT PES, Bloch states and friction fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.interpolate
import scipy.optimize

from deep_tunnelling.bloch import BlochSolver
from deep_tunnelling.model import rates
from deep_tunnelling.potential import SurfacePotential
from deep_tunnelling.ume import BandData, Friction

if TYPE_CHECKING:
    from deep_tunnelling.model import RateCurve

Array = np.ndarray[Any, Any]

# DFT PES of H on Ni(111) (M. Sacchi), six sites x nine heights
RAW_DATA = Path(__file__).parent / "raw_data.json"


# PES corrections fitted to H: the hcp-fcc ground-state splitting of H is the
# measured 19 meV, and the activation energy of the H slow rate above 170 K
# (single friction) is the measured 97.7 meV (THEORY.md 7.4)
OPTIMIZED_PES = {"splitting_scale": 1.472, "bridge_shift": -3.5}


def build_bands(mass: float, n: int = 6, *, optimized: bool = False) -> BandData:
    """Bloch states of an adsorbate of the given mass on the DFT (or optimized) PES, n x n supercell."""
    potential = SurfacePotential.from_dft(
        RAW_DATA, **(OPTIMIZED_PES if optimized else {})
    )
    return BandData.build(BlochSolver(potential, mass), n)


@dataclass(frozen=True, kw_only=True)
class FrictionFit:
    """A rate curve at the friction that best fits the measured slow rates."""

    curve: RateCurve
    chi2: float  # reduced chi^2 in log rate


def chi2(data: Array, temperatures: Array, slow: Array) -> float:
    """Reduced chi^2 in log rate of a model slow rate against measured slow rates."""
    t, r, lo, hi = np.asarray(data).T
    sigma = 0.5 * (np.log(hi) - np.log(lo))
    return float(
        np.mean(((np.interp(t, temperatures, np.log(slow)) - np.log(r)) / sigma) ** 2)
    )


def fit_frictions(  # noqa: PLR0913, PLR0917
    bands: BandData,
    mass: float,
    temperatures: Array,
    data: Array,
    gammas: list[float],
    ground_gammas: list[float] | None = None,
) -> FrictionFit:
    """Fit gamma (and gamma_g, if a grid is given) to measured slow rates.

    The slow rate is computed on the grid of frictions, chi^2 is minimized on a
    cubic spline of log rate against log friction, and the best fit is evaluated.
    """
    axes = [np.asarray(gammas), np.asarray(ground_gammas or [0.0])]
    free = [i for i, a in enumerate(axes) if len(a) > 1]
    grid = [Friction(g, gg) for g in axes[0] for gg in axes[1]]
    curves = rates(bands, mass, temperatures, grid)
    log_slow = np.log([c.slow for c in curves]).reshape(*map(len, axes), -1)
    spline = scipy.interpolate.RegularGridInterpolator(
        [np.log(axes[i]) for i in free],
        log_slow.squeeze(axis=tuple(i for i in range(2) if i not in free)),
        method="cubic",
    )
    lower = np.array([np.log(axes[i][0]) for i in free])
    upper = np.array([np.log(axes[i][-1]) for i in free])

    def friction(x: Array) -> Friction:
        values = [float(a[0]) for a in axes]
        for i, xi in zip(free, np.exp(np.clip(x, lower, upper)), strict=True):
            values[i] = float(xi)
        return Friction(*values)

    def objective(x: Array) -> float:
        return chi2(data, temperatures, np.exp(spline(np.clip(x, lower, upper))[0]))

    start = grid[np.argmin([chi2(data, temperatures, c.slow) for c in curves])]
    best = scipy.optimize.minimize(
        objective,
        np.log([(start.gamma, start.ground)[i] for i in free]),
        method="Nelder-Mead",
        bounds=list(zip(lower, upper, strict=True)),
        options={"xatol": 1e-4, "fatol": 1e-6},
    ).x
    if np.any(np.isclose(best, lower, atol=1e-3) | np.isclose(best, upper, atol=1e-3)):
        print(f"warning: best fit {friction(best)} on the edge of the friction grid")
    (curve,) = rates(bands, mass, temperatures, [friction(best)])
    return FrictionFit(curve=curve, chi2=chi2(data, temperatures, curve.slow))
