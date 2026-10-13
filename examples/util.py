"""Shared helpers for the examples: the DFT surfaces, friction fits to measured rates, chi^2 in log rate."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.interpolate
import scipy.optimize

from deep_tunnelling.bloch import BlochSolver
from deep_tunnelling.model import rates
from deep_tunnelling.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    plot_experimental_rates,
    plot_rate_arrhenius,
)
from deep_tunnelling.potential import SurfacePotential, load_points
from deep_tunnelling.ume import HCP, BandData, Friction
from deep_tunnelling.units import KB

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from matplotlib.axes import Axes

    from deep_tunnelling.bloch import Grid
    from deep_tunnelling.model import RateCurve

Array = np.ndarray[Any, Any]


@dataclass(frozen=True, kw_only=True)
class Surface:
    """DFT data (raw_data.json format) of one adsorbate/surface system and its numerical grid."""

    raw_data: Path
    grid: Grid  # its lattice constant is that of the DFT slab
    n_z: int = 6  # z states of the contracted basis (BlochSolver)

    def potential(self, splitting_scale: float = 1.0) -> SurfacePotential:
        """Return the PES, with the fcc/hcp-odd stars scaled by splitting_scale."""
        return SurfacePotential.from_points(
            *load_points(self.raw_data, self.grid.a), splitting_scale=splitting_scale
        )

    def bands(self, mass: float, splitting_scale: float = 1.0, n: int = 6) -> BandData:
        """Bloch states of an adsorbate of the given mass on the PES, n x n supercell."""
        solver = BlochSolver(
            self.potential(splitting_scale), mass, self.grid, n_z=self.n_z
        )
        return BandData.build(solver, n)


def hollow_splitting(bands: BandData) -> float:
    """Return the energy (meV) of the lowest hcp band above the ground (fcc) band, averaged over k."""
    mean = bands.energies.mean(axis=0)
    return float(mean[np.argmax(bands.sites[0] == HCP)] - mean[0])


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


def slow_rate(curve: RateCurve) -> Array:
    """Return the escape rate out of fcc (H/Ni(111): Zhu's measured rate)."""
    return curve.slow


def fast_rate(curve: RateCurve) -> Array:
    """Return the escape rate out of hcp (H/Ru(0001): Townsend's 1 / tau)."""
    return curve.fast


def fit_frictions(  # noqa: PLR0913, PLR0917
    bands: BandData,
    mass: float,
    temperatures: Array,
    data: Array,
    gammas: list[float],
    ground_gammas: list[float] | None = None,
    *,
    observable: Callable[[RateCurve], Array] = slow_rate,
    **rate_options: Any,  # noqa: ANN401
) -> FrictionFit:
    """Fit gamma (and gamma_g, if a grid is given) to a measured rate.

    The observable (by default the slow rate) is computed on the grid of
    frictions, chi^2 is minimized on a cubic spline of its log against log
    friction, and the best fit is evaluated. rate_options go to model.rates.
    """
    axes = [np.asarray(gammas), np.asarray(ground_gammas or [0.0])]
    free = [i for i, a in enumerate(axes) if len(a) > 1]
    grid = [Friction(g, gg) for g in axes[0] for gg in axes[1]]
    curves = rates(bands, mass, temperatures, grid, **rate_options)
    log_slow = np.log([observable(c) for c in curves]).reshape(*map(len, axes), -1)
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

    start = grid[np.argmin([chi2(data, temperatures, observable(c)) for c in curves])]
    best = scipy.optimize.minimize(
        objective,
        np.log([(start.gamma, start.ground)[i] for i in free]),
        method="Nelder-Mead",
        bounds=list(zip(lower, upper, strict=True)),
        options={"xatol": 1e-4, "fatol": 1e-6},
    ).x
    if np.any(np.isclose(best, lower, atol=1e-3) | np.isclose(best, upper, atol=1e-3)):
        warnings.warn(
            f"best fit {friction(best)} on the edge of the friction grid", stacklevel=2
        )
    (curve,) = rates(bands, mass, temperatures, [friction(best)], **rate_options)
    return FrictionFit(curve=curve, chi2=chi2(data, temperatures, observable(curve)))


def activation_energy(
    temperatures: Array, rate: Array, sigma: Array | None = None
) -> float:
    """Return the slope (meV) of a (weighted) straight-line fit of ln rate against -1 / kT."""
    weights = None if sigma is None else 1 / sigma
    return float(np.polyfit(-1 / (KB * temperatures), np.log(rate), 1, w=weights)[0])


# colours shared by the slow/fast rate plots: data in cherry (slow dark and filled,
# fast light and open), one blue per PES (DFT dark, corrected light), and the model
# rates solid (slow) and dashed (fast)
PES_COLORS = (CAM_BLUE.dark, CAM_BLUE.warm)


def plot_slow_fast_data(
    ax: Axes, slow: tuple[Array, str] | None, fast: tuple[Array, str] | None
) -> None:
    """Plot measured slow (filled) and fast (open) escape rates, each with its label."""
    for series, color in ((slow, CAM_CHERRY.dark), (fast, CAM_CHERRY.warm)):
        if series is None:
            continue
        data, label = series
        _, _, points = plot_experimental_rates(data, ax=ax)
        points.set_label(label)
        for artist in points.lines[0], *points.lines[2]:
            artist.set_color(color)
        if series is fast:
            points.lines[0].set_markerfacecolor("none")


def plot_slow_fast_curve(ax: Axes, curve: RateCurve, color: str, label: str) -> None:
    """Plot a model's slow (solid, labelled) and fast (dashed) escape rates."""
    _, _, slow = plot_rate_arrhenius(curve.temperatures, curve.slow, ax=ax)
    slow.set(color=color, label=label)
    _, _, fast = plot_rate_arrhenius(curve.temperatures, curve.fast, ax=ax)
    fast.set(color=color, linestyle="--")
