"""HeSE hop rates of H and D on Ni(111), J. Zhu PhD thesis (Cambridge, 2015).

Digitized from Fig. 4.21 (0.07 ML). The rate is the 'slow' rate, i.e. the total
escape rate out of the lower-energy (fcc) hollow, in ps^-1, with the +-1 sigma
range. Repeated temperatures are separate measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from deep_tunnelling.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
    get_figure,
    plot_rate_arrhenius,
    set_arrhenius_axes,
)
from deep_tunnelling.units import KB

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure

    from deep_tunnelling.model import RateCurve

Array = np.ndarray[Any, Any]

# DFT PES of H on Ni(111) (M. Sacchi), six sites x nine heights
RAW_DATA = Path(__file__).parent / "raw_data.json"

# columns: temperature in K, rate, lower and upper error bounds
H_RATES = np.array(
    [
        (255, 0.0237, 0.0228, 0.0245),
        (245, 0.0166, 0.0157, 0.0174),
        (235, 0.0145, 0.0137, 0.0153),
        (230, 0.0144, 0.0137, 0.0152),
        (225, 0.0117, 0.0110, 0.0125),
        (220, 0.0105, 0.0097, 0.0112),
        (210, 0.0099, 0.0088, 0.0110),
        (205, 0.0069, 0.0067, 0.0071),
        (200, 0.0064, 0.0061, 0.0067),
        (195, 0.0048, 0.0046, 0.0051),
        (190, 0.0048, 0.0043, 0.0052),
        (185, 0.0037, 0.0034, 0.0040),
        (180, 0.0039, 0.0034, 0.0043),
        (175, 0.0029, 0.0028, 0.0031),
        (165, 0.0032, 0.0027, 0.0038),
        (160, 0.0030, 0.0029, 0.0032),
        (155, 0.0030, 0.0027, 0.0033),
        (145, 0.0025, 0.0019, 0.0032),
        (145, 0.0024, 0.0019, 0.0032),
        (135, 0.0034, 0.0029, 0.0039),
        (135, 0.0030, 0.0029, 0.0039),
        (130, 0.0039, 0.0030, 0.0047),
        (125, 0.0026, 0.0015, 0.0037),
        (120, 0.0044, 0.0037, 0.0051),
    ]
)

D_RATES = np.array(
    [
        (230, 0.00614, 0.0059, 0.0064),
        (225, 0.00533, 0.0051, 0.0056),
        (215, 0.00429, 0.0041, 0.0045),
        (210, 0.00344, 0.0032, 0.0037),
        (205, 0.00319, 0.0030, 0.0034),
        (200, 0.00259, 0.0024, 0.0028),
        (190, 0.00221, 0.0020, 0.0024),
        (180, 0.00168, 0.0015, 0.0019),
        (175, 0.00161, 0.0014, 0.0019),
        (165, 0.00148, 0.0013, 0.0017),
        (158, 0.00155, 0.0014, 0.0017),
        (155, 0.00148, 0.0012, 0.0018),
        (153, 0.00138, 0.0011, 0.0017),
        (150, 0.00131, 0.0011, 0.0015),
        (148, 0.00148, 0.0013, 0.0017),
    ]
)

# fcc-hcp energy difference from the T dependence of the fast/slow ratio, meV
DELTA_E_HOLLOW = 19.0


def plot_experimental_rates(
    data: Array, *, ax: Axes | None = None, fast: bool = False
) -> tuple[Figure, Axes, ErrorbarContainer]:
    """Plot Zhu's HeSE rates (columns T, rate, low, high) against 1000 / T.

    With fast=True the measured slow rate is converted to the fast rate with the
    measured fast/slow ratio exp(Delta E / kT), Delta E = 19 meV.
    """
    fig, ax = get_figure(ax)
    t, r, lo, hi = np.asarray(data).T
    scale = np.exp(DELTA_E_HOLLOW / (KB * t)) if fast else np.ones_like(t)
    container = ax.errorbar(
        1000 / t,
        r * scale,
        yerr=[(r - lo) * scale, (hi - r) * scale],
        linestyle="",
        marker="o",
        markersize=4,
    )
    set_arrhenius_axes(ax)
    return fig, ax, container


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


def plot_against_experiment(
    curve: RateCurve, data: Array, title: str, path: str
) -> None:
    """Plot model slow/fast rates against Zhu's measured slow and implied fast rates."""
    fig, ax = get_fancy_figure()

    _, _, implied_fast = plot_experimental_rates(data, ax=ax, fast=True)
    implied_fast.set_label(r"Zhu, fast ($\times e^{\Delta E / k_B T}$)")
    for artist in implied_fast.lines[0], *implied_fast.lines[2]:
        artist.set_color(CAM_CHERRY.warm)
    implied_fast.lines[0].set_markerfacecolor("none")

    _, _, measured = plot_experimental_rates(data, ax=ax)
    measured.set_label("Zhu, slow")
    for artist in measured.lines[0], *measured.lines[2]:
        artist.set_color(CAM_CHERRY.dark)

    _, _, slow_line = plot_rate_arrhenius(curve.temperatures, curve.slow, ax=ax)
    slow_line.set_label(rf"UME slow, $\gamma = {curve.gamma:.2f}$ ps$^{{-1}}$")
    slow_line.set_color(CAM_BLUE.dark)
    slow_line.set_linewidth(2.5)

    _, _, fast_line = plot_rate_arrhenius(curve.temperatures, curve.fast, ax=ax)
    fast_line.set_label("UME fast")
    fast_line.set_color(CAM_BLUE.warm)
    fast_line.set_linestyle("--")

    ax.legend(
        handles=[measured, implied_fast, slow_line, fast_line],
        loc="lower left",
        fontsize=7,
    )
    ax.set_title(title, fontsize=10)
    fig.savefig(path)
