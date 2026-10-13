"""Plotting: Cambridge-styled figures (from classical_diffusion) and hop-rate plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

Array = np.ndarray[Any, Any]


def get_figure(ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Get the figure of the given axis.

    If no figure exists, a new figure is created
    """
    if ax is None:
        return cast("tuple[Figure, Axes]", plt.subplots())  # type: ignore plt.subplots Unknown type

    fig = cast("Figure|None", ax.get_figure())
    if fig is None:
        fig = cast("Figure", plt.figure())  # type: ignore plt.figure Unknown type
        ax.set_figure(fig)
    return fig, ax


@dataclass(frozen=True, kw_only=True)
class CamColor:
    """A class to hold CAM color palettes."""

    light: str
    warm: str
    base: str
    dark: str


CAM_BLUE = CamColor(light="#D1F9F1", warm="#00BDB6", base="#8EE8D8", dark="#133844")
CAM_CHERRY = CamColor(light="#F2CAD8", warm="#E18AAC", base="#CD3572", dark="#911449")
CAM_CREST = CamColor(light="#FFE2C8", warm="#FFC392", base="#FD8153", dark="#DD3025")

CAM_SLATE_1 = "#ECEEF1"
CAM_SLATE_4 = "#232830"

CAM_COLOR_CYCLE = [
    CAM_BLUE.dark,
    CAM_BLUE.warm,
    CAM_CHERRY.dark,
    CAM_CHERRY.warm,
    CAM_CREST.dark,
    CAM_CREST.warm,
]


def get_fig_size() -> tuple[float, float]:
    """Get default figure size in inches based on document text width."""
    total_textwidth_pt = 437.5
    pt_to_inch = 1 / 72.27
    # We want half width
    plot_width_in = 1.5 * (total_textwidth_pt / 2) * pt_to_inch
    # Height using Golden Ratio (Height = Width * 0.618)
    plot_height_in = plot_width_in * 0.618
    return plot_width_in, plot_height_in


def setup_rc_params() -> None:
    """Set up matplotlib rcParams for consistent figure styling."""
    plt.rcParams.update(
        {
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": CAM_SLATE_4,
            "axes.prop_cycle": cycler(color=CAM_COLOR_CYCLE),
        }
    )


def setup_fancy_figure(fig: Figure, ax: list[Axes]) -> None:
    """Set up a figure and axis with fancy styling."""
    fig.set_facecolor((0, 0, 0, 0))
    for a in ax:
        a.set_facecolor(CAM_SLATE_1)
        a.set_prop_cycle(cycler(color=CAM_COLOR_CYCLE))
        a.tick_params(
            axis="both", direction="in", top=True, right=True, labelsize=8, which="both"
        )
        a.xaxis.label.set_fontsize(11)
        a.yaxis.label.set_fontsize(11)


def get_fancy_figure(
    *, fig_size: tuple[float, float] | None = None
) -> tuple[Figure, Axes]:
    """Create a figure and axis with fancy styling."""
    setup_rc_params()
    fig, ax = plt.subplots(figsize=fig_size or get_fig_size(), layout="constrained")
    setup_fancy_figure(fig, [ax])
    return fig, ax


def set_arrhenius_axes(ax: Axes) -> None:
    """Log rate against 1000 / T axes."""
    ax.set_yscale("log")  # cspell: disable-line
    ax.set_xlabel(r"1000 / T (K$^{-1}$)")
    ax.set_ylabel(r"Escape rate (ps$^{-1}$)")


def plot_rate_arrhenius(
    temperatures: Array, rates: Array, *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    """Plot a model escape rate against 1000 / T on a log scale."""
    fig, ax = get_figure(ax)
    (line,) = ax.plot(1000 / np.asarray(temperatures), rates)
    set_arrhenius_axes(ax)
    return fig, ax, line


def plot_convergence(
    sizes: list[int], rates: Array, *, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    """Plot rates for increasing supercell size n, relative to the largest n."""
    fig, ax = get_figure(ax)
    rates = np.asarray(rates)
    (line,) = ax.plot(sizes, rates / rates[-1], marker="o")
    ax.set_xlabel(r"Supercell size $n$")
    ax.set_ylabel(rf"Rate / rate($n = {sizes[-1]}$)")
    return fig, ax, line


def plot_experimental_rates(
    data: Array, *, ax: Axes | None = None
) -> tuple[Figure, Axes, ErrorbarContainer]:
    """Plot measured rates (columns T, rate, low, high) with their ranges against 1000 / T."""
    fig, ax = get_figure(ax)
    t, r, lo, hi = np.asarray(data).T
    container = ax.errorbar(
        1000 / t, r, yerr=[r - lo, hi - r], linestyle="", marker="o", markersize=4
    )
    set_arrhenius_axes(ax)
    return fig, ax, container
