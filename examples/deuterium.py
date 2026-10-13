"""D on Ni(111): DFT and optimized PES, with the single friction fitted to H on each.

Electron-hole friction acts through eta = m gamma, which does not depend on the
adsorbate mass, so gamma_D = gamma_H m_H / m_D. Nothing is refitted to D, which
is a secondary test (only its activation energy is considered reliable).
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from experiment import D_RATES, implied_fast_rates, plot_experimental_rates
from hydrogen import fit_hydrogen
from util import build_bands

from deep_tunnelling.model import RateCurve, rates
from deep_tunnelling.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
    plot_rate_arrhenius,
)
from deep_tunnelling.units import M_D, M_H
from deep_tunnelling.util import cache_base_path, cached

if TYPE_CHECKING:
    from deep_tunnelling.ume import Friction

TEMPERATURES = np.linspace(115.0, 260.0, 8)


@cached(
    lambda friction, *, optimized: Path(
        f"deuterium.{'optimized' if optimized else 'dft'}.{friction.gamma:.6f}.pkl"
    )
)
def solve_deuterium(friction: Friction, *, optimized: bool) -> RateCurve:
    (curve,) = rates(
        build_bands(M_D, optimized=optimized), M_D, TEMPERATURES, [friction]
    )
    return curve


def _main() -> None:
    fig, ax = get_fancy_figure()
    for data, color, label in (
        (D_RATES, CAM_CHERRY.dark, "Zhu, slow"),
        (
            implied_fast_rates(D_RATES),
            CAM_CHERRY.warm,
            r"Zhu, fast ($\times e^{\Delta E / k_B T}$)",
        ),
    ):
        _, _, points = plot_experimental_rates(data, ax=ax)
        points.set_label(label)
        for artist in points.lines[0], *points.lines[2]:
            artist.set_color(color)
    points.lines[0].set_markerfacecolor("none")  # fast rates: open circles
    for name, optimized, color in (
        ("DFT", False, CAM_BLUE.dark),
        ("optimized", True, CAM_BLUE.warm),
    ):
        # same eta = m gamma as the H fit on this PES
        friction = fit_hydrogen(optimized=optimized).curve.friction.scaled(M_H / M_D)
        print(f"D, {name} PES: gamma = {friction.gamma:.3f} ps^-1 (same eta as H)")
        curve = solve_deuterium(friction, optimized=optimized)
        _, _, slow = plot_rate_arrhenius(curve.temperatures, curve.slow, ax=ax)
        slow.set(
            color=color,
            label=rf"{name} PES, $\gamma = {friction.gamma:.2f}$ ps$^{{-1}}$",
        )
        _, _, fast = plot_rate_arrhenius(curve.temperatures, curve.fast, ax=ax)
        fast.set(color=color, linestyle="--")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("D/Ni(111), same $\\eta$ as H (solid: slow, dashed: fast)", fontsize=9)
    fig.savefig("examples/deuterium.rates.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _main()
