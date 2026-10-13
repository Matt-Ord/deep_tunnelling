"""D on Ni(111): DFT and optimized PES, with the single friction fitted to H on each.

Electron-hole friction acts through eta = m gamma, which does not depend on the
adsorbate mass, so gamma_D = gamma_H m_H / m_D. Nothing is refitted to D, which
is a secondary test (only its activation energy is considered reliable).
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from deep_tunnelling.model import RateCurve, rates
from deep_tunnelling.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
    plot_experimental_rates,
    plot_rate_arrhenius,
)
from deep_tunnelling.units import M_D, M_H
from deep_tunnelling.util import cache_base_path, cached
from examples.nickel.experiment import D_RATES, implied_fast_rates
from examples.nickel.hydrogen import fit_hydrogen
from examples.nickel.util import MAX_WORKERS, NICKEL, PES

if TYPE_CHECKING:
    from deep_tunnelling.ume import Friction

TEMPERATURES = np.linspace(115.0, 260.0, 8)


@cached(lambda scale, friction: Path(f"deuterium.{scale:.4f}.{friction.gamma:.6f}.pkl"))
def solve_deuterium(scale: float, friction: Friction) -> RateCurve:
    (curve,) = rates(
        NICKEL.bands(M_D, scale),
        M_D,
        TEMPERATURES,
        [friction],
        max_workers=MAX_WORKERS["D"],
    )
    return curve


def _main() -> None:
    fig, ax = get_fancy_figure()
    _, _, points = plot_experimental_rates(D_RATES, ax=ax)
    points.set_label("Zhu, slow")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.dark)
    _, _, points = plot_experimental_rates(implied_fast_rates(D_RATES), ax=ax)
    points.set_label(r"Zhu, fast ($\times e^{\Delta E / k_B T}$)")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.warm)
    points.lines[0].set_markerfacecolor("none")  # fast rates: open circles
    for (name, scale), color in zip(
        PES.items(), (CAM_BLUE.dark, CAM_BLUE.warm), strict=True
    ):
        # same eta = m gamma as the H fit on this PES
        friction = fit_hydrogen(scale).curve.friction.scaled(M_H / M_D)
        print(f"D, {name} PES: gamma = {friction.gamma:.3f} ps^-1 (same eta as H)")
        curve = solve_deuterium(scale, friction)
        _, _, slow = plot_rate_arrhenius(curve.temperatures, curve.slow, ax=ax)
        slow.set(
            color=color,
            label=rf"{name} PES, $\gamma = {friction.gamma:.2f}$ ps$^{{-1}}$",
        )
        _, _, fast = plot_rate_arrhenius(curve.temperatures, curve.fast, ax=ax)
        fast.set(color=color, linestyle="--")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("D/Ni(111), same $\\eta$ as H (solid: slow, dashed: fast)", fontsize=9)
    fig.savefig("examples/nickel/nickel.deuterium.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/nickel/data")):
        _main()
