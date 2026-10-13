"""D on Ru(0001): DFT and corrected PES, with the single friction fitted to H on each.

Electron-hole friction acts through eta = m gamma, so gamma_D = gamma_H m_H / m_D.
Nothing is refitted to D, which tests the PES and the isotope scaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from deep_tunnelling.model import rates
from deep_tunnelling.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
    plot_experimental_rates,
    plot_rate_arrhenius,
)
from deep_tunnelling.units import M_D, M_H
from deep_tunnelling.util import cache_base_path, cached
from examples.ruthenium.experiment import D_RATES
from examples.ruthenium.hydrogen import PES, fit_hydrogen
from examples.ruthenium.util import (
    MAX_ENERGY,
    MAX_WORKERS,
    RUTHENIUM,
    SUPERCELL,
    TEMPERATURES,
    print_activation_energy,
)
from examples.util import (
    chi2,
)

if TYPE_CHECKING:
    from deep_tunnelling.model import RateCurve
    from deep_tunnelling.ume import Friction


@cached(
    lambda scale, friction: Path(
        f"ruthenium.deuterium.{scale:.4f}.{friction.gamma:.6f}.pkl"
    )
)
def solve_deuterium(scale: float, friction: Friction) -> RateCurve:
    (curve,) = rates(
        RUTHENIUM.bands(M_D, scale, SUPERCELL["D"]),
        M_D,
        TEMPERATURES,
        [friction],
        max_energy=MAX_ENERGY,
        max_workers=MAX_WORKERS["D"],
    )
    return curve


def _main() -> None:
    fig, ax = get_fancy_figure()
    _, _, points = plot_experimental_rates(D_RATES, ax=ax)
    points.set_label("Townsend, fast")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.warm)
    points.lines[0].set_markerfacecolor("none")  # fast rates: open circles
    for (name, scale), color in zip(
        PES.items(), (CAM_BLUE.dark, CAM_BLUE.warm), strict=True
    ):
        # same eta = m gamma as the H fit on this PES
        friction = fit_hydrogen(scale).curve.friction.scaled(M_H / M_D)
        curve = solve_deuterium(scale, friction)
        fit_chi2 = chi2(D_RATES, TEMPERATURES, curve.fast)
        print(
            f"D, {name} PES (splitting_scale {scale:.3f}): gamma = "
            f"{friction.gamma:.3f} ps^-1 (same eta as H), chi2 = {fit_chi2:.2f}"
        )
        print_activation_energy("D", D_RATES, curve)
        _, _, slow = plot_rate_arrhenius(curve.temperatures, curve.slow, ax=ax)
        slow.set(
            color=color,
            label=rf"{name} PES, $\gamma = {friction.gamma:.2f}$ ps$^{{-1}}$, "
            rf"$\chi^2 = {fit_chi2:.1f}$ (fast)",
        )
        _, _, fast = plot_rate_arrhenius(curve.temperatures, curve.fast, ax=ax)
        fast.set(color=color, linestyle="--")
    ax.set_ylabel(r"Escape rate (ps$^{-1}$)")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title(
        r"D/Ru(0001), same $\eta$ as H (solid: slow, dashed: fast)", fontsize=9
    )
    fig.savefig("examples/ruthenium/ruthenium.deuterium.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/ruthenium/data")):
        _main()
