"""D on Pt(111): DFT and corrected PES, with the single friction fitted to H on each.

Electron-hole friction acts through eta = m gamma, so gamma_D = gamma_H m_H / m_D.
Nothing is refitted to D, which tests the PES and the isotope scaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from deep_tunnelling.model import rates
from deep_tunnelling.plot import (
    CAM_CHERRY,
    get_fancy_figure,
    plot_experimental_rates,
    plot_rate_arrhenius,
)
from deep_tunnelling.units import M_D, M_H
from deep_tunnelling.util import cache_base_path, cached
from examples.platinum.experiment import D_RATES, jump_rate
from examples.platinum.hydrogen import corrected_scale, fit_hydrogen
from examples.platinum.util import (
    MAX_ENERGY,
    MAX_WORKERS,
    PLATINUM,
    TEMPERATURES,
    print_activation_energies,
)
from examples.util import chi2

if TYPE_CHECKING:
    from deep_tunnelling.model import RateCurve
    from deep_tunnelling.ume import Friction


@cached(
    lambda scale, friction: Path(
        f"platinum.deuterium.{scale:.4f}.{friction.gamma:.6f}.pkl"
    )
)
def solve_deuterium(scale: float, friction: Friction) -> RateCurve:
    (curve,) = rates(
        PLATINUM.bands(M_D, scale),
        M_D,
        TEMPERATURES,
        [friction],
        max_energy=MAX_ENERGY["D"],
        max_workers=MAX_WORKERS["D"],
    )
    return curve


def _main() -> None:
    fig, ax = get_fancy_figure()
    _, _, points = plot_experimental_rates(D_RATES, ax=ax)
    points.set_label("Jardine, D")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.dark)
    for name, scale, linestyle in (
        ("corrected", corrected_scale(), "-"),
        ("DFT", 1.0, "--"),
    ):
        # same eta = m gamma as the H fit on this PES
        friction = fit_hydrogen(scale).curve.friction.scaled(M_H / M_D)
        curve = solve_deuterium(scale, friction)
        fit_chi2 = chi2(D_RATES, TEMPERATURES, jump_rate(curve))
        print(
            f"D, {name} PES (splitting_scale {scale:.2f}): gamma = "
            f"{friction.gamma:.3f} ps^-1 (same eta as H), chi2 = {fit_chi2:.2f}"
        )
        print_activation_energies("D", D_RATES, curve)
        _, _, line = plot_rate_arrhenius(curve.temperatures, jump_rate(curve), ax=ax)
        line.set(
            color=CAM_CHERRY.warm,
            linestyle=linestyle,
            label=rf"{name} PES ({scale:.1f}), $\gamma = {friction.gamma:.2f}$"
            rf" ps$^{{-1}}$, $\chi^2 = {fit_chi2:.1f}$",
        )
    ax.set_ylabel(r"Hopping rate $\Gamma$ (ps$^{-1}$)")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title(r"D/Pt(111), same $\eta$ as H", fontsize=9)
    fig.savefig("examples/platinum/deuterium.rates.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/platinum/data")):
        _main()
