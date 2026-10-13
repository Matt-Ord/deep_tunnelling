"""H on Ni(111): a single friction fitted to Zhu's slow rates, on the DFT and the optimized PES.

The DFT PES has hcp 16.9 meV above fcc. The optimized PES (util.OPTIMIZED_PES) is
tuned to the measured hollow splitting and activation energy of H.
"""

from pathlib import Path

import numpy as np
from experiment import H_FAST_RATES, H_RATES, plot_experimental_rates
from util import FrictionFit, build_bands, fit_frictions

from deep_tunnelling.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
    plot_rate_arrhenius,
)
from deep_tunnelling.units import M_H
from deep_tunnelling.util import cache_base_path, cached

GAMMAS = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0]  # ps^-1
TEMPERATURES = np.linspace(115.0, 260.0, 10)


@cached(
    lambda *, optimized: Path(f"hydrogen.{'optimized' if optimized else 'dft'}.pkl")
)
def fit_hydrogen(*, optimized: bool) -> FrictionFit:
    """Fit a single friction gamma to the measured slow rates."""
    return fit_frictions(
        build_bands(M_H, optimized=optimized), M_H, TEMPERATURES, H_RATES, GAMMAS
    )


def _main() -> None:
    fits = {
        name: fit_hydrogen(optimized=name == "optimized")
        for name in ("DFT", "optimized")
    }
    fig, ax = get_fancy_figure()
    for data, color, label in (
        (H_RATES, CAM_CHERRY.dark, "Zhu, slow"),
        (H_FAST_RATES, CAM_CHERRY.warm, "Zhu, fast"),
    ):
        _, _, points = plot_experimental_rates(data, ax=ax)
        points.set_label(label)
        for artist in points.lines[0], *points.lines[2]:
            artist.set_color(color)
    points.lines[0].set_markerfacecolor("none")  # fast rates: open circles
    for (name, fit), color in zip(
        fits.items(), (CAM_BLUE.dark, CAM_BLUE.warm), strict=True
    ):
        gamma = fit.curve.friction.gamma
        print(f"H, {name} PES: gamma = {gamma:.3f} ps^-1, chi2 = {fit.chi2:.1f}")
        _, _, slow = plot_rate_arrhenius(fit.curve.temperatures, fit.curve.slow, ax=ax)
        slow.set(
            color=color,
            label=rf"{name} PES, $\gamma = {gamma:.2f}$ ps$^{{-1}}$, $\chi^2 = {fit.chi2:.1f}$",
        )
        _, _, fast = plot_rate_arrhenius(fit.curve.temperatures, fit.curve.fast, ax=ax)
        fast.set(color=color, linestyle="--")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("H/Ni(111), single friction (solid: slow, dashed: fast)", fontsize=9)
    fig.savefig("examples/hydrogen.rates.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _main()
