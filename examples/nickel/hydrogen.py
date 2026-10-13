"""H on Ni(111): a single friction fitted to Zhu's slow rates, on the DFT and the optimized PES.

The DFT PES has hcp 16.9 meV above fcc. The optimized PES (util.OPTIMIZED_SCALE)
has the measured hollow splitting of H.
"""

from pathlib import Path

import numpy as np

from deep_tunnelling.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
    plot_experimental_rates,
    plot_rate_arrhenius,
)
from deep_tunnelling.units import M_H
from deep_tunnelling.util import cache_base_path, cached
from examples.nickel.experiment import H_FAST_RATES, H_RATES
from examples.nickel.util import MAX_WORKERS, NICKEL, PES
from examples.util import (
    FrictionFit,
    fit_frictions,
)

GAMMAS = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0]  # ps^-1
TEMPERATURES = np.linspace(115.0, 260.0, 10)


@cached(lambda scale: Path(f"hydrogen.{scale:.4f}.pkl"))
def fit_hydrogen(scale: float) -> FrictionFit:
    """Fit a single friction gamma to the measured slow rates."""
    return fit_frictions(
        NICKEL.bands(M_H, scale),
        M_H,
        TEMPERATURES,
        H_RATES,
        GAMMAS,
        max_workers=MAX_WORKERS["H"],
    )


def _main() -> None:
    fits = {name: fit_hydrogen(scale) for name, scale in PES.items()}
    fig, ax = get_fancy_figure()
    _, _, points = plot_experimental_rates(H_RATES, ax=ax)
    points.set_label("Zhu, slow")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.dark)
    _, _, points = plot_experimental_rates(H_FAST_RATES, ax=ax)
    points.set_label("Zhu, fast")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.warm)
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
    fig.savefig("examples/nickel/nickel.hydrogen.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/nickel/data")):
        _main()
