"""H on Ni(111): a single friction fitted to Zhu's slow rates, on the DFT and the optimized PES.

The DFT PES has hcp 16.9 meV above fcc. The optimized PES (util.OPTIMIZED_SCALE)
has the measured hollow splitting of H.
"""

from pathlib import Path

import numpy as np

from deep_tunnelling.plot import get_fancy_figure
from deep_tunnelling.units import M_H
from deep_tunnelling.util import cache_base_path, cached
from examples.nickel.experiment import H_FAST_RATES, H_RATES
from examples.nickel.util import MAX_WORKERS, NICKEL, PES
from examples.util import (
    PES_COLORS,
    FrictionFit,
    fit_frictions,
    plot_slow_fast_curve,
    plot_slow_fast_data,
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
    plot_slow_fast_data(ax, (H_RATES, "Zhu, slow"), (H_FAST_RATES, "Zhu, fast"))
    for (name, fit), color in zip(fits.items(), PES_COLORS, strict=True):
        gamma = fit.curve.friction.gamma
        print(f"H, {name} PES: gamma = {gamma:.3f} ps^-1, chi2 = {fit.chi2:.1f}")
        plot_slow_fast_curve(
            ax,
            fit.curve,
            color,
            rf"{name} PES, $\gamma = {gamma:.2f}$ ps$^{{-1}}$, $\chi^2 = {fit.chi2:.1f}$",
        )
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("H/Ni(111), single friction (solid: slow, dashed: fast)", fontsize=9)
    fig.savefig("examples/nickel/hydrogen.rates.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/nickel/data")):
        _main()
