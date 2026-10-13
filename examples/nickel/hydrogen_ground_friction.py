"""H on Ni(111): a separate friction for ground-state fcc <-> hcp transitions, on the DFT and optimized PES.

A single friction describes the activated regime but not the deep-tunneling
plateau. The ground-state friction gamma_g (ume.py) is a phenomenological
placeholder for dissipation between the hollow ground states that the Ohmic
friction misses (e.g. surface relaxation). gamma and gamma_g are fitted to Zhu's
slow rates on each PES.
"""

from pathlib import Path

from deep_tunnelling.plot import get_fancy_figure
from deep_tunnelling.units import M_H
from deep_tunnelling.util import cache_base_path, cached
from examples.nickel.experiment import H_FAST_RATES, H_RATES
from examples.nickel.hydrogen import GAMMAS, TEMPERATURES
from examples.nickel.util import MAX_WORKERS, NICKEL, PES
from examples.util import (
    PES_COLORS,
    FrictionFit,
    fit_frictions,
    plot_slow_fast_curve,
    plot_slow_fast_data,
)

GROUND_GAMMAS = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]  # ps^-1


@cached(lambda scale: Path(f"hydrogen.{scale:.4f}.ground.pkl"))
def fit_hydrogen_ground(scale: float) -> FrictionFit:
    """Fit gamma and gamma_g to the measured slow rates."""
    return fit_frictions(
        NICKEL.bands(M_H, scale),
        M_H,
        TEMPERATURES,
        H_RATES,
        GAMMAS,
        GROUND_GAMMAS,
        max_workers=MAX_WORKERS["H"],
    )


def _main() -> None:
    fits = {name: fit_hydrogen_ground(scale) for name, scale in PES.items()}
    fig, ax = get_fancy_figure()
    plot_slow_fast_data(ax, (H_RATES, "Zhu, slow"), (H_FAST_RATES, "Zhu, fast"))
    for (name, fit), color in zip(fits.items(), PES_COLORS, strict=True):
        friction = fit.curve.friction
        print(
            f"H, {name} PES, two frictions: gamma = {friction.gamma:.3f} ps^-1, "
            f"gamma_g = {friction.ground:.3f} ps^-1, chi2 = {fit.chi2:.1f}"
        )
        plot_slow_fast_curve(
            ax,
            fit.curve,
            color,
            rf"{name} PES, $\gamma = {friction.gamma:.2f}$, "
            rf"$\gamma_g = {friction.ground:.1f}$ ps$^{{-1}}$, $\chi^2 = {fit.chi2:.1f}$",
        )
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title(
        "H/Ni(111), ground-state friction (solid: slow, dashed: fast)", fontsize=9
    )
    fig.savefig("examples/nickel/hydrogen.rates.ground.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/nickel/data")):
        _main()
