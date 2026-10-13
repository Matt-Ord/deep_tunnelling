"""H on Ni(111): a separate friction for ground-state fcc <-> hcp transitions, on the DFT and optimized PES.

A single friction describes the activated regime but not the deep-tunneling
plateau. The ground-state friction gamma_g (ume.py) is a phenomenological
placeholder for dissipation between the hollow ground states that the Ohmic
friction misses (e.g. surface relaxation). gamma and gamma_g are fitted to Zhu's
slow rates on each PES.
"""

from pathlib import Path

from experiment import H_FAST_RATES, H_RATES, plot_experimental_rates
from hydrogen import GAMMAS, TEMPERATURES
from util import FrictionFit, build_bands, fit_frictions

from deep_tunnelling.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
    plot_rate_arrhenius,
)
from deep_tunnelling.units import M_H
from deep_tunnelling.util import cache_base_path, cached

GROUND_GAMMAS = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]  # ps^-1


@cached(
    lambda *, optimized: Path(
        f"hydrogen.{'optimized' if optimized else 'dft'}.ground.pkl"
    )
)
def fit_hydrogen_ground(*, optimized: bool) -> FrictionFit:
    """Fit gamma and gamma_g to the measured slow rates."""
    return fit_frictions(
        build_bands(M_H, optimized=optimized),
        M_H,
        TEMPERATURES,
        H_RATES,
        GAMMAS,
        GROUND_GAMMAS,
    )


def _main() -> None:
    fits = {
        name: fit_hydrogen_ground(optimized=name == "optimized")
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
        friction = fit.curve.friction
        print(
            f"H, {name} PES, two frictions: gamma = {friction.gamma:.3f} ps^-1, "
            f"gamma_g = {friction.ground:.3f} ps^-1, chi2 = {fit.chi2:.1f}"
        )
        _, _, slow = plot_rate_arrhenius(fit.curve.temperatures, fit.curve.slow, ax=ax)
        slow.set(
            color=color,
            label=rf"{name} PES, $\gamma = {friction.gamma:.2f}$, "
            rf"$\gamma_g = {friction.ground:.1f}$ ps$^{{-1}}$, $\chi^2 = {fit.chi2:.1f}$",
        )
        _, _, fast = plot_rate_arrhenius(fit.curve.temperatures, fit.curve.fast, ax=ax)
        fast.set(color=color, linestyle="--")
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title(
        "H/Ni(111), ground-state friction (solid: slow, dashed: fast)", fontsize=9
    )
    fig.savefig("examples/hydrogen.rates.ground.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _main()
