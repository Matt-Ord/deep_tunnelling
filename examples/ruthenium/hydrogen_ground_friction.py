"""H on Ru(0001): a separate friction for ground-state fcc <-> hcp transitions, on the DFT and corrected PES.

As on Ni(111), a single friction describes the activated regime but not the
deep-tunneling plateau. The ground-state friction gamma_g (ume.py) is a
phenomenological placeholder for dissipation between the hollow ground states
that the Ohmic friction misses. gamma and gamma_g are fitted to McIntosh's fcc
escape rates on each PES, which reach the plateau (75-280 K); Townsend's hcp
escape rates are not fitted.
"""

from pathlib import Path

from deep_tunnelling.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
    plot_experimental_rates,
    plot_rate_arrhenius,
)
from deep_tunnelling.units import M_H
from deep_tunnelling.util import cache_base_path, cached
from examples.ruthenium.experiment import H_RATES, MCINTOSH_H_RATES
from examples.ruthenium.hydrogen import PES
from examples.ruthenium.util import (
    MAX_ENERGY,
    MAX_WORKERS,
    RUTHENIUM,
    SUPERCELL,
    TEMPERATURES,
)
from examples.util import (
    FrictionFit,
    chi2,
    fit_frictions,
    slow_rate,
)

GAMMAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # ps^-1
GROUND_GAMMAS = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]  # ps^-1


@cached(lambda scale: Path(f"ruthenium.hydrogen.{scale:.4f}.ground.pkl"))
def fit_hydrogen_ground(scale: float) -> FrictionFit:
    """Fit gamma and gamma_g to McIntosh's H fcc escape rates on one PES."""
    return fit_frictions(
        RUTHENIUM.bands(M_H, scale, SUPERCELL["H"]),
        M_H,
        TEMPERATURES,
        MCINTOSH_H_RATES,
        GAMMAS,
        GROUND_GAMMAS,
        observable=slow_rate,
        max_energy=MAX_ENERGY,
        max_workers=MAX_WORKERS["H"],
    )


def _main() -> None:
    fig, ax = get_fancy_figure()
    _, _, points = plot_experimental_rates(MCINTOSH_H_RATES, ax=ax)
    points.set_label("McIntosh, slow")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.dark)
    _, _, points = plot_experimental_rates(H_RATES, ax=ax)
    points.set_label("Townsend, fast (not fitted)")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.warm)
    points.lines[0].set_markerfacecolor("none")  # fast rates: open circles
    for (name, scale), color in zip(
        PES.items(), (CAM_BLUE.dark, CAM_BLUE.warm), strict=True
    ):
        fit = fit_hydrogen_ground(scale)
        curve, friction = fit.curve, fit.curve.friction
        townsend_chi2 = chi2(H_RATES, curve.temperatures, curve.fast)
        print(
            f"H, {name} PES, two frictions: gamma = {friction.gamma:.3f} ps^-1, "
            f"gamma_g = {friction.ground:.3f} ps^-1, chi2 = {fit.chi2:.2f} "
            f"(Townsend, not fitted: {townsend_chi2:.2f})"
        )
        _, _, slow = plot_rate_arrhenius(curve.temperatures, curve.slow, ax=ax)
        slow.set(
            color=color,
            label=rf"{name} PES, $\gamma = {friction.gamma:.2f}$, "
            rf"$\gamma_g = {friction.ground:.0f}$ ps$^{{-1}}$, "
            rf"$\chi^2 = {fit.chi2:.1f}$ (slow), {townsend_chi2:.1f} (fast)",
        )
        _, _, fast = plot_rate_arrhenius(curve.temperatures, curve.fast, ax=ax)
        fast.set(color=color, linestyle="--")
    ax.set_ylabel(r"Escape rate (ps$^{-1}$)")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title(
        "H/Ru(0001), ground-state friction (solid: slow, dashed: fast)", fontsize=9
    )
    fig.savefig("examples/ruthenium/ruthenium.hydrogen.ground.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/ruthenium/data")):
        _main()
