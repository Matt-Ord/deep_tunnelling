"""H on Ru(0001): a separate friction for ground-state fcc <-> hcp transitions, on the DFT and corrected PES.

As on Ni(111), a single friction describes the activated regime but not the
deep-tunneling plateau. The ground-state friction gamma_g (ume.py) is a
phenomenological placeholder for dissipation between the hollow ground states
that the Ohmic friction misses. gamma and gamma_g are fitted to McIntosh's fcc
escape rates on each PES, which reach the plateau (75-280 K); Townsend's hcp
escape rates are not fitted.
"""

from pathlib import Path

from deep_tunnelling.plot import get_fancy_figure
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
    PES_COLORS,
    FrictionFit,
    chi2,
    fit_frictions,
    plot_slow_fast_curve,
    plot_slow_fast_data,
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
    plot_slow_fast_data(
        ax,
        (MCINTOSH_H_RATES, "McIntosh, slow"),
        (H_RATES, "Townsend, fast (not fitted)"),
    )
    for (name, scale), color in zip(PES.items(), PES_COLORS, strict=True):
        fit = fit_hydrogen_ground(scale)
        curve, friction = fit.curve, fit.curve.friction
        townsend_chi2 = chi2(H_RATES, curve.temperatures, curve.fast)
        print(
            f"H, {name} PES, two frictions: gamma = {friction.gamma:.3f} ps^-1, "
            f"gamma_g = {friction.ground:.3f} ps^-1, chi2 = {fit.chi2:.2f} "
            f"(Townsend, not fitted: {townsend_chi2:.2f})"
        )
        plot_slow_fast_curve(
            ax,
            curve,
            color,
            rf"{name} PES, $\gamma = {friction.gamma:.2f}$, "
            rf"$\gamma_g = {friction.ground:.0f}$ ps$^{{-1}}$, "
            rf"$\chi^2 = {fit.chi2:.1f}$ (slow), {townsend_chi2:.1f} (fast)",
        )
    ax.set_ylabel(r"Escape rate (ps$^{-1}$)")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title(
        "H/Ru(0001), ground-state friction (solid: slow, dashed: fast)", fontsize=9
    )
    fig.savefig("examples/ruthenium/hydrogen.rates.ground.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/ruthenium/data")):
        _main()
