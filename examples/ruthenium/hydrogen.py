"""H on Ru(0001): a single friction fitted to Townsend's hcp escape rates, on the DFT and the corrected PES.

On the DFT PES the H hcp ground state lies 53 meV above fcc; the corrected PES
(util.CORRECTED_SCALE) brings it to the measured 22 meV. gamma is fitted to
Townsend's 1 / tau in hcp, the model's escape rate out of hcp. McIntosh's rates,
the escape rate out of fcc down to 75 K, are an independent test.
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
from examples.ruthenium.util import (
    CORRECTED_SCALE,
    MAX_ENERGY,
    MAX_WORKERS,
    RUTHENIUM,
    SUPERCELL,
    TEMPERATURES,
    print_activation_energy,
)
from examples.util import (
    FrictionFit,
    chi2,
    fast_rate,
    fit_frictions,
)

# HREELS linewidths bound gamma below ~4 ps^-1 (Townsend thesis Sec. 6.3.7)
GAMMAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # ps^-1
PES = {"DFT": 1.0, "corrected": CORRECTED_SCALE}


@cached(lambda scale: Path(f"ruthenium.hydrogen.{scale:.4f}.pkl"))
def fit_hydrogen(scale: float) -> FrictionFit:
    """Fit a single friction gamma to the measured H hcp escape rates on one PES."""
    return fit_frictions(
        RUTHENIUM.bands(M_H, scale, SUPERCELL["H"]),
        M_H,
        TEMPERATURES,
        H_RATES,
        GAMMAS,
        observable=fast_rate,
        max_energy=MAX_ENERGY,
        max_workers=MAX_WORKERS["H"],
    )


def _main() -> None:
    fig, ax = get_fancy_figure()
    _, _, points = plot_experimental_rates(MCINTOSH_H_RATES, ax=ax)
    points.set_label("McIntosh, slow (not fitted)")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.dark)
    _, _, points = plot_experimental_rates(H_RATES, ax=ax)
    points.set_label("Townsend, fast")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.warm)
    points.lines[0].set_markerfacecolor("none")  # fast rates: open circles
    for (name, scale), color in zip(
        PES.items(), (CAM_BLUE.dark, CAM_BLUE.warm), strict=True
    ):
        fit = fit_hydrogen(scale)
        curve, gamma = fit.curve, fit.curve.friction.gamma
        mcintosh_chi2 = chi2(MCINTOSH_H_RATES, curve.temperatures, curve.slow)
        print(
            f"H, {name} PES (splitting_scale {scale:.3f}): gamma = {gamma:.3f} ps^-1, "
            f"chi2 = {fit.chi2:.2f} (McIntosh, not fitted: {mcintosh_chi2:.2f})"
        )
        print_activation_energy("H", H_RATES, curve)
        _, _, slow = plot_rate_arrhenius(curve.temperatures, curve.slow, ax=ax)
        slow.set(
            color=color,
            label=rf"{name} PES, $\gamma = {gamma:.2f}$ ps$^{{-1}}$, "
            rf"$\chi^2 = {fit.chi2:.1f}$ (fast), {mcintosh_chi2:.1f} (slow)",
        )
        _, _, fast = plot_rate_arrhenius(curve.temperatures, curve.fast, ax=ax)
        fast.set(color=color, linestyle="--")
    ax.set_ylabel(r"Escape rate (ps$^{-1}$)")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("H/Ru(0001), single friction (solid: slow, dashed: fast)", fontsize=9)
    fig.savefig("examples/ruthenium/ruthenium.hydrogen.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/ruthenium/data")):
        _main()
