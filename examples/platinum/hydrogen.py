"""H on Pt(111): a single friction fitted to Jardine's rates, on the DFT and the corrected PES.

On the DFT PES the hcp ground state of H lies 37 meV above fcc. The corrected PES
lowers it by scaling the fcc/hcp-odd stars (util.SCALES); gamma is fitted at each
scale and the corrected PES is the scale with the lowest chi^2. This improves the
fit and brings gamma down, but not the low-temperature slope.

The measured rate is the Chudley-Elliott fcc -> fcc rate at Delta K = 0.86 A^-1
(experiment.jump_rate).
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
from examples.platinum.experiment import H_RATES, jump_rate
from examples.platinum.util import (
    MAX_ENERGY,
    MAX_WORKERS,
    PLATINUM,
    SCALES,
    TEMPERATURES,
    print_activation_energies,
)
from examples.util import FrictionFit, fit_frictions

GAMMAS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]  # ps^-1


@cached(lambda scale: Path(f"platinum.hydrogen.{scale:.4f}.pkl"))
def fit_hydrogen(scale: float) -> FrictionFit:
    """Fit a single friction gamma to the measured H rates on one PES."""
    return fit_frictions(
        PLATINUM.bands(M_H, scale),
        M_H,
        TEMPERATURES,
        H_RATES,
        GAMMAS,
        observable=jump_rate,
        max_energy=MAX_ENERGY["H"],
        max_workers=MAX_WORKERS["H"],
    )


def corrected_scale() -> float:
    """Return the splitting_scale in SCALES with the lowest H chi^2."""
    best = min(SCALES, key=lambda s: fit_hydrogen(s).chi2)
    if best in {SCALES[0], SCALES[-1]}:
        print(f"warning: best splitting_scale {best} on the edge of the grid")
    return best


def _main() -> None:
    for scale in SCALES:
        fit = fit_hydrogen(scale)
        print(
            f"splitting_scale {scale:.2f}: gamma = {fit.curve.friction.gamma:.3f} "
            f"ps^-1, chi2 = {fit.chi2:.2f}"
        )
    fig, ax = get_fancy_figure()
    _, _, points = plot_experimental_rates(H_RATES, ax=ax)
    points.set_label("Jardine, H")
    for artist in points.lines[0], *points.lines[2]:
        artist.set_color(CAM_CHERRY.dark)
    for (name, scale), color in zip(
        (("DFT", 1.0), ("corrected", corrected_scale())),
        (CAM_BLUE.dark, CAM_BLUE.warm),
        strict=True,
    ):
        fit = fit_hydrogen(scale)
        print(f"{name} PES (splitting_scale {scale:.2f}):")
        print_activation_energies("H", H_RATES, fit.curve)
        _, _, line = plot_rate_arrhenius(
            fit.curve.temperatures, jump_rate(fit.curve), ax=ax
        )
        line.set(
            color=color,
            label=rf"{name} PES ({scale:.1f}), $\gamma = {fit.curve.friction.gamma:.2f}$"
            rf" ps$^{{-1}}$, $\chi^2 = {fit.chi2:.1f}$",
        )
    ax.set_ylabel(r"Hopping rate $\Gamma$ (ps$^{-1}$)")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("H/Pt(111), single friction (Chudley-Elliott rate)", fontsize=9)
    fig.savefig("examples/platinum/platinum.hydrogen.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/platinum/data")):
        _main()
