"""H on Ni(111): fit the friction to Zhu's rates and plot the slow and fast escape rates.

Two PESs are compared: 'symmetric', with degenerate fcc and hcp hollows (the best
model), and 'raw', the unmodified DFT with hcp 16.9 meV above fcc, where the slow
and fast rates differ.
"""

from pathlib import Path

import numpy as np
import scipy.interpolate
from experiment import H_RATES, RAW_DATA, FrictionFit, chi2, plot_against_experiment

from deep_tunnelling.bloch import BlochSolver
from deep_tunnelling.model import rates
from deep_tunnelling.potential import SurfacePotential
from deep_tunnelling.ume import BandData
from deep_tunnelling.units import M_H
from deep_tunnelling.util import cache_base_path, cached

GAMMAS = [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]  # ps^-1
TEMPERATURES = np.linspace(115.0, 260.0, 10)


@cached(
    lambda *, degenerate: Path(f"hydrogen.{'symmetric' if degenerate else 'raw'}.pkl")
)
def fit_hydrogen(*, degenerate: bool) -> FrictionFit:
    """Fit gamma on a log-log spline through a grid of frictions, then evaluate the best fit."""
    potential = SurfacePotential.from_dft(RAW_DATA, degenerate=degenerate)
    bands = BandData.build(BlochSolver(potential, M_H), n=6)
    slow = np.array(
        [curve.slow for curve in rates(bands, M_H, TEMPERATURES, GAMMAS)]
    )  # (gamma, T)
    spline = scipy.interpolate.CubicSpline(np.log(GAMMAS), np.log(slow), axis=0)
    trial = np.linspace(np.log(GAMMAS[0]), np.log(GAMMAS[-1]), 2001)
    gamma = float(
        np.exp(
            trial[
                np.argmin(
                    [chi2(H_RATES, TEMPERATURES, np.exp(spline(x))) for x in trial]
                )
            ]
        )
    )
    (curve,) = rates(bands, M_H, TEMPERATURES, [gamma])
    return FrictionFit(curve=curve, chi2=chi2(H_RATES, TEMPERATURES, curve.slow))


def _main() -> None:
    for degenerate, title in (
        (True, "symmetric hollows"),
        (False, "raw DFT, hcp +16.9 meV"),
    ):
        fit = fit_hydrogen(degenerate=degenerate)
        name = "symmetric" if degenerate else "raw"
        print(f"H, {name}: gamma = {fit.curve.gamma:.3f} ps^-1, chi2 = {fit.chi2:.1f}")
        plot_against_experiment(
            fit.curve,
            H_RATES,
            rf"H/Ni(111), {title}, $\chi^2 = {fit.chi2:.1f}$",
            f"examples/hydrogen.rates.{name}.pdf",
        )


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _main()
