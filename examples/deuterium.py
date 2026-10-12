"""D on Ni(111): same PES and friction coefficient eta = m gamma as H, no refitting.

Electron-hole friction acts through eta, which does not depend on the adsorbate
mass, so gamma_D = gamma_H m_H / m_D. Uses the cached H fits from hydrogen.py.
"""

from pathlib import Path

import numpy as np
from experiment import D_RATES, RAW_DATA, plot_against_experiment
from hydrogen import fit_hydrogen

from deep_tunnelling.bloch import BlochSolver
from deep_tunnelling.model import RateCurve, rates
from deep_tunnelling.potential import SurfacePotential
from deep_tunnelling.ume import BandData
from deep_tunnelling.units import M_D
from deep_tunnelling.util import cache_base_path, cached

TEMPERATURES = np.linspace(115.0, 260.0, 8)


@cached(
    lambda gamma, *, degenerate: Path(
        f"deuterium.{'symmetric' if degenerate else 'raw'}.{gamma:.6f}.pkl"
    )
)
def solve_deuterium(gamma: float, *, degenerate: bool) -> RateCurve:
    potential = SurfacePotential.from_dft(RAW_DATA, degenerate=degenerate)
    (curve,) = rates(
        BandData.build(BlochSolver(potential, M_D), n=6), M_D, TEMPERATURES, [gamma]
    )
    return curve


def _main() -> None:
    for degenerate, title in (
        (True, "symmetric hollows"),
        (False, "raw DFT, hcp +16.9 meV"),
    ):
        gamma = (
            fit_hydrogen(degenerate=degenerate).curve.eta / M_D
        )  # same eta as H on this PES
        curve = solve_deuterium(gamma, degenerate=degenerate)
        name = "symmetric" if degenerate else "raw"
        print(f"D, {name}: gamma = {gamma:.3f} ps^-1 (same eta as H)")
        plot_against_experiment(
            curve,
            D_RATES,
            rf"D/Ni(111), {title}, same $\eta = m\gamma$ as H",
            f"examples/deuterium.rates.{name}.pdf",
        )


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _main()
