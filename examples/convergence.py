"""Convergence of the H hop rate in the supercell size n (n x n k-points)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from experiment import RAW_DATA

from deep_tunnelling.bloch import BlochSolver
from deep_tunnelling.model import rates
from deep_tunnelling.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
    plot_convergence,
)
from deep_tunnelling.potential import SurfacePotential
from deep_tunnelling.ume import BandData
from deep_tunnelling.units import M_H
from deep_tunnelling.util import cache_base_path, cached

SIZES = [
    2,
    3,
    4,
    6,
    8,
    10,
]  # n = 1 couples with q = a full reciprocal vector and is skipped
TEMPERATURES = np.array([120.0, 160.0, 250.0])
GAMMA = 1.1  # ps^-1, the H best fit


@dataclass(frozen=True, kw_only=True)
class ConvergenceResult:
    """Slow escape rate for increasing supercell size n."""

    gamma: float
    sizes: list[int]
    temperatures: np.ndarray
    rates: np.ndarray  # (len(sizes), len(temperatures))


@cached(Path("convergence.pkl"))
def solve_convergence() -> ConvergenceResult:
    potential = SurfacePotential.from_dft(RAW_DATA)
    bands = [BandData.build(BlochSolver(potential, M_H), n) for n in SIZES]
    table = np.array([rates(b, M_H, TEMPERATURES, [GAMMA])[0].slow for b in bands])
    return ConvergenceResult(
        gamma=GAMMA, sizes=SIZES, temperatures=TEMPERATURES, rates=table
    )


def _plot_convergence(result: ConvergenceResult) -> None:
    fig, ax = get_fancy_figure()
    lines = []
    colors = [CAM_BLUE.dark, CAM_BLUE.warm, CAM_CHERRY.dark]
    for j, (temperature, color) in enumerate(
        zip(result.temperatures, colors, strict=True)
    ):
        _, _, line = plot_convergence(result.sizes, result.rates[:, j], ax=ax)
        line.set_color(color)
        line.set_label(f"{temperature:.0f} K")
        lines.append(line)
    ax.axhspan(0.99, 1.01, color="white", zorder=0)
    ax.legend(handles=lines)
    fig.savefig("examples/convergence.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _plot_convergence(solve_convergence())
