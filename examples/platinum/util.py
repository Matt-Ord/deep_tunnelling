"""Shared helpers for the Pt(111) examples: the DFT PES and Bloch states.

The PES is M. Sacchi's DFT calculation (31 May 2013): 31 points of the
irreducible triangle at 18 heights, -1 to 1.5 A about the fcc minimum, processed
by J. Ellis (raw_data.json). The top site, 46 meV above fcc, sits 0.66 A higher
than the hollows.

The corrected PES scales the fcc/hcp-odd stars by splitting_scale (SCALES), which
moves the hcp hollow relative to fcc (THEORY.md 7.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from deep_tunnelling.bloch import Grid
from examples.platinum.experiment import jump_rate
from examples.util import Surface, activation_energy

if TYPE_CHECKING:
    from deep_tunnelling.model import RateCurve

Array = np.ndarray[Any, Any]

# the slab's surface lattice constant is 2.809 A (the fcc site is at x = a / 2); z
# covers the data. The top-site states need more z states than the hollows (10
# match the full 32-point basis to 0.1 meV for the hollow bands, 0.7 meV for top)
PLATINUM = Surface(
    raw_data=Path(__file__).parent / "raw_data.json",
    grid=Grid(a=2.809, z_min=-1.0, z_max=1.5, nz=32),
    n_z=10,
)

# splitting_scale: 1 is the DFT PES; 0.7 lowers the hcp site from 46 to 39 meV and
# the H hcp ground state from 37 to 26 meV
SCALES = [0.6, 0.7, 0.8, 0.9, 1.0]
TEMPERATURES = np.linspace(80.0, 235.0, 12)
# bands in the master equation and worker processes: memory per worker grows as
# n_rate^4 (THEORY.md 5.1)
# H: 9-11 bands, within 4% of 120 meV; D: 12-13 bands, within 1.2% of 100 meV
MAX_ENERGY = {"H": 100.0, "D": 90.0}  # meV
MAX_WORKERS = {"H": 4, "D": 2}

# Arrhenius ranges of the two regimes of the data
REGIMES = {"80-105 K": (80.0, 105.0), "150-230 K": (150.0, 230.0)}


def print_activation_energies(name: str, data: Array, curve: RateCurve) -> None:
    """Print the measured and model activation energies in the two REGIMES."""
    for label, (lo, hi) in REGIMES.items():
        t, r, r_lo, r_hi = data[(data[:, 0] >= lo) & (data[:, 0] <= hi)].T
        measured = activation_energy(t, r, 0.5 * np.log(r_hi / r_lo))
        model_t = np.linspace(lo, hi, 20)
        model = activation_energy(
            model_t,
            np.exp(np.interp(model_t, curve.temperatures, np.log(jump_rate(curve)))),
        )
        print(
            f"  {name} {label}: Ea measured {measured:5.1f} meV, model {model:5.1f} meV"
        )
