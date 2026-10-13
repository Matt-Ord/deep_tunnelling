"""Shared helpers for the Ru(0001) examples: the DFT PES and Bloch states.

The PES is T. Wikfeldt's DFT calculation (7 March 2012, Thor_07-03-2012.mat): 31
points of the irreducible triangle at 31 heights, lattice constant 2.7276 A. In
raw_data.json the energies are those of the .mat (eV; an earlier JSON conversion
flipped their sign), heights below 22.17 A are dropped (H at or below the Ru
nuclei: the top site is singular there and the hollows open into the subsurface)
and z is measured from the fcc minimum at 23.07 A, leaving -0.9 to 2.4 A.

The hcp site lies 63 meV above fcc and the H hcp ground state 53 meV above the
fcc one, against 22 meV measured. The corrected PES scales the fcc/hcp-odd stars
to reproduce the measured splitting (as for Ni(111), THEORY.md 7.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from deep_tunnelling.bloch import Grid
from examples.util import Surface, activation_energy

if TYPE_CHECKING:
    from deep_tunnelling.model import RateCurve

Array = np.ndarray[Any, Any]

# H energies at Gamma change by < 0.03 meV with n_z = 14, z up to 2 A or a 24 x 24 grid
RUTHENIUM = Surface(
    raw_data=Path(__file__).parent / "raw_data.json",
    grid=Grid(a=2.7276, z_min=-0.9, z_max=1.6, nz=32),
    n_z=10,
)

# splitting_scale with the H hcp ground state 22 meV above fcc (D: 23 meV)
CORRECTED_SCALE = 0.411
TEMPERATURES = np.linspace(75.0, 300.0, 16)
# bands in the master equation, supercell and worker processes. Memory per worker
# grows as n^2 n_rate^4; D (19 bands) runs on n = 4, within 0.5% of n = 6. At
# 150 K / 300 K, 150 meV is within 0.7% / 3% (H, 11 bands) of 180 meV and within
# 0.3% / 1.5% (D) of 165 meV
MAX_ENERGY = 150.0  # meV
SUPERCELL = {"H": 6, "D": 4}
MAX_WORKERS = {"H": 4, "D": 2}

# Arrhenius range of Townsend's activation energies (thesis Sec. 6.3.5)
HIGH_T = (200.0, 300.0)


def print_activation_energy(name: str, data: Array, curve: RateCurve) -> None:
    """Print the measured and model activation energies of the hcp escape rate in HIGH_T."""
    lo, hi = HIGH_T
    t, r, r_lo, r_hi = data[(data[:, 0] >= lo) & (data[:, 0] <= hi)].T
    measured = activation_energy(t, r, 0.5 * np.log(r_hi / r_lo))
    model_t = np.linspace(lo, hi, 20)
    model = activation_energy(
        model_t, np.exp(np.interp(model_t, curve.temperatures, np.log(curve.fast)))
    )
    print(
        f"  {name} {lo:.0f}-{hi:.0f} K: Ea measured {measured:5.1f} meV, model {model:5.1f} meV"
    )
