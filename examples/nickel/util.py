"""Shared helpers for the Ni(111) examples: the DFT PES and its correction."""

from __future__ import annotations

from pathlib import Path

from deep_tunnelling.bloch import Grid
from examples.util import Surface

# DFT PES of H on Ni(111) (M. Sacchi), six sites x nine heights, -1 to 1.2 A
# about the fcc minimum; a is the Ni-Ni distance of the slab
NICKEL = Surface(
    raw_data=Path(__file__).parent / "raw_data.json", grid=Grid(a=2.4890158697766473)
)

# splitting_scale with the H hcp ground state at the measured 19.0 meV above fcc
# (DFT: 13.2 meV). The DFT barrier needs no correction: the activation energy of
# the H slow rate above 170 K is then 95.5 meV, against 97.7 +- 5.2 meV measured
# (THEORY.md 7.2)
OPTIMIZED_SCALE = 1.439
PES = {"DFT": 1.0, "optimized": OPTIMIZED_SCALE}

# worker processes (memory per worker grows as n^2 n_rate^4: H 10 bands, D 16)
MAX_WORKERS = {"H": 4, "D": 1}
