"""Unit system used throughout: energy meV, time ps, length Angstrom."""

from __future__ import annotations

import scipy.constants as sc

MEV = 1e-3 * sc.electron_volt
HBAR = sc.hbar / (MEV * 1e-12)  # meV ps
KB = sc.Boltzmann / MEV  # meV / K
# mass unit meV ps^2 / A^2
_MASS_UNIT = MEV * 1e-24 / 1e-20
M_H = 1.00782503207 * sc.atomic_mass / _MASS_UNIT
M_D = 2.01410177812 * sc.atomic_mass / _MASS_UNIT
