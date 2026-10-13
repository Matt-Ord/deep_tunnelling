"""HeSE hop rates of H and D on Pt(111), Jardine et al., PRL 105, 136101 (2010).

Extracted from the vector graphic of Fig. 3 (0.1 ML, Delta K = 0.86 A^-1 along
[11-2]). The rate is the total hopping rate Gamma of a Chudley-Elliott fit to
nearest-neighbour fcc -> fcc jumps (jump length a = 2.77 A, D = a^2 Gamma / 4),
in ps^-1, with the +-1 sigma range. The temperature axis at the top of Fig. 3 is
mislabelled (evenly spaced ticks), so temperatures are read from 1000 / T.
Repeated temperatures are separate measurements. A D marker at 1000 / T = 9.45
without an error bar is omitted.

hcp is metastable on Pt(111), so the measured rate is fcc -> (hcp) -> fcc
hopping. hopping_rate maps the escape rates of the two-sublattice jump model
onto the Chudley-Elliott rate measured at the same Delta K.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from deep_tunnelling.model import RateCurve

Array = np.ndarray[Any, Any]

# columns: temperature in K, rate, lower and upper error bounds
H_RATES = np.array(
    [
        (230, 0.1443, 0.1346, 0.1540),
        (220, 0.1239, 0.1159, 0.1320),
        (210, 0.1339, 0.1257, 0.1422),
        (200, 0.1026, 0.0978, 0.1074),
        (190, 0.0943, 0.0903, 0.0984),
        (180, 0.0730, 0.0691, 0.0768),
        (170, 0.0592, 0.0562, 0.0623),
        (160, 0.0446, 0.0427, 0.0465),
        (150, 0.0334, 0.0320, 0.0348),
        (140, 0.0263, 0.0250, 0.0276),
        (140, 0.0269, 0.0258, 0.0280),
        (120, 0.0123, 0.0117, 0.0128),
        (105, 0.00585, 0.00558, 0.00614),
        (100, 0.00559, 0.00510, 0.00609),
        (95, 0.00435, 0.00403, 0.00467),
        (90, 0.00324, 0.00300, 0.00348),
        (90, 0.00390, 0.00357, 0.00422),
        (85, 0.00255, 0.00218, 0.00290),
        (80, 0.00158, 0.00107, 0.00210),
    ]
)

D_RATES = np.array(
    [
        (230, 0.1452, 0.1328, 0.1575),
        (220, 0.1232, 0.1139, 0.1328),
        (210, 0.1029, 0.0963, 0.1095),
        (200, 0.0754, 0.0680, 0.0828),
        (190, 0.0659, 0.0600, 0.0719),
        (180, 0.0523, 0.0480, 0.0567),
        (170, 0.0393, 0.0356, 0.0429),
        (160, 0.0325, 0.0304, 0.0347),
        (150, 0.0227, 0.0207, 0.0248),
        (120, 0.00823, 0.00775, 0.00873),
        (105, 0.00400, 0.00369, 0.00430),
        (100, 0.00316, 0.00262, 0.00370),
        (95, 0.00233, 0.00194, 0.00274),
        (90, 0.00178, 0.00150, 0.00206),
        (85, 0.00147, 0.00099, 0.00196),
    ]
)

DELTA_K = 0.86  # A^-1, along [11-2], perpendicular to the close-packed rows
A_PT = 2.7747  # A, Pt(111) surface lattice constant (bulk 3.924 A / sqrt 2)


def _phases(delta_k: float) -> tuple[Array, Array]:
    """K . d for the three fcc -> hcp vectors and the six fcc -> fcc jump vectors."""
    k = np.array([0.0, delta_k])  # [11-2] is perpendicular to a1 = a (1, 0)
    angles = np.pi / 6 + 2 * np.pi / 3 * np.arange(3)
    hollow = A_PT / np.sqrt(3) * np.array([np.cos(angles), np.sin(angles)]).T
    angles = np.pi / 3 * np.arange(6)
    jump = A_PT * np.array([np.cos(angles), np.sin(angles)]).T
    return hollow @ k, jump @ k


def hopping_rate(slow: Array, fast: Array, delta_k: float = DELTA_K) -> Array:
    """Chudley-Elliott fcc -> fcc rate Gamma with the same dephasing rate as the two-sublattice model.

    slow and fast are the escape rates out of fcc and hcp (each split equally over
    three neighbours). The slower root of the 2 x 2 jump matrix at Delta K is
    alpha = (S - sqrt(S^2 - 4 slow fast (1 - |F|^2))) / 2, S = slow + fast, with
    F = mean exp(i K . d) over the three fcc -> hcp vectors; Chudley-Elliott on
    the fcc lattice gives alpha = (Gamma / 3) sum_j sin^2(K . l_j / 2).
    For fast >> slow, Gamma -> 2 slow / 3: a third of the hcp visits return.
    """
    slow, fast = np.asarray(slow), np.asarray(fast)
    hollow, jump = _phases(delta_k)
    f2 = np.abs(np.mean(np.exp(1j * hollow))) ** 2
    total = slow + fast
    alpha = 0.5 * (total - np.sqrt(total**2 - 4 * slow * fast * (1 - f2)))
    return alpha / (np.sum(np.sin(jump / 2) ** 2) / 3)


def jump_rate(curve: RateCurve) -> Array:
    """Return the measured quantity for a model rate curve: the Chudley-Elliott fcc -> fcc rate."""
    return hopping_rate(curve.slow, curve.fast)
