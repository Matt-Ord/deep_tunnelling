"""HeSE hop rates of H and D on Ni(111), J. Zhu PhD thesis (Cambridge, 2015).

Digitized from Fig. 4.21 (0.07 ML). The rate is the 'slow' rate, i.e. the total
escape rate out of the lower-energy (fcc) hollow, in ps^-1, with the +-1 sigma
range. Repeated temperatures are separate measurements.

The H fast rate (out of hcp) is the slow rate times the fast/slow ratio lambda
fitted at each measurement (Fig. 4.20, extracted from the vector graphic; its y
axis is log10(lambda), though labelled ln). Fig. 4.20 has no error bars, so the
fast-rate range is the slow-rate range times lambda.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from deep_tunnelling.plot import get_figure, set_arrhenius_axes
from deep_tunnelling.units import KB

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure

Array = np.ndarray[Any, Any]

# columns: temperature in K, rate, lower and upper error bounds
H_RATES = np.array(
    [
        (255, 0.0237, 0.0228, 0.0245),
        (245, 0.0166, 0.0157, 0.0174),
        (235, 0.0145, 0.0137, 0.0153),
        (230, 0.0144, 0.0137, 0.0152),
        (225, 0.0117, 0.0110, 0.0125),
        (220, 0.0105, 0.0097, 0.0112),
        (210, 0.0099, 0.0088, 0.0110),
        (205, 0.0069, 0.0067, 0.0071),
        (200, 0.0064, 0.0061, 0.0067),
        (195, 0.0048, 0.0046, 0.0051),
        (190, 0.0048, 0.0043, 0.0052),
        (185, 0.0037, 0.0034, 0.0040),
        (180, 0.0039, 0.0034, 0.0043),
        (175, 0.0029, 0.0028, 0.0031),
        (165, 0.0032, 0.0027, 0.0038),
        (160, 0.0030, 0.0029, 0.0032),
        (155, 0.0030, 0.0027, 0.0033),
        (145, 0.0025, 0.0019, 0.0032),
        (145, 0.0024, 0.0019, 0.0032),
        (135, 0.0034, 0.0029, 0.0039),
        (135, 0.0030, 0.0029, 0.0039),
        (130, 0.0039, 0.0030, 0.0047),
        (125, 0.0026, 0.0015, 0.0037),
        (120, 0.0044, 0.0037, 0.0051),
    ]
)

D_RATES = np.array(
    [
        (230, 0.00614, 0.0059, 0.0064),
        (225, 0.00533, 0.0051, 0.0056),
        (215, 0.00429, 0.0041, 0.0045),
        (210, 0.00344, 0.0032, 0.0037),
        (205, 0.00319, 0.0030, 0.0034),
        (200, 0.00259, 0.0024, 0.0028),
        (190, 0.00221, 0.0020, 0.0024),
        (180, 0.00168, 0.0015, 0.0019),
        (175, 0.00161, 0.0014, 0.0019),
        (165, 0.00148, 0.0013, 0.0017),
        (158, 0.00155, 0.0014, 0.0017),
        (155, 0.00148, 0.0012, 0.0018),
        (153, 0.00138, 0.0011, 0.0017),
        (150, 0.00131, 0.0011, 0.0015),
        (148, 0.00148, 0.0013, 0.0017),
    ]
)

# fast rates of the same measurements, columns as H_RATES
H_FAST_RATES = np.array(
    [
        (255, 0.0540, 0.0520, 0.0559),
        (245, 0.0397, 0.0375, 0.0416),
        (235, 0.0370, 0.0349, 0.0390),
        (230, 0.0395, 0.0375, 0.0416),
        (225, 0.0311, 0.0293, 0.0333),
        (220, 0.0284, 0.0262, 0.0302),
        (210, 0.0283, 0.0252, 0.0315),
        (205, 0.0204, 0.0198, 0.0210),
        (200, 0.0185, 0.0176, 0.0194),
        (195, 0.0155, 0.0148, 0.0164),
        (190, 0.0148, 0.0133, 0.0161),
        (185, 0.0122, 0.0112, 0.0132),
        (180, 0.0128, 0.0111, 0.0141),
        (175, 0.0106, 0.0102, 0.0113),
        (165, 0.0120, 0.0101, 0.0142),
        (160, 0.0115, 0.0111, 0.0122),
        (155, 0.0121, 0.0109, 0.0133),
        (145, 0.0110, 0.0084, 0.0141),
        (145, 0.0107, 0.0085, 0.0143),
        (135, 0.0168, 0.0143, 0.0192),
        (135, 0.0148, 0.0143, 0.0192),
        (130, 0.0209, 0.0161, 0.0252),
        (125, 0.0146, 0.0084, 0.0208),
        (120, 0.0271, 0.0228, 0.0314),
    ]
)

# fcc-hcp energy difference from the T dependence of the fast/slow ratio, meV;
# implies the fast rates of D, for which the ratio was not measured
DELTA_E_HOLLOW = 19.0


def plot_experimental_rates(
    data: Array, *, ax: Axes | None = None
) -> tuple[Figure, Axes, ErrorbarContainer]:
    """Plot Zhu's HeSE rates (columns T, rate, low, high) against 1000 / T."""
    fig, ax = get_figure(ax)
    t, r, lo, hi = np.asarray(data).T
    container = ax.errorbar(
        1000 / t, r, yerr=[r - lo, hi - r], linestyle="", marker="o", markersize=4
    )
    set_arrhenius_axes(ax)
    return fig, ax, container


def implied_fast_rates(data: Array) -> Array:
    """Slow rates (and ranges) times exp(Delta E / kT), where the fast rate was not measured."""
    data = np.array(data, dtype=float)
    data[:, 1:] *= np.exp(DELTA_E_HOLLOW / (KB * data[:, :1]))
    return data
