"""H-Ni(111) potential energy surface.

The DFT data (M. Sacchi) is sampled at the six symmetry-distinct points of the
irreducible triangle (top, fcc, hcp, bridge, top-fcc, top-hcp) at 9 heights.
In each plane we fit the six lowest C3v-invariant Fourier "star" functions,
which reproduce the six site energies exactly, and interpolate the coefficients
in z with a cubic spline. Large repulsive energies are log-compressed before
fitting (and expanded afterwards) so that the top-site wall at small z does not
cause ringing in the Fourier series.

The best description of the HeSE rates is obtained with degenerate hollows:
the hcp curve is replaced by the fcc one (and top-hcp by top-fcc), leaving the
barrier, vibrations and z dependence as in the DFT.

Fractional coordinates (u, v) refer to r = u a1 + v a2 with
a1 = a (1, 0), a2 = a (1/2, sqrt(3)/2), the top Ni atom at the origin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.interpolate

from deep_tunnelling.units import A_NI, MEV

if TYPE_CHECKING:
    from pathlib import Path

# fractional coordinates of the six sampled sites
SITES: dict[str, tuple[float, float]] = {
    "top": (0.0, 0.0),
    "fcc": (1 / 3, 1 / 3),
    "hcp": (2 / 3, 2 / 3),
    "bridge": (0.0, 0.5),
    "top_fcc": (1 / 6, 1 / 6),
    "top_hcp": (5 / 6, 1 / 3),
}

# compression scale for the log transform, meV
_B = 2500.0


def _compress(v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return _B * np.log1p(v / _B)


def _expand(t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return _B * np.expm1(t / _B)


def star_functions(
    u: np.ndarray[Any, Any], v: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Return the six lowest C3v-invariant plane-wave stars, shape (6, *u.shape)."""
    p = 2 * np.pi
    return np.array(
        [
            np.ones_like(u),
            np.cos(p * u) + np.cos(p * v) + np.cos(p * (u + v)),
            np.sin(p * u) + np.sin(p * v) - np.sin(p * (u + v)),
            np.cos(p * (u - v)) + np.cos(p * (2 * u + v)) + np.cos(p * (u + 2 * v)),
            np.cos(2 * p * u) + np.cos(2 * p * v) + np.cos(2 * p * (u + v)),
            np.sin(2 * p * u) + np.sin(2 * p * v) - np.sin(2 * p * (u + v)),
        ]
    )


def _site_of(x: float, y: float) -> str:
    v = y / (A_NI * np.sqrt(3) / 2)
    u = x / A_NI - v / 2
    for name, (su, sv) in SITES.items():
        du, dv = (u - su) % 1, (v - sv) % 1
        if min(du, 1 - du) < 1e-3 and min(dv, 1 - dv) < 1e-3:  # noqa: PLR2004
            return name
    msg = f"unknown site at {x}, {y}"
    raise ValueError(msg)


def load_site_curves(
    path: Path,
) -> tuple[np.ndarray[Any, Any], dict[str, np.ndarray[Any, Any]]]:
    """Load raw DFT data as V(z) at each site, in meV relative to the global minimum, z in A."""
    data = json.loads(path.read_text(encoding="utf-8"))
    x = np.asarray(data["x_points"]) * 1e10
    y = np.asarray(data["y_points"]) * 1e10
    z = np.asarray(data["z_points"]) * 1e10
    e = np.asarray(data["points"]) / MEV
    e -= e.min()
    z_c = np.unique(z)
    curves: dict[str, np.ndarray[Any, Any]] = {}
    for xi, yi in sorted(set(zip(x.tolist(), y.tolist(), strict=True))):
        mask = (x == xi) & (y == yi)
        order = np.argsort(z[mask])
        curves[_site_of(xi, yi)] = e[mask][order]
    return z_c, curves


@dataclass(frozen=True)
class SurfacePotential:
    """Smooth periodic H-Ni(111) PES built from the six-site DFT data."""

    z_points: np.ndarray[Any, Any]
    coefficients: np.ndarray[Any, Any]  # (6, nz) star coefficients of compressed V
    v_cap: float = 4000.0  # meV, energies above this are irrelevant for low states

    @classmethod
    def from_dft(cls, path: Path, *, degenerate: bool = True) -> SurfacePotential:
        """Build the PES from the DFT site curves.

        degenerate=True (the default, best fit to experiment) makes the fcc and
        hcp hollows equivalent; degenerate=False keeps the raw DFT, where hcp
        lies 16.9 meV above fcc.
        """
        z, curves = load_site_curves(path)
        if degenerate:
            curves = {**curves, "hcp": curves["fcc"], "top_hcp": curves["top_fcc"]}
        names = list(SITES)
        uv = np.array([SITES[n] for n in names])
        basis = star_functions(uv[:, 0], uv[:, 1]).T  # (site, star)
        values = np.array([_compress(curves[n]) for n in names])  # (site, nz)
        coefficients = np.linalg.solve(basis, values)
        return cls(z_points=z, coefficients=coefficients)

    def __call__(
        self, u: np.ndarray[Any, Any], v: np.ndarray[Any, Any], z: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Evaluate V (meV) at fractional (u, v) and height z (A); arrays broadcast."""
        spline = scipy.interpolate.CubicSpline(self.z_points, self.coefficients, axis=1)
        u, v, z = np.broadcast_arrays(u, v, z)
        c = spline(z.ravel()).reshape(6, *z.shape)
        t = np.einsum("s...,s...->...", c, star_functions(u, v))
        return np.minimum(_expand(t), self.v_cap)
