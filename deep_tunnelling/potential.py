"""H-Ni(111) potential energy surface.

The DFT data (M. Sacchi) is sampled at the six symmetry-distinct points of the
irreducible triangle (top, fcc, hcp, bridge, top-fcc, top-hcp) at 9 heights.
In each plane we fit the six lowest C3v-invariant Fourier "star" functions,
which reproduce the six site energies exactly, and interpolate the coefficients
in z with a cubic spline. Large repulsive energies are log-compressed before
fitting (and expanded afterwards) so that the top-site wall at small z does not
cause ringing in the Fourier series. The hcp hollow lies 16.9 meV above fcc.

Two optional corrections tune the PES to experiment (from_dft):
splitting_scale multiplies the fcc/hcp-antisymmetric (sine) stars, which moves
the hollows apart, and bridge_shift adds a star combination that is 1 at the
bridge and 0, with zero curvature, at both hollows (and 0 at the top site), so
it changes the barrier but not the hollow energies or harmonic frequencies.

Fractional coordinates (u, v) refer to r = u a1 + v a2 with
a1 = a (1, 0), a2 = a (1/2, sqrt(3)/2), the top Ni atom at the origin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

# the stars odd under the fcc <-> hcp mirror, which carry the hollow splitting
_SINE_STARS = [2, 5]
# |G|^2 of each star in units of |b|^2: each star is a single shell
_SHELLS = np.array([0.0, 1.0, 1.0, 3.0, 4.0, 4.0])


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


def _bridge_star() -> np.ndarray[Any, Any]:
    """Star coefficients of g: 1 at the bridge, 0 at top, fcc and hcp, zero Laplacian at fcc and hcp."""
    b2 = (4 * np.pi / (np.sqrt(3) * A_NI)) ** 2
    rows, rhs = [], []
    stars = {
        name: star_functions(np.array(u), np.array(v)) for name, (u, v) in SITES.items()
    }
    for site, value in (("bridge", 1.0), ("top", 0.0), ("fcc", 0.0), ("hcp", 0.0)):
        rows.append(stars[site])
        rhs.append(value)
    for site in ("fcc", "hcp"):
        rows.append(-b2 * _SHELLS * stars[site])
        rhs.append(0.0)
    return np.linalg.solve(np.array(rows, dtype=float), np.array(rhs))


@dataclass(frozen=True)
class SurfacePotential:
    """Smooth periodic H-Ni(111) PES built from the six-site DFT data."""

    z_points: np.ndarray[Any, Any]
    coefficients: np.ndarray[Any, Any]  # (6, nz) star coefficients of compressed V
    # (6,) star coefficients added to V (meV), independent of z
    correction: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(6))
    v_cap: float = 4000.0  # meV, energies above this are irrelevant for low states

    @classmethod
    def from_dft(
        cls, path: Path, *, splitting_scale: float = 1.0, bridge_shift: float = 0.0
    ) -> SurfacePotential:
        """Build the PES from the DFT site curves, with the optional corrections above.

        bridge_shift is in meV (negative lowers the barrier).
        """
        z, curves = load_site_curves(path)
        names = list(SITES)
        uv = np.array([SITES[n] for n in names])
        basis = star_functions(uv[:, 0], uv[:, 1]).T  # (site, star)
        values = np.array([_compress(curves[n]) for n in names])  # (site, nz)
        coefficients = np.linalg.solve(basis, values)
        coefficients[_SINE_STARS] *= splitting_scale
        return cls(
            z_points=z,
            coefficients=coefficients,
            correction=bridge_shift * _bridge_star(),
        )

    def __call__(
        self, u: np.ndarray[Any, Any], v: np.ndarray[Any, Any], z: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Evaluate V (meV) at fractional (u, v) and height z (A); arrays broadcast."""
        spline = scipy.interpolate.CubicSpline(self.z_points, self.coefficients, axis=1)
        u, v, z = np.broadcast_arrays(u, v, z)
        c = spline(z.ravel()).reshape(6, *z.shape)
        stars = star_functions(u, v)
        t = np.einsum("s...,s...->...", c, stars)
        return np.minimum(
            _expand(t) + np.einsum("s,s...->...", self.correction, stars), self.v_cap
        )
