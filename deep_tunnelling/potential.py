"""Potential energy surface of H on a close-packed metal surface, interpolated from DFT.

The DFT energies are sampled at a few in-plane points of the irreducible triangle
(H/Ni(111): 6 sites; H/Pt(111), H/Ru(0001): 31 points), each at a set of heights.
They are interpolated with C3v-invariant plane-wave "stars" (symmetric_stars), with
z measured from the local equilibrium height h(u, v):

    V(u, v, z) = expand(sum_s c_s(z - h(u, v)) f_s(u, v)),

where h is a star fit of the height of the minimum at each point and c_s(dz) is a
cubic spline of the star coefficients of the compressed energies at each dz. The
height matters because H sits ~0.6 A higher on the top site than in the hollows:
at fixed z the hollow wells lie next to the multi-eV wall of the top site, which
rings through the Fourier series (into spurious minima, for 31 points). Energies
are capped at v_cap and log-compressed before fitting for the same reason.

One correction tunes the PES to experiment: splitting_scale multiplies the stars
odd under the fcc <-> hcp mirror, which moves the hollows apart (> 1) or together
(< 1) and leaves the barrier and the top site unchanged.

Fractional coordinates (u, v) refer to r = u a1 + v a2 with a1 = a (1, 0),
a2 = a (1/2, sqrt(3)/2), a top atom at the origin, fcc at (1/3, 1/3) and hcp at
(2/3, 2/3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.interpolate

from deep_tunnelling.units import MEV

if TYPE_CHECKING:
    from pathlib import Path

Array = np.ndarray[Any, Any]

# compression scale for the log transform, meV
_B = 2500.0


def _compress(v: Array) -> Array:
    return _B * np.log1p(v / _B)


def _expand(t: Array) -> Array:
    return _B * np.expm1(t / _B)


# a star: "cos" or "sin" and the orbit of reciprocal indices (m, n), G = m b1 + n b2
Star = tuple[str, tuple[tuple[int, int], ...]]

# C3v about the top site acting on fractional (u, v): the 120 degree rotation
# (a1 -> a2 - a1, a2 -> -a1) and the mirror u <-> v through top, fcc and hcp
C3V = [
    np.linalg.matrix_power(np.array([[-1, -1], [1, 0]]), k) @ m
    for k in range(3)
    for m in (np.eye(2, dtype=int), np.array([[0, 1], [1, 0]]))
]


def _shell(star: Star) -> int:
    """|G|^2 of a star in units of |b|^2 (b1 . b2 = -|b|^2 / 2)."""
    m, n = star[1][0]
    return m * m + n * n - m * n


def symmetric_stars(count: int) -> list[Star]:
    """Return the count lowest C3v-invariant stars, ordered by |G|^2 and cosine before sine.

    Sine stars are odd under the fcc <-> hcp mirror.
    """
    n_max = int(np.sqrt(count)) + 2
    seen: set[tuple[int, int]] = set()
    stars: list[Star] = []
    for m in range(-n_max, n_max + 1):
        for n in range(-n_max, n_max + 1):
            if (m, n) in seen:
                continue
            # G transforms with the inverse transpose of the action on (u, v)
            images = (np.linalg.inv(g).T.round() @ (m, n) for g in C3V)
            orbit = tuple(sorted({(int(a), int(b)) for a, b in images}))
            seen |= set(orbit) | {(-a, -b) for a, b in orbit}
            stars.append(("cos", orbit))
            if (-m, -n) not in orbit:  # otherwise the sine sum vanishes
                stars.append(("sin", orbit))
    stars.sort(key=lambda s: (_shell(s), s[0]))
    return stars[:count]


def evaluate_stars(stars: list[Star], u: Array | float, v: Array | float) -> Array:
    """Values of the stars at fractional (u, v), shape (len(stars), *u.shape)."""
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    out = np.zeros((len(stars), *u.shape))
    for i, (kind, orbit) in enumerate(stars):
        trig = np.cos if kind == "cos" else np.sin
        for m, n in orbit:
            out[i] += trig(2 * np.pi * (m * u + n * v))
    return out


def load_points(path: Path, a: float) -> tuple[Array, Array, Array]:
    """Load DFT energies at in-plane points (raw_data.json format, SI units).

    Returns fractional (u, v) of each point (shape (n, 2), top site at the origin,
    lattice constant a in A), the heights z (A) and V (meV, (n, nz)) relative to
    the global minimum.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    x = np.asarray(data["x_points"]) * 1e10
    y = np.asarray(data["y_points"]) * 1e10
    z = np.asarray(data["z_points"]) * 1e10
    e = np.asarray(data["points"]) / MEV
    e -= e.min()
    xy = sorted(set(zip(x.tolist(), y.tolist(), strict=True)))
    z_c = np.unique(z)
    energies = np.empty((len(xy), len(z_c)))
    for i, (xi, yi) in enumerate(xy):
        mask = (x == xi) & (y == yi)
        energies[i] = e[mask][np.argsort(z[mask])]
    xy_a = np.array(xy)
    v = xy_a[:, 1] / (a * np.sqrt(3) / 2)
    return np.array([xy_a[:, 0] / a - v / 2, v]).T, z_c, energies


def _minimum(spline: scipy.interpolate.CubicSpline, z: Array) -> float:
    """Height of the minimum of a spline V(z) within the data range."""
    fine = np.linspace(z[0], z[-1], 2501)
    return float(fine[np.argmin(spline(fine))])


@dataclass(frozen=True, kw_only=True)
class SurfacePotential:
    """Periodic PES fitted to DFT energies at in-plane points (module docstring)."""

    stars: list[Star]
    height_stars: list[Star]
    dz_points: Array
    coefficients: Array  # (len(stars), len(dz_points)), of the compressed V
    height_coefficients: Array  # (len(height_stars),)
    v_cap: float = 4000.0  # meV, energies above this are irrelevant for the low states

    @classmethod
    def from_points(  # noqa: PLR0913
        cls,
        uv: Array,
        z: Array,
        energies: Array,
        *,
        n_stars: int = 25,
        n_height_stars: int = 20,
        ridge: float = 1e-7,
        dz_step: float = 0.05,
        v_cap: float = 4000.0,
        splitting_scale: float = 1.0,
    ) -> SurfacePotential:
        """Fit the stars to energies (meV, (n_points, nz)) at fractional uv and heights z (A).

        At most one star per point is used, so six sites are fitted exactly.
        ridge damps high stars by ridge |G|^4.
        """
        n_stars, n_height_stars = min(n_stars, len(uv)), min(n_height_stars, len(uv))
        splines = [scipy.interpolate.CubicSpline(z, row) for row in energies]
        heights = np.array([_minimum(s, z) for s in splines])
        # every dz covered by the data at some point; beyond its own data each
        # point is held at its end value
        lo, hi = z[0] - heights.max(), z[-1] - heights.min()
        dz = np.linspace(lo, hi, round((hi - lo) / dz_step) + 1)
        stars, height_stars = symmetric_stars(n_stars), symmetric_stars(n_height_stars)
        basis = evaluate_stars(stars, uv[:, 0], uv[:, 1]).T
        height_basis = evaluate_stars(height_stars, uv[:, 0], uv[:, 1]).T
        height_coefficients = np.linalg.lstsq(height_basis, heights)[0]
        # energies above v_cap are irrelevant and only drive ringing: cap, then compress
        values = np.array(
            [
                _compress(np.minimum(s(np.clip(dz + h, z[0], z[-1])), v_cap))
                for s, h in zip(splines, heights, strict=True)
            ]
        )
        shells = np.array([_shell(s) for s in stars], dtype=float)
        coefficients = np.linalg.solve(
            basis.T @ basis + ridge * np.diag(shells**2), basis.T @ values
        )
        coefficients[[kind == "sin" for kind, _ in stars]] *= splitting_scale
        return cls(
            stars=stars,
            height_stars=height_stars,
            dz_points=dz,
            coefficients=coefficients,
            height_coefficients=height_coefficients,
            v_cap=v_cap,
        )

    def height(self, u: Array, v: Array) -> Array:
        """Equilibrium height h(u, v) (A)."""
        return np.einsum(
            "s,s...->...",
            self.height_coefficients,
            evaluate_stars(self.height_stars, u, v),
        )

    def __call__(self, u: Array, v: Array, z: Array) -> Array:
        """Evaluate V (meV) at fractional (u, v) and height z (A); arrays broadcast."""
        u, v, z = np.broadcast_arrays(u, v, z)
        # flat beyond the fitted range, where V is far above the low states
        dz = np.clip(z - self.height(u, v), self.dz_points[0], self.dz_points[-1])
        spline = scipy.interpolate.CubicSpline(
            self.dz_points, self.coefficients, axis=1
        )
        c = spline(dz.ravel()).reshape(len(self.stars), *z.shape)
        t = np.einsum("s...,s...->...", c, evaluate_stars(self.stars, u, v))
        return np.minimum(_expand(t), self.v_cap)
