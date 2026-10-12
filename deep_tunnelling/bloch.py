"""Bloch eigenstates of H on Ni(111).

The unit cell (hexagonal, lattice constant a) is sampled on an N1 x N2 grid in
fractional coordinates and an Nz grid over the z range of the DFT data, which
is treated as periodic (the potential is > 1 eV at both ends so the low states
vanish there). For crystal momentum k the periodic part u_k of the Bloch state
satisfies [hbar^2 (-i grad + k)^2 / 2m + V] u_k = E u_k.

The fast z vibrations are removed first: the 1D problem T_z + V(fcc, z) is
diagonalized once and z is contracted onto its lowest n_z states (6 reproduce
the full grid to ~1e-3 meV). The in-plane kinetic energy is applied with FFTs
and the n1 x n2 x n_z Hamiltonian is diagonalized with Lanczos.

An n x n supercell with periodic boundary conditions is equivalent to the
n x n Monkhorst-Pack grid k = (i b1 + j b2) / n, i, j = 0..n-1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse.linalg

from deep_tunnelling.units import A_NI, HBAR

if TYPE_CHECKING:
    from deep_tunnelling.potential import SurfacePotential

KPoint = tuple[Fraction, Fraction]


@dataclass(frozen=True)
class Grid:
    """Real-space grid of the unit cell."""

    n1: int = 18
    n2: int = 18
    nz: int = 28
    z_min: float = -1.0
    z_max: float = 1.2
    a: float = A_NI

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape (n1, n2, nz)."""
        return (self.n1, self.n2, self.nz)

    @property
    def lz(self) -> float:
        """Length of the periodic z range (A)."""
        return self.z_max - self.z_min

    @property
    def reciprocal(self) -> np.ndarray[Any, Any]:
        """Rows are b1, b2 (A^-1) with a_i . b_j = 2 pi delta_ij."""
        a1 = np.array([1.0, 0.0]) * self.a
        a2 = np.array([0.5, np.sqrt(3) / 2]) * self.a
        return 2 * np.pi * np.linalg.inv(np.array([a1, a2])).T

    def fractional(
        self,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Fractional in-plane coordinates (u, v) and z (A) of every grid point."""
        u = np.arange(self.n1) / self.n1
        v = np.arange(self.n2) / self.n2
        z = self.z_min + np.arange(self.nz) * self.lz / self.nz
        return np.meshgrid(u, v, z, indexing="ij")  # type: ignore[return-value]

    def g_vectors(
        self,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Cartesian (gx, gy, gz) for each FFT component, broadcast to the grid shape."""
        m1 = np.fft.fftfreq(self.n1, 1 / self.n1)
        m2 = np.fft.fftfreq(self.n2, 1 / self.n2)
        mz = np.fft.fftfreq(self.nz, 1 / self.nz)
        b = self.reciprocal
        mm1, mm2, mmz = np.meshgrid(m1, m2, mz, indexing="ij")
        return (
            mm1 * b[0, 0] + mm2 * b[1, 0],
            mm1 * b[0, 1] + mm2 * b[1, 1],
            mmz * 2 * np.pi / self.lz,
        )


@dataclass
class BlochSolver:
    """Diagonalize the Bloch Hamiltonian at arbitrary k, caching results."""

    potential: SurfacePotential
    mass: float
    grid: Grid = field(default_factory=Grid)
    n_states: int = 48  # bands per k for the Lamb shift (converged to ~0.4%)
    n_z: int = 6  # z states kept from the 1D problem at the fcc site
    _g_xy: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] = field(
        init=False, repr=False
    )
    _phi: np.ndarray[Any, Any] = field(init=False, repr=False)
    _local: np.ndarray[Any, Any] = field(init=False, repr=False)
    _cache: dict[KPoint, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        u, v, z = self.grid.fractional()
        gx, gy, gz = self.grid.g_vectors()
        self._g_xy = (gx[:, :, 0], gy[:, :, 0])
        gz = gz[0, 0]
        # pre-diagonalize H_z = T_z + V(fcc, z) and project V(x, y, z) onto its lowest states
        nz = self.grid.nz
        kinetic = np.fft.ifft(
            HBAR**2
            / (2 * self.mass)
            * gz[:, None] ** 2
            * np.fft.fft(np.eye(nz), axis=0),
            axis=0,
        ).real
        v_fcc = self.potential(np.full(nz, 1 / 3), np.full(nz, 1 / 3), z[0, 0])
        self._phi = np.linalg.eigh(kinetic + np.diag(v_fcc))[1][
            :, : self.n_z
        ]  # (nz, n_z), real
        h_z = self._phi.T @ kinetic @ self._phi  # same at every (x, y)
        self._local = (
            np.einsum("zi,xyz,zj->xyij", self._phi, self.potential(u, v, z), self._phi)
            + h_z
        )  # (n1, n2, n_z, n_z)

    def _solve(self, k: KPoint) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        b = self.grid.reciprocal
        kc = float(k[0]) * b[0] + float(k[1]) * b[1]
        gx, gy = self._g_xy
        t_xy = HBAR**2 / (2 * self.mass) * ((gx + kc[0]) ** 2 + (gy + kc[1]) ** 2)
        local = self._local
        shape = (self.grid.n1, self.grid.n2, self.n_z)

        def matvec(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            c = x.reshape(shape)
            kin = np.fft.ifft2(
                t_xy[:, :, None] * np.fft.fft2(c, axes=(0, 1)), axes=(0, 1)
            )
            return (kin + np.einsum("xyij,xyj->xyi", local, c)).ravel()

        size = int(np.prod(shape))
        op = scipy.sparse.linalg.LinearOperator(  # ty: ignore[missing-argument]
            (size, size),
            matvec=matvec,  # ty: ignore[unknown-argument]
            dtype=np.complex128,  # ty: ignore[parameter-already-assigned]
        )
        # a generic start vector: a uniform one is invariant under the little
        # group of k, so in exact arithmetic Lanczos would only find totally
        # symmetric states at high-symmetry k (it works only through round-off)
        rng = np.random.default_rng(0)
        v0 = rng.standard_normal(size) + 1j * rng.standard_normal(size)
        energies, vectors = scipy.sparse.linalg.eigsh(
            op,
            k=self.n_states,
            which="SA",
            tol=1e-10,
            ncv=max(3 * self.n_states, 60),
            v0=v0,
        )
        order = np.argsort(energies)
        coeffs = vectors[:, order].T.reshape(self.n_states, *shape)
        return energies[order], np.einsum("sxyi,zi->sxyz", coeffs, self._phi)

    def eigenstates(
        self, k: KPoint
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return (energies (meV), u_k on the grid normalized to sum |u|^2 = 1)."""
        k = (Fraction(k[0]) % 1, Fraction(k[1]) % 1)
        if k in self._cache:
            return self._cache[k]
        minus = (-k[0] % 1, -k[1] % 1)
        if minus in self._cache:
            # time reversal: conj(u_m) is the periodic part at -m = k - G,
            # so u_k = exp(-i G.r) conj(u_m) with G = k + m
            e, u = self._cache[minus]
            g = (int(k[0] + minus[0]), int(k[1] + minus[1]))
            uu, vv, _ = self.grid.fractional()
            result = (e, u.conj() * np.exp(-2j * np.pi * (g[0] * uu + g[1] * vv)))
        else:
            result = self._solve(k)
        self._cache[k] = result
        return result


def k_grid(n: int) -> list[KPoint]:
    """Monkhorst-Pack grid equivalent to an n x n supercell with periodic BCs."""
    return [(Fraction(i, n), Fraction(j, n)) for i in range(n) for j in range(n)]
