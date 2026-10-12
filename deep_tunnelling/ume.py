"""Universal Caldeira-Leggett master equation for H on Ni(111) in a Bloch basis.

System-bath coupling (periodic Caldeira-Leggett, thesis Ch. 6, extended to 2D):
for each of the three 120-degree directions q_j = (b1, b2, -b1-b2) / n of the
n x n supercell, the pair cos(q_j.r)/q, sin(q_j.r)/q couples to independent
baths with weight 2/3, giving isotropic in-plane friction gamma as q -> 0.
Unitary mixing of each pair into e^{+-i q_j.r} / q leaves the master equation
unchanged and makes every coupling shift the crystal momentum by +-q_j. The
z motion couples linearly (z), with the same friction.

The universal temperature correction (thesis Sec. 7.12) is applied in the
eigenbasis of H_s: for each coupling A,

    (L_U)_ab = sqrt(S(w_ba)) A_ab / hbar
    H_U = -(i / 2 hbar)(A^dag Lambda - Lambda^dag A),
    Lambda_ab = A_ab Gamma(w_ba),

with the cutoff-free Gamma of bath.py: the counterterm 1/2 kappa A^dag A cancels
the divergent constant of the principal value exactly (thesis Eqs. 7.77-7.78),
so neither appears.

H_U is built in a large basis of n_big bands and truncated with the jump
operators to the n_rate lowest bands (fcc, hcp and the first parallel
vibrational cluster). Since the steady state is translation invariant, hopping
lives entirely in the sector of density matrices diagonal in k, of dimension
n^2 n_rate^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from deep_tunnelling.bath import OhmicBath
from deep_tunnelling.bloch import BlochSolver, Grid, KPoint, k_grid
from deep_tunnelling.units import HBAR

Array = np.ndarray[Any, Any]

# fractional directions of the three in-plane couplings
DIRECTIONS = ((1, 0), (0, 1), (-1, -1))


@dataclass(frozen=True)
class Channel:
    """A coupling operator mapping crystal momentum k (index src) to k' (index dst)."""

    src: int
    dst: int
    matrix: Array  # (n_big, n_big), rows in k', columns in k
    weight: float


@dataclass
class BandData:
    """Bloch eigenstates on an n x n k grid and all coupling matrix elements."""

    energies: Array  # (nk, n_big)
    fcc_projector: Array  # (nk, n_big, n_big)
    channels: list[Channel]

    @classmethod
    def build(cls, solver: BlochSolver, n: int) -> BandData:
        """Diagonalize every k point of the n x n grid and build all coupling channels."""
        kpoints = k_grid(n)
        states = [
            solver.eigenstates(k)[1].reshape(solver.n_states, -1) for k in kpoints
        ]
        u, v, z = solver.grid.fractional()
        fcc_mask = _fcc_mask(u, v).ravel()
        projectors = [(uk.conj() * fcc_mask) @ uk.T for uk in states]
        channels = [
            *_momentum_channels(solver.grid, kpoints, states, n),
            *(
                Channel(i, i, (uk.conj() * z.ravel()) @ uk.T, 1.0)
                for i, uk in enumerate(states)
            ),
        ]
        energies = np.array([solver.eigenstates(k)[0] for k in kpoints])
        return cls(energies, np.array(projectors), channels)


def _momentum_channels(
    grid: Grid, kpoints: list[KPoint], states: list[Array], n: int
) -> list[Channel]:
    """Channels e^{+-i q.r} / q, weight 1/3 each, for the three directions q = d . b / n."""
    index = {k: i for i, k in enumerate(kpoints)}
    u, v, _ = grid.fractional()
    q_length = float(np.linalg.norm(grid.reciprocal[0])) / n
    channels = []
    for i, k in enumerate(kpoints):
        for d in DIRECTIONS:
            kq = (k[0] + Fraction(d[0], n), k[1] + Fraction(d[1], n))
            wrapped = (kq[0] % 1, kq[1] % 1)
            g = (int(kq[0] - wrapped[0]), int(kq[1] - wrapped[1]))
            j = index[wrapped]
            # a stored state at k + q - G has periodic part e^{-iG.r} u_{k+q}
            phase = np.exp(2j * np.pi * (g[0] * u + g[1] * v)).ravel()
            m = (
                (states[j].conj() * phase) @ states[i].T / q_length
            )  # <k+q| e^{iqr} |k> / q
            # e^{+iqr}: k -> k+q ; e^{-iqr}: k+q -> k with matrix m^dag
            channels.extend((Channel(i, j, m, 1 / 3), Channel(j, i, m.conj().T, 1 / 3)))
    return channels


def _fcc_mask(u: Array, v: Array) -> Array:
    """1 where the in-plane point is closer to an fcc than an hcp hollow."""

    def dist2(su: float, sv: float) -> Array:
        best = np.full(u.shape, np.inf)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                fu, fv = u - su - du, v - sv - dv
                best = np.minimum(best, (fu + 0.5 * fv) ** 2 + 0.75 * fv**2)
        return best

    return (dist2(1 / 3, 1 / 3) < dist2(2 / 3, 2 / 3)).astype(float)


@dataclass(frozen=True)
class RateResult:
    """Slowest Liouvillian eigenvalues and the steady-state fcc population at one friction."""

    eigenvalues: (
        Array  # slowest few, sorted by -Re; eigenvalues[0] = 0 is the steady state
    )
    p_fcc: float

    @property
    def lambda1(self) -> complex:
        """Slowest decay eigenvalue (ps^-1)."""
        return complex(self.eigenvalues[1])

    @property
    def total_rate(self) -> float:
        """k_fh + k_hf = -Re lambda_1 (ps^-1)."""
        return -self.lambda1.real

    @property
    def fcc_escape_rate(self) -> float:
        """Total escape rate out of an fcc hollow (ps^-1); detailed balance k_fh p_f = k_hf p_h."""
        return self.total_rate * (1 - self.p_fcc)

    @property
    def hcp_escape_rate(self) -> float:
        """Total escape rate out of an hcp hollow (ps^-1); equal to fcc for degenerate hollows."""
        return self.total_rate * self.p_fcc


def _assemble(
    bands: BandData, bath: OhmicBath | None, n_rate: int
) -> scipy.sparse.csr_matrix:
    """Assemble the k-diagonal Liouvillian: coherent part from the bare energies (bath=None) or the bath terms."""
    nk, n_big = bands.energies.shape
    e = bands.energies
    h_u = np.zeros((nk, n_big, n_big), dtype=complex)
    jumps: list[tuple[int, int, Array]] = []
    if bath is not None:
        for ch in bands.channels:
            w = (
                e[ch.src][None, :] - e[ch.dst][:, None]
            ) / HBAR  # w_ba, b in k (col), a in k' (row)
            a = ch.matrix
            lam = a * bath.gamma_function(w)
            h_u[ch.src] += (
                ch.weight * (-0.5j / HBAR) * (a.conj().T @ lam - lam.conj().T @ a)
            )
            lu = np.sqrt(ch.weight) * a * np.sqrt(bath.s(w)) / HBAR
            jumps.append((ch.src, ch.dst, lu[:n_rate, :n_rate]))
    eye = np.eye(n_rate)
    g = np.zeros((nk, n_rate, n_rate), dtype=complex)
    for src, _, lu in jumps:
        g[src] += lu.conj().T @ lu
    blocks: dict[tuple[int, int], Array] = {}
    for i in range(nk):
        h = h_u[i, :n_rate, :n_rate] + (np.diag(e[i, :n_rate]) if bath is None else 0)
        h = 0.5 * (h + h.conj().T)
        blocks[i, i] = -1j / HBAR * (np.kron(h, eye) - np.kron(eye, h.T)) - 0.5 * (
            np.kron(g[i], eye) + np.kron(eye, g[i].T)
        )
    for src, dst, lu in jumps:
        blocks[dst, src] = blocks.get((dst, src), 0) + np.kron(lu, lu.conj())
    sparse = {key: scipy.sparse.csr_matrix(block) for key, block in blocks.items()}
    return scipy.sparse.bmat(
        [[sparse.get((i, j)) for j in range(nk)] for i in range(nk)], format="csr"
    )


def liouvillian_parts(
    bands: BandData, mass: float, temperature: float, n_rate: int
) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """Split L(gamma) = L0 + gamma L1: every bath term (S, Gamma, H_U, L^dag L) is linear in gamma."""
    return _assemble(bands, None, n_rate), _assemble(
        bands, OhmicBath(mass, 1.0, temperature), n_rate
    )


def slowest_eigenvalues(
    liouvillian: scipy.sparse.csr_matrix, n_eig: int = 4
) -> tuple[Array, Array]:
    """Eigenvalues closest to zero (sorted by -Re) and the stationary state, by shift-invert."""
    w, vr = scipy.sparse.linalg.eigs(
        liouvillian.tocsc(), k=n_eig, sigma=1e-9, which="LM"
    )
    order = np.argsort(-w.real)
    return w[order], vr[:, order[0]]


def hop_rates(
    bands: BandData, mass: float, temperature: float, gammas: list[float], n_rate: int
) -> list[RateResult]:
    """Hop rates for several frictions at one temperature, assembling the Liouvillian once."""
    l0, l1 = liouvillian_parts(bands, mass, temperature, n_rate)
    nk = bands.energies.shape[0]
    projector = bands.fcc_projector[:, :n_rate, :n_rate]
    out = []
    for gamma in gammas:
        w, stationary = slowest_eigenvalues((l0 + gamma * l1).tocsr())
        rho = stationary.reshape(nk, n_rate, n_rate)
        rho /= np.trace(rho, axis1=1, axis2=2).sum()
        out.append(
            RateResult(w, float(np.real(np.einsum("kab,kba->", rho, projector))))
        )
    return out
