# Theory

This package computes hop rates of H and D on close-packed metal surfaces (Ni(111),
Pt(111), Ru(0001)) from a Markovian open-system
master equation, the *universal* Caldeira–Leggett master equation (UME), evaluated
in the Bloch eigenbasis of a three-dimensional periodic potential energy surface,
and compares them with helium spin-echo (HeSE) measurements. This document
describes the method; fitted values are printed by the examples.

The pipeline is

```
DFT site energies ──► periodic 3D PES ──► Bloch states on an n×n k grid
   (potential.py)         (potential.py)          (bloch.py)
                                                      │
          periodic Caldeira–Leggett couplings ◄───────┘
          + universal temperature correction           (ume.py, bath.py)
                          │
          Liouvillian in the translation-invariant sector
                          │
          slowest decay rate ──► fcc / hcp escape rates   (ume.py, model.py)
```

References to the documents in `resources/` use these short names:

| Short name | File | Content |
|---|---|---|
| **[MO]** | `thesis.pdf` | M. Ord, *Quantum Simulations of Adsorbate Dynamics*, PhD thesis (Cambridge, 2026). PDF page = printed page. |
| **[UME]** | `Universal_Master_Equation_DoubleWell_Hopping (1).pdf` | Note: hopping rates in a double well from the spectrum of the universal Caldeira–Leggett master equation. |
| **[NR]** | `Nathan_2020.pdf` | F. Nathan and M. S. Rudner, *Universal Lindblad equation for open quantum systems*, PRB **102**, 115109 (2020). |
| **[JZ]** | `2015 - Jianding Zhu.pdf` | J. Zhu, *Study of quantum effects in adsorbate diffusion*, PhD thesis (Cambridge, 2015). Page numbers below are printed pages (PDF page = printed + 14). |
| **[PT]** | `2018 - Peter Townsend.pdf` | P. Townsend, *Diffusion of light adsorbates on transition metal surfaces*, PhD thesis (Cambridge, 2018). |
| **[OP]** | `Ord_2026.pdf` | M. Ord *et al.*, *Quantum simulations of the ballistic motion of a surface adsorbate*, PCCP **28**, 17708 (2026). |

Units throughout are meV, ps and Å (`deep_tunnelling/units.py`), so
$\hbar = 0.6582$ meV ps and $\hbar^2/2m_\mathrm{H} = 2.074$ meV Å².

---

## 1. Potential energy surface (`potential.py`)

### 1.1 DFT data

The PES is built from DFT total energies of a single H atom on a metal slab, sampled
at points of the irreducible triangle of the (111)/(0001) surface at a set of
heights (`raw_data.json` in each example folder, with $z = 0$ at the global
minimum). For Ni(111) these are the six symmetry-distinct sites, top, fcc, hcp,
bridge and the top–fcc and top–hcp midpoints; for Pt(111) and Ru(0001), 31 points
on a 1/12 grid of the cell. The provenance of each data set is in its example's
`util.py`. In
fractional coordinates $\mathbf r = u\,\mathbf a_1 + v\,\mathbf a_2$, with $\mathbf a_1 = a(1,0)$,
$\mathbf a_2 = a(\tfrac12, \tfrac{\sqrt3}{2})$ and a top atom at the origin, the sites are
fcc $(\tfrac13,\tfrac13)$, hcp $(\tfrac23,\tfrac23)$ and bridge $(0,\tfrac12)$, and $a$ is the
lattice constant of the DFT slab.

### 1.2 Interpolation

The in-plane dependence is expanded in $C_{3v}$-invariant plane-wave "stars"
(`symmetric_stars`), sums of $\cos\mathbf G\cdot\mathbf r$ or $\sin\mathbf G\cdot\mathbf r$ over the
orbit of a reciprocal-lattice vector $\mathbf G$, ordered by $|\mathbf G|^2$. The lowest six
are

$$
\begin{aligned}
f_0 &= 1, &
f_1 &= \textstyle\sum_{3} \cos 2\pi(\cdot), &
g_1 &= \sin 2\pi u + \sin 2\pi v - \sin 2\pi(u+v), \\
f_2 &= \textstyle\sum_3 \cos 2\pi(u-v,\,2u+v,\,u+2v), &
f_3 &= \textstyle\sum_3 \cos 4\pi(\cdot), &
g_3 &= \sin 4\pi u + \sin 4\pi v - \sin 4\pi(u+v),
\end{aligned}
$$

where $\sum_3 \cos 2\pi(\cdot)$ means $\cos 2\pi u + \cos 2\pi v + \cos 2\pi(u+v)$. The
cosine stars are symmetric under the mirror that exchanges fcc and hcp; the sine
stars are antisymmetric and carry the fcc–hcp energy difference.

H sits higher on the top site than in the hollows (~0.6 Å), so at the hollow
height the top site is a wall of several eV. Fitted at fixed z, this wall leaks through the Fourier
series into the hollows, and with 31 points rings into spurious minima.
`SurfacePotential` therefore measures z from the local equilibrium height,

$$
V(u,v,z) = \mathrm{expand}\Big[\sum_s c_s\big(z - h(u,v)\big)\,f_s(u,v)\Big],
$$

with $h$ a star fit to the height of the minimum at each point and, at each
$\delta z$ on a 0.05 Å grid, $c_s(\delta z)$ a least-squares fit of the stars to the
energies at $z = h_i + \delta z$ (each point held at its end value beyond its own data),
splined in $\delta z$. At most one star per point is used (6 points: exact at the
sites; 31 points: 25 stars and 20 for $h$, with a ridge $10^{-7}|\mathbf G|^4$). Energies are
capped at 4 eV, which does not affect any state below the barrier, and compressed
before fitting, $T(V) = B\ln(1 + V/B)$ with $B = 2.5$ eV.

### 1.3 Benchmark

The 31-point data sets unfold by symmetry onto a full 12 × 12 grid of the cell,
which a local tricubic spline interpolates without any Fourier series (cubic in z
per column, a periodic cubic B-spline in $(u,v)$ at each z, on the same capped,
compressed energies; `test_potential_matches_local_spline`). Below 400 meV the
star fit agrees with it to a few meV, and the H bands to a fraction of a meV.

Six sites do not fix the interpolation between them: fitted to the six Ni-type
sites of the 31-point data, either the height-following or a fixed-z star fit
lowers the H in-plane excitation by ~10% and the zero-point energy by ~20 meV
against the spline. This is a limit of six-site data, not of the scheme.

### 1.4 Correction

`splitting_scale` multiplies the fcc/hcp-antisymmetric (sine) stars, which moves
hcp relative to fcc and leaves the barrier and the top site unchanged. It is the
one correction applied to a PES; how it is chosen is described in §7.2.

---

## 2. Bloch states (`bloch.py`)

### 2.1 Supercell and Bloch's theorem

An $n\times n$ supercell of the surface with periodic boundary conditions is
equivalent to the $n\times n$ Monkhorst–Pack grid $\mathbf k = (i\,\mathbf b_1 + j\,\mathbf b_2)/n$,
$i,j = 0\dots n-1$ (`k_grid`), with $\mathbf a_i\cdot\mathbf b_j = 2\pi\delta_{ij}$.
By Bloch's theorem the supercell Hamiltonian is block diagonal in $\mathbf k$, and the
periodic part $u_\mathbf k$ of each eigenstate obeys

$$
\left[\frac{\hbar^2}{2m}(-i\nabla + \mathbf k)^2 + V(\mathbf r)\right] u_\mathbf k = E\,u_\mathbf k ,
$$

the same block structure as [MO] Eqs. 3.17–3.20 (pp. 31–32). Each block is solved
on an $18\times18$ grid of the unit cell, with the kinetic energy applied by FFT. The
$z$ direction is periodic over a range of the data where the potential exceeds
1 eV at both ends, so the low states vanish there.

### 2.2 Contracted z basis

The fast $z$ vibrations are removed before diagonalizing. The 1D problem
$T_z + \min_{u,v}V(u,v,z)$ is solved once on the z grid and $z$ is projected onto its
lowest $n_z$ states $\phi_i(z)$. The lower envelope $\min_{u,v}V$ follows the hollows
at small z and the top site above it; a basis on $V(\mathrm{fcc}, z)$ alone misses the
top-site states, which matters where they are low (Pt(111)). The 3D Hamiltonian
then acts on $18\times18\times n_z$ coefficients:

$$
H_{ij}(\mathbf r_\parallel) = \langle\phi_i|T_z|\phi_j\rangle + \langle\phi_i|V(\mathbf r_\parallel, z)|\phi_j\rangle ,
$$

plus the in-plane kinetic energy. The lowest 48 states per $\mathbf k$ are found by
Lanczos (`eigsh`).

### 2.3 Time reversal

Only about half the k points are diagonalized. Time reversal gives
$\psi_{-\mathbf k} = \psi_\mathbf k^*$; because states are stored at $\mathbf k$ reduced into
$[0,1)^2$, the periodic part picks up a reciprocal-lattice phase,

$$
u_\mathbf k = e^{-i\mathbf G\cdot\mathbf r}\,u_{\mathbf m}^*, \qquad \mathbf G = \mathbf k + \mathbf m,
$$

where $\mathbf m = -\mathbf k \bmod 1$ is the stored partner. Omitting this phase silently
corrupts every coupling matrix element involving those k points.

---

## 3. System–bath coupling (`ume.py`)

### 3.1 Periodic Caldeira–Leggett model

The linear Caldeira–Leggett coupling $x\,\phi(t)$ is not periodic. [MO] §6.2
(Eqs. 6.3–6.4, p. 94) replaces it by a pair of operators coupled to independent baths
with the same correlation function,

$$
H = H_s + \frac{1}{k}\cos(k\hat x)\,\phi_c(t) + \frac{1}{k}\sin(k\hat x)\,\phi_s(t),
$$

which is periodic when $k$ is a multiple of $2\pi/L$ and reduces to linear coupling
for $k\,\Delta x \ll 1$, where $\Delta x$ is the coherence length. No counterterm is
needed because $\cos^2 + \sin^2 = 1$.

### 3.2 Extension to the surface

The package uses one $(\cos,\sin)$ pair along each of the three equivalent directions
$\mathbf q_j \in \{\mathbf b_1, \mathbf b_2, -\mathbf b_1-\mathbf b_2\}/n$. These are the smallest
wavevectors the $n\times n$ supercell allows, $|\mathbf q| = |\mathbf b|/n = 2.92/n$ Å⁻¹.
Each pair has weight $2/3$, because $\sum_j \tfrac23\hat{\mathbf q}_j\hat{\mathbf q}_j^{\mathsf T} = \mathbb 1$,
so the in-plane friction is isotropic as $q\to0$.

A unitary rotation of each pair into $e^{\pm i\mathbf q_j\cdot\mathbf r}/q$ leaves the master
equation unchanged:

$$
\mathcal D[\cos/q] + \mathcal D[\sin/q] = \tfrac{1}{2q^2}\left(\mathcal D[e^{i\mathbf q\cdot\mathbf r}] + \mathcal D[e^{-i\mathbf q\cdot\mathbf r}]\right).
$$

In this form every coupling shifts the crystal momentum by $\pm\mathbf q_j$, so its matrix
elements are overlaps of periodic parts:

$$
\langle\psi_{\mathbf k+\mathbf q,a}|\,e^{i\mathbf q\cdot\mathbf r}\,|\psi_{\mathbf k,b}\rangle = \sum_{\mathbf r} u^*_{\mathbf k+\mathbf q,a}(\mathbf r)\,u_{\mathbf k,b}(\mathbf r),
$$

with an extra $e^{i\mathbf G\cdot\mathbf r}$ when $\mathbf k+\mathbf q$ wraps past the zone edge.
This is the scattering-operator structure of [MO] Eqs. 3.21–3.22. Each of the six
operators carries weight $1/3$. The $z$ motion couples linearly ($A = z$, weight 1,
same friction) and does not change $\mathbf k$.

The supercell size $n$ is therefore also the coupling wavevector: increasing $n$
takes the periodic coupling towards linear coupling, and is the main convergence
parameter (§6).

---

## 4. Universal temperature correction (`ume.py`, `bath.py`)

### 4.1 From Bloch–Redfield to the UME

For a coupling $A\,\phi(t)$ with bath correlation $D(\tau)$, the Bloch–Redfield
equation ([MO] Eqs. 7.75–7.76, p. 130) has a Hamiltonian part
$H_\Lambda = \tfrac12\kappa A^2 - \tfrac{i}{2\hbar}(A\Lambda - \Lambda^\dagger A)$, with
$\Lambda = \int_0^\infty D(\tau)A(-\tau)\,d\tau$, and a dissipator that is not of Lindblad
form. The *universal* correction ([MO] §7.12, Eqs. 7.79–7.81, pp. 130–131; following
[NR]) keeps the Hamiltonian part exactly and replaces the dissipator by a single
jump operator,

$$
\partial_t\rho = -\frac{i}{\hbar}[H_s + H_U, \rho] + L_U\rho L_U^\dagger - \tfrac12\{L_U^\dagger L_U, \rho\},
\qquad L_U = \int d\tau\, g(\tau)\,A(\tau).
$$

The result is completely positive and trace preserving, and the transition rates
between eigenstates of $H_s$ equal the Fermi golden-rule (Bloch–Redfield) rates.
It needs the bath correlation time $\hbar/k_BT$ to be short compared with every
system rate the bath generates, with no condition on the level spacing of $H_s$.
For hopping, the fastest of these is not γ. It is the dephasing of a superposition
of two neighboring hollows a distance $d = a/\sqrt3 = 1.44$ Å apart,
$\Gamma_\phi = \eta d^2 k_BT/\hbar^2$, which is $\approx 5\gamma$ at 120 K for H. The condition
$\Gamma_\phi \ll k_BT/\hbar$ is therefore $\eta d^2/\hbar = 2\pi\alpha \ll 1$, independent of
temperature, where α is the spin–boson coupling. For H on Ni(111), $2\pi\alpha \approx 0.33\,\gamma$/ps⁻¹,
which is of order 1 for the fitted frictions of a few ps⁻¹. This is marginal. [UME] §7.3 already finds a 20% spread between second-order master
equations at $\alpha \approx 0.02$. This matters for H, where
$\hbar\omega/k_BT \approx 4$–8 and the standard high-temperature correction (L =
x + iħp/4mk_BT) fails ([MO] §7.11, Table 7.4).

### 4.2 Evaluation in the eigenbasis

For a general potential, $L_U$ has no closed form ([MO] p. 131). In the eigenbasis of
$H_s$, however, the time integrals become multiplication by a function of the transition
frequency $\omega_{ba} = (E_b - E_a)/\hbar$ ([UME] Eqs. 5–7):

$$
(L_U)_{ab} = \frac{1}{\hbar}\sqrt{S(\omega_{ba})}\,A_{ab}, \qquad
H_U = -\frac{i}{2\hbar}\left(A^\dagger\Lambda - \Lambda^\dagger A\right), \qquad
\Lambda_{ab} = A_{ab}\,\Gamma(\omega_{ba}).
$$

Here $a$ is the final and $b$ the initial state, and

$$
S(\omega) = \frac{2\hbar J(\omega)}{1 - e^{-\hbar\omega/k_BT}}, \qquad
\Gamma(\nu) = \tfrac12 S(\nu) + i P(\nu), \qquad
P(\nu) = \frac{1}{2\pi}\,\mathcal P\!\!\int \frac{S(\omega)}{\nu-\omega}\,d\omega .
$$

$S$ obeys detailed balance, $S(-\omega) = e^{-\hbar\omega/k_BT}S(\omega)$, so every
transition samples the bath at its own frequency and the steady state is Boltzmann up
to $O(\gamma)$. For a non-Hermitian coupling such as $e^{i\mathbf q\cdot\mathbf r}$ the same
expressions hold with $A^\dagger$ in place of $A$, which is what the $\pm\mathbf q$ pairs need.

### 4.3 Ohmic bath without a cutoff

The bath is Ohmic, $J(\omega) = \eta\,\omega$ with $\eta = m\gamma$, equivalent to the
$\gamma(\omega) = 2m\gamma\omega/\pi$ of [MO]. For this bath $P(\nu)$ diverges, but only
through two terms, both of which drop out:

1. **A constant**, $-\hbar\eta\omega_c/2$, which cancels the counterterm $\tfrac12\kappa A^\dagger A$
   with $\kappa = \eta\omega_c$ exactly ([MO] Eq. 7.77). This is why the UME is
   well defined for an Ohmic bath ([MO] Eq. 7.78 and the text after it).
2. **A term linear in ν**, $\propto \nu\ln\omega_c$, which multiplies the double
   commutator, $2A^\dagger H_sA - \{A^\dagger A, H_s\}$. For $A = z$ this is the c-number
   $\hbar^2/m$. For a single $A = e^{i\mathbf q\cdot\mathbf r}/q$ it is
   $(\hbar^2/m)(1 + 2\mathbf q\cdot\hat{\mathbf p}/\hbar q^2)$, which is *not* a c-number. The
   momentum term cancels only between the $+\mathbf q$ and $-\mathbf q$ channels, which carry
   equal weight. The sum then only shifts all energies equally.

What remains is cutoff-free. Writing $\coth$ as a Matsubara sum and removing the two
terms above gives the closed form used in `OhmicBath.principal_value`:

$$
P(\nu) = 2\eta k_BT\;x\left[\mathrm{Re}\,\psi(1 + ix) + \gamma_E\right], \qquad x = \frac{\hbar\nu}{2\pi k_BT},
$$

which tends to the standard Ohmic Lamb shift $(\hbar\eta\nu/\pi)\ln(\hbar\nu/2\pi k_BT)$ at
large ν. The cancellation of the c-number term is exact only in a complete basis,
which is why $H_U$ is built in a large basis (§5).

### 4.4 Friction

The friction is electron–hole (electronic) friction. To leading order this is Ohmic
and independent of temperature, because the Fermi window $-dn_f/dE$ reduces to
$\delta(E - E_F)$ for $k_BT \ll E_F$ ([MO] §5.4, Eqs. 5.38–5.43, pp. 81–83).
Temperature and frequency dependence enter only through structure in the density of
states near $E_F$.

The friction coefficient $\eta = m\gamma$ couples to the velocity and does not depend
on the adsorbate's mass, so deuterium uses $\gamma_\mathrm D = \gamma_\mathrm H\,m_\mathrm H/m_\mathrm D$.
Independent estimates for H on close-packed metals lie around γ ≈ 1–2.5 ps⁻¹ ([PT]
§6.3.7 and §6.5: HREELS and IR linewidths and TDDFT friction; [PT] uses γ_H = 1 and
γ_D = 0.5 ps⁻¹).

---

## 5. Truncation and assembly (`ume.py`, `model.py`)

### 5.1 Two basis sizes

* **Lamb-shift basis**: $H_U$ is a sum over intermediate states,
  $(A^\dagger\Lambda)_{ab} = \sum_c A^*_{ca}A_{cb}\Gamma(\omega_{bc})$, and its cutoff-free form
  relies on completeness (§4.3). It is built from the lowest 48 bands per $\mathbf k$
  (up to ~360 meV), which converges the rates to ~0.4%. This follows [UME] §5.1
  ("$N_\mathrm{big} \ge 2N$").
* **Master-equation basis**: the Liouvillian keeps every band whose mean energy lies
  within a fixed energy (`max_energy`) of the ground band, about the barrier height.
  D needs more bands than H: its first vibrational band barely tunnels, so its
  activated hopping runs through higher bands.

  The cutoff is a fixed energy because activated hopping runs through the bands
  just below the barrier at every temperature: on Ni(111) a cutoff of $6k_BT$
  underestimates the H rate by 87% at 160 K. Memory grows as $n^2 n_\mathrm{rate}^4$,
  which limits it for D.

$H_U$ and $L_U$ are truncated to the master-equation basis, and $L_U^\dagger L_U$ is built
from the *truncated* $L_U$, so the truncated generator stays exactly trace preserving
and completely positive ([UME] §5.1). Real transitions out of the kept bands are
neglected; they are thermally negligible below the barrier.

### 5.2 Translation-invariant sector

Each coupling shifts $\mathbf k$ by $\pm\mathbf q_j$ (or 0 for $z$), so the dissipator maps a
density matrix that is diagonal in $\mathbf k$, $\rho = \oplus_\mathbf k\rho_\mathbf k$, to another
one. The steady state and fcc ↔ hcp population exchange live entirely in this
sector, of dimension $n^2 N^2$ rather than $(n^2N)^2$. In row-major vectorization
($\mathrm{vec}(A\rho B) = (A\otimes B^{\mathsf T})\,\mathrm{vec}\,\rho$) the blocks are

$$
\mathcal L_{\mathbf k\mathbf k} = -\tfrac{i}{\hbar}\left(H_\mathbf k\otimes\mathbb 1 - \mathbb 1\otimes H_\mathbf k^{\mathsf T}\right) - \tfrac12\left(G_\mathbf k\otimes\mathbb 1 + \mathbb 1\otimes G_\mathbf k^{\mathsf T}\right),
\qquad
\mathcal L_{\mathbf k'\mathbf k} = L_{\mathbf k'\leftarrow\mathbf k}\otimes L^*_{\mathbf k'\leftarrow\mathbf k},
$$

with $H_\mathbf k = E_\mathbf k + H_{U,\mathbf k}$ and $G_\mathbf k = \sum L^\dagger L$ over the channels leaving
$\mathbf k$.

Every bath term ($S$, $\Gamma$, $H_U$, $L^\dagger L$) is linear in γ, so
$\mathcal L(\gamma) = \mathcal L_0 + \gamma\,\mathcal L_1$ (plus $\gamma_g\,\mathcal L_g$ for the ground-state
couplings of §7.4). Each temperature is therefore assembled
once and reused for every friction.

---

## 6. Hop rates (`ume.py`, `model.py`)

### 6.1 From the spectrum

The slowest eigenvalues of $\mathcal L$ are found by sparse shift-invert. $\lambda_0 = 0$ is
the steady state $\rho_\mathrm{ss}$, and $\lambda_1$ is the exchange of population between
the fcc and hcp sublattices ([UME] Eq. 9 and §6):

$$
-\lambda_1 = k_{\mathrm f\to\mathrm h} + k_{\mathrm h\to\mathrm f}.
$$

The two escape rates follow from detailed balance,
$k_{\mathrm f\to\mathrm h}\,p_\mathrm f = k_{\mathrm h\to\mathrm f}\,p_\mathrm h$, with the steady-state hollow
populations $p_\mathrm f, p_\mathrm h$. Each eigenstate counts wholly towards the site (fcc,
hcp or top) nearest most of its weight (`BandData.sites`), so the populations are
diagonal in the eigenbasis; top states belong to neither hollow (§7.5). With
$p_\mathrm f + p_\mathrm h$ normalized to 1,

$$
k_\mathrm{slow} = k_{\mathrm f\to\mathrm h} = -\lambda_1\,p_\mathrm h, \qquad
k_\mathrm{fast} = k_{\mathrm h\to\mathrm f} = -\lambda_1\,p_\mathrm f .
$$

These are total escape rates out of a site, summed over its three neighbors. For
degenerate hollows $p_\mathrm f = \tfrac12$ and the two rates would be equal.

### 6.2 Validity

Describing hopping by one rate needs $\lambda_1$ to be well separated from the next
eigenvalue, which is intra-site vibrational relaxation at about γ. For H and D on
Ni(111) the ratio is ~20–3000, falling with temperature, and $\lambda_1$ is real, so
hopping is incoherent.

**Limitation.** The k-diagonal sector sees only fcc ↔ hcp exchange. A long jump
(fcc → fcc) leaves the sublattice populations unchanged and is invisible to
$\lambda_1$. Capturing it requires the intermediate scattering function,
$I(\Delta\mathbf K, t) = \mathrm{Tr}\!\left[e^{i\Delta\mathbf K\cdot\mathbf r}e^{\mathcal Lt}\left(\rho_\mathrm{ss}e^{-i\Delta\mathbf K\cdot\mathbf r}\right)\right]$
([MO] Eq. 6.38, p. 105), which evolves in the sector of elements $|\mathbf k+\Delta\mathbf K\rangle\langle\mathbf k|$.
Its two slowest eigenvalues would be the two exponential decay rates measured by HeSE.
This is not yet implemented.

### 6.3 The two regimes

* **Deep tunneling** (below ~160 K). Only the fcc and hcp ground bands are occupied.
  The fcc and hcp ground states are coupled by a tunneling matrix element
  $t \approx 0.03$ meV for H. With the bias $\varepsilon \approx 13$ meV $\gg t$, the eigenstates
  localize and a hop is a one-boson transition with matrix element $\approx t d/\varepsilon$.
  Downhill this gives $\sim 2\eta d^2t^2/\hbar^2\varepsilon$, and uphill an extra
  $e^{-\varepsilon/k_BT}$. The rate grows as γ and is suppressed by the off-resonance
  factor $(t/\varepsilon)^2$, so a single friction fitted to the activated regime gives
  no plateau (§7.4).
* **Activated** (above ~160 K). The bath excites the parallel vibration, whose band is
  wide enough to tunnel between sites before it relaxes. The activation energy is set
  by the hopping band, not by the classical barrier ([UME] §4), and the rate rises
  with γ up to a turnover.

---

## 7. Comparison with experiment (`examples/`)

### 7.1 Observables

HeSE measures the decay of the intermediate scattering function at a momentum
transfer ΔK. The published analyses reduce it to rates of a jump model, and the
model's escape rates are mapped onto whichever rate was reported:

* **Two-sublattice rates.** Fits of the non-Bravais fcc/hcp jump model give the
  escape rate out of fcc (the slow rate) and out of hcp (the fast rate, or the
  inverse residence time $1/\tau$ in hcp), or their ratio. These are
  $k_\mathrm{slow}$ and $k_\mathrm{fast}$ (§6.1) directly. For H/Ni(111) [JZ] reports the
  slow rate and the fast/slow ratio ([JZ] Eqs. 2.34–2.44); for H/Ru(0001)
  Townsend ([PT] §6.3) reports $1/\tau_\mathrm{hcp}$ and McIntosh *et al.* (2013) the
  fcc → hcp rate.
* **Chudley–Elliott rate.** Where hcp is only a short-lived intermediate the data
  are fitted with fcc → fcc jumps (H/Pt(111), Jardine *et al.* 2010). The model is
  mapped onto it by equating dephasing rates at the measured ΔK: the slower root
  of the two-sublattice jump matrix,
  $\alpha = \tfrac12\left[S - \sqrt{S^2 - 4k_\mathrm{fh}k_\mathrm{hf}(1-|F|^2)}\right]$,
  $S = k_\mathrm{fh}+k_\mathrm{hf}$, $F$ the mean of $e^{i\Delta\mathbf K\cdot\mathbf d}$ over the three
  fcc → hcp vectors, is set equal to the Chudley–Elliott form. For $k_\mathrm{hf} \gg k_\mathrm{fh}$,
  $\Gamma \to \tfrac23 k_\mathrm{fh}$: a third of the hcp visits return.

The measured rates are extracted from the vector graphics of the published figures
(each `experiment.py` documents its source).

### 7.2 PES correction

DFT places the hcp hollow too high on some surfaces. `splitting_scale` (§1.4) is
fixed by experiment independently of the rates where possible: by the measured
hcp–fcc ground-state splitting of H (from the fast/slow ratio, or from the site
populations of the jump-model fit). Where no splitting is measured (Pt), it is
scanned and the scale with the best H fit is taken. The barrier is left at its
DFT value and the high-temperature activation energy is checked against the data
rather than fitted.

### 7.3 Fitting

The frictions are fitted to the H rates by minimizing

$$
\chi^2 = \frac{1}{N}\sum_i \left(\frac{\ln k_\mathrm{model}(T_i) - \ln k_i}{\sigma_i}\right)^2,
\qquad \sigma_i = \tfrac12\ln(k_i^\mathrm{high}/k_i^\mathrm{low}),
$$

on a cubic spline of $\ln k$ through a log grid of frictions (`examples/util.py`).
D uses the H friction coefficient $\eta = m\gamma$ (§4.4) with nothing refitted, so
it tests the PES and the isotope scaling; a second H data set, where available, is
compared without fitting.

### 7.4 Ground-state friction

With a single friction the bias suppresses deep tunneling (§6.3), so the model has
no low-temperature plateau. `hydrogen_ground_friction.py` (Ni, Ru) adds a second set
of couplings: the periodic $\cos(\mathbf q_j\cdot\mathbf r)/q$, $\sin(\mathbf q_j\cdot\mathbf r)/q$ projected onto the two
ground bands at every $\mathbf k$ (fcc and hcp ground states), coupled to independent
baths with friction $\gamma_g$, so $\mathcal L = \mathcal L_0 + \gamma\mathcal L_1 + \gamma_g\mathcal L_g$. The projection
commutes with the mixing into $e^{\pm i\mathbf q_j\cdot\mathbf r}/q$ (§3.2).

This is a phenomenological placeholder for dissipation between the hollow ground
states that Ohmic friction misses (e.g. surface relaxation), not a friction in the
sense of §4. The projected operator is almost diagonal in site, so it mainly
dephases fcc against hcp; when $\hbar\Gamma_\phi \gg \varepsilon$ the rate becomes the overdamped
incoherent-tunneling rate $\sim 2t^2/\hbar^2\Gamma_\phi$, in which the bias drops out and which
rises on cooling. The values of γ_g needed for this lie far outside the validity
condition $2\pi\alpha \ll 1$ (§4.1).

Weak-coupling mechanisms that keep $2\pi\alpha \ll 1$ cannot produce the plateau: a
one-boson fcc → hcp transition has matrix element $\approx td/\varepsilon$ and grows with T, as do
couplings that modulate the tunneling or the site energy; a temperature-dependent
friction cannot raise the biased rate enough; and treating the bath exactly
(polaron, golden rule in $t$) only lowers the rate. Resonant tunneling between
degenerate hollows would give a plateau, but the measured hollow splittings rule
this out.

### 7.5 Populations and the exchange mode

Two simplifications of §6.1 can fail when the wells are shallow (Pt(111)):

* **Populations.** The fcc ground state can spill past the fcc/hcp midline, so a
  projector onto the region nearer hcp would overstate the hcp population.
  Populations are therefore assigned by eigenstate (§6.1), and top states, which
  belong to neither hollow, drop out of the detailed balance.
* **The slowest mode.** A weakly coupled top-site state can relax more slowly than
  the hollows exchange. The exchange mode is chosen among the six slowest as the
  one whose population change is closest to fcc ↔ hcp exchange, each hollow in
  internal equilibrium.

---

## 8. Numerical parameters and convergence

| Parameter | Default | Where | Checked against |
|---|---|---|---|
| Real-space grid | 18 × 18 × nz | `Grid` | 30 × 30 grids (0.01 meV) |
| z states | 6 (10 with low top-site states) | `Surface.n_z` | full z grid (≤ 0.1 meV for hollow bands) |
| Lamb-shift bands | 48 per k | `BlochSolver.n_states` | 72 (0.4%) |
| Master-equation bands | within `max_energy` of the ground band | `model.rates` | higher cutoffs (§5.1) |
| Supercell | n = 6 (4 for D where memory requires) | examples | n = 8–10 (`examples/nickel/convergence.py`) |

Convergence in $n$ is slowest at low temperature, where the deep-tunneling rate is
sensitive to how closely the periodic coupling approaches linear coupling ($q\,\Delta x \ll 1$).
The cutoffs and grids used for each surface, with their convergence, are recorded
in its `util.py`.
