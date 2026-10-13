from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deep_tunnelling.bath import OhmicBath
from deep_tunnelling.potential import SITES, SurfacePotential, load_site_curves
from deep_tunnelling.units import HBAR, KB, M_H

RAW_DATA = Path(__file__).parent.parent / "examples" / "raw_data.json"


def test_potential_reproduces_dft_sites() -> None:
    z, curves = load_site_curves(RAW_DATA)
    potential = SurfacePotential.from_dft(RAW_DATA)
    for name, (u, v) in SITES.items():
        fitted = potential(np.full(z.shape, u), np.full(z.shape, v), z)
        np.testing.assert_allclose(
            fitted, np.minimum(curves[name], potential.v_cap), atol=1e-8
        )


def test_bath_detailed_balance() -> None:
    bath = OhmicBath(M_H, 2.0, 150.0)
    w = np.linspace(1, 300, 7)
    np.testing.assert_allclose(bath.s(-w), np.exp(-HBAR * w / (KB * 150.0)) * bath.s(w))
    assert bath.s(np.array([0.0]))[0] == pytest.approx(2 * bath.eta * bath.kt)


def test_cutoff_free_lamb_shift_matches_matsubara_series() -> None:
    """Compare the closed form with its Matsubara series.

    P(nu) = hbar eta theta sum_n nu^3 / (nu_n (nu^2 + nu_n^2)), the series with the
    divergent constant and linear terms removed (theta = 2 kT / hbar, nu_n = 2 pi n kT / hbar).
    """
    bath = OhmicBath(M_H, 1.0, 150.0)
    nu1 = 2 * np.pi * bath.kt / HBAR
    n = np.arange(1, 200_001)[:, None]
    nu = np.array([0.5, 15.0, -130.0, 400.0])
    series = (
        HBAR
        * bath.eta
        * (2 * bath.kt / HBAR)
        * np.sum(nu**3 / (nu1 * n * (nu**2 + (nu1 * n) ** 2)), axis=0)
    )
    np.testing.assert_allclose(bath.principal_value(nu), series, rtol=1e-5)
    # large-nu limit: the standard Ohmic Lamb shift, (hbar eta nu / pi) ln(hbar nu / 2 pi kT)
    big = np.array([1e5])
    x = HBAR * big / (2 * np.pi * bath.kt)
    assert bath.principal_value(big)[0] == pytest.approx(
        HBAR * bath.eta * big[0] / np.pi * (np.log(x[0]) + np.euler_gamma), rel=1e-6
    )
