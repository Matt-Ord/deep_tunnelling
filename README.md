# deep_tunnelling

Hop rates of H and D on close-packed metal surfaces from the universal
Caldeira-Leggett master equation (UME), compared with helium spin-echo
measurements: Ni(111) (J. Zhu, PhD thesis, Cambridge, 2015), Pt(111) (Jardine et
al., PRL 105, 136101 (2010)) and Ru(0001) (P. Townsend, PhD thesis, Cambridge,
2018; McIntosh et al., J. Phys. Chem. Lett. 4, 1565 (2013)).

- `deep_tunnelling/potential.py`: periodic PES interpolated from DFT energies at in-plane points, with z measured from the local equilibrium height.
- `deep_tunnelling/bloch.py`: Bloch eigenstates on an n x n supercell.
- `deep_tunnelling/bath.py`: Ohmic bath with the cutoff-free Lamb shift.
- `deep_tunnelling/ume.py`: periodic Caldeira-Leggett couplings and the UME Liouvillian.
- `deep_tunnelling/model.py`: escape-rate curves for sets of temperatures and frictions.

## Examples

```bash
# Ni(111): Zhu's data
uv run python -m examples.nickel.hydrogen                  # fit the friction to the H data
uv run python -m examples.nickel.hydrogen_ground_friction  # add a separate fcc <-> hcp ground-state friction
uv run python -m examples.nickel.deuterium                 # D with the same friction coefficients
uv run python -m examples.nickel.convergence               # convergence in supercell size
# Pt(111): Jardine's data
uv run python -m examples.platinum.hydrogen                # fit the friction to H on the DFT and corrected PES
uv run python -m examples.platinum.deuterium               # D with the same friction coefficients
# Ru(0001): Townsend's data
uv run python -m examples.ruthenium.hydrogen               # fit the friction to H on the DFT and corrected PES
uv run python -m examples.ruthenium.hydrogen_ground_friction  # add a separate fcc <-> hcp ground-state friction
uv run python -m examples.ruthenium.deuterium              # D with the same friction coefficients
```

Run the examples as modules from the repository root.

Each folder holds its DFT data (`raw_data.json`), the measured rates
(`experiment.py`), its surface (DFT data, grid and PES correction) and settings
(`util.py`), cached results (`data/`) and figures (`*.pdf`). `examples/util.py`
has what they share: `Surface`, which builds the PES and Bloch states of one
system, and the friction fits.

The Pt and Ru examples run about an hour each from scratch and keep at most 4
worker processes (~0.5 GB each for H; D runs two ~2 GB workers); `model.rates`
takes `max_workers` and `max_energy` to bound memory, which grows as the fourth
power of the number of bands.

## Development

```bash
uv sync --all-extras
ruff check . && ty check . && pytest
```
