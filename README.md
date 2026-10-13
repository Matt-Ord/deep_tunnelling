# deep_tunnelling

Hop rates of H and D on Ni(111) from the universal Caldeira-Leggett master
equation (UME), compared with the HeSE measurements of J. Zhu (PhD thesis,
Cambridge, 2015).

- `deep_tunnelling/potential.py`: periodic PES fitted to six-site DFT data.
- `deep_tunnelling/bloch.py`: Bloch eigenstates on an n x n supercell.
- `deep_tunnelling/bath.py`: Ohmic bath with the cutoff-free Lamb shift.
- `deep_tunnelling/ume.py`: periodic Caldeira-Leggett couplings and the UME Liouvillian.
- `deep_tunnelling/model.py`: escape-rate curves for sets of temperatures and frictions.

## Examples

```bash
uv run python examples/hydrogen.py                  # fit the friction to the H data
uv run python examples/hydrogen_ground_friction.py  # add a separate fcc <-> hcp ground-state friction
uv run python examples/deuterium.py                 # D with the same friction coefficients
uv run python examples/convergence.py               # convergence in supercell size
```

Shared fitting helpers are in `examples/util.py`. Results are cached in
`examples/data/` and figures written to `examples/*.pdf`.

## Development

```bash
uv sync --all-extras
ruff check . && ty check . && pytest
```
