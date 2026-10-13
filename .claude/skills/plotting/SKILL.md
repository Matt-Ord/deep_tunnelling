---
name: plotting
description: Conventions for writing plot functions and figures in this repo (deep_tunnelling/plot.py, examples/experiment.py, example scripts). Use whenever adding or changing a plot function, or composing a figure in an example script.
---

# Plotting conventions

Plot functions draw one thing with sensible defaults; example scripts compose
figures and own all styling.

## Plot functions

- Signature: data arguments first, then a keyword-only `ax: Axes | None = None`.
  Return `(fig, ax, artist)` so the caller can style what was drawn.

  ```python
  def plot_rate_arrhenius(
      temperatures: Array, rates: Array, *, ax: Axes | None = None
  ) -> tuple[Figure, Axes, Line2D]:
      """Plot a model escape rate against 1000 / T on a log scale."""
      fig, ax = get_figure(ax)
      (line,) = ax.plot(1000 / np.asarray(temperatures), rates)
      return fig, ax, line
  ```

- Always start with `fig, ax = get_figure(ax)`, which creates a figure when `ax`
  is None and otherwise draws into the given axes.
- Use sensible defaults for what makes the plot readable on its own: marker,
  linestyle, axis scales and labels (e.g. `set_arrhenius_axes`).
- Do not take style parameters (color, label, linewidth, ...), and do not forward
  `**kwargs` to matplotlib. Never set colors or legend labels inside a plot
  function.
- One plot function draws one dataset or curve. Combining several on one figure is
  the script's job, not a library function's.

## Example scripts (`examples/*.py`)

- Create the figure with `fig, ax = get_fancy_figure()`.
- Call plot functions with `ax=ax` and style the returned artists:

  ```python
  _, _, slow = plot_rate_arrhenius(curve.temperatures, curve.slow, ax=ax)
  slow.set(color=CAM_BLUE.dark, label="slow")
  _, _, fast = plot_rate_arrhenius(curve.temperatures, curve.fast, ax=ax)
  fast.set(color=CAM_BLUE.dark, linestyle="--")
  ```

- Error-bar containers (`plot_experimental_rates`) are styled through their
  artists:

  ```python
  _, _, points = plot_experimental_rates(H_RATES, ax=ax)
  points.set_label("Zhu, slow")
  for artist in points.lines[0], *points.lines[2]:
      artist.set_color(CAM_CHERRY.dark)
  points.lines[0].set_markerfacecolor("none")  # open circles
  ```

- Colors come from the Cambridge palette in `deep_tunnelling/plot.py`
  (`CAM_BLUE`, `CAM_CHERRY`, `CAM_CREST`, ...).
- Legend, title and `fig.savefig(...)` are set in the script.
