from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import State


def plot(
    x: State,
    *,
    controls: Any | None = None,
    dt: float | None = None,
    ax: Axes | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot a particle position or its time-stacked trajectory."""
    del controls, dt  # Part of the common environment plotting interface.

    positions = np.asarray(x.p)
    if positions.shape[-1:] != (2,):
        raise ValueError("particle positions must have shape (..., 2)")

    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    else:
        figure = ax.figure

    if positions.ndim == 2:
        ax.plot(positions[:, 0], positions[:, 1], linewidth=1.5)
        final_position = positions[-1]
    else:
        final_position = positions

    ax.scatter(
        final_position[0],
        final_position[1],
        s=45,
        color="tab:blue",
        zorder=3,
        label="final position" if positions.ndim == 2 else "position",
    )
    ax.set(xlabel="x", ylabel="y", title="Particle simulation")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend()

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
    if show:
        plt.show()

    return figure, ax
