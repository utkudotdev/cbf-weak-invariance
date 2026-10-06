from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from . import Parameters, State, default_parameters


def plot(
    x: State,
    *,
    controls: Any | None = None,
    parameters: Parameters | None = None,
    dt: float | None = None,
    ax: Axes | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot agent positions and animate time-stacked trajectories."""
    del controls  # Part of the common environment plotting interface.

    positions = np.asarray(x.p)
    if positions.shape[-1:] != (2,):
        raise ValueError("particle positions must have shape (..., 2)")
    parameters = parameters or default_parameters()

    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    else:
        figure = ax.figure

    if positions.ndim == 1:
        positions = positions[np.newaxis, :]
    is_trajectory = positions.ndim == 3
    frames = positions if is_trajectory else positions[np.newaxis, ...]
    agent_count = frames.shape[1]

    colors = [f"C{index % 10}" for index in range(agent_count)]
    trails = [ax.plot([], [], color=color, linewidth=1.5)[0] for color in colors]
    centers = ax.scatter(
        frames[0, :, 0], frames[0, :, 1], s=30, c=colors, zorder=3
    )
    collision_disks = []
    for index, color in enumerate(colors):
        disk = Circle(
            (0, 0),
            parameters.collision_radius,
            facecolor=color,
            edgecolor=color,
            alpha=0.2,
            label="collision radius" if index == 0 else None,
        )
        ax.add_patch(disk)
        collision_disks.append(disk)

    margin = parameters.collision_radius * 1.5
    minimum = frames.min(axis=(0, 1)) - margin
    maximum = frames.max(axis=(0, 1)) + margin
    span = np.maximum(maximum - minimum, 2 * margin)
    center = (minimum + maximum) / 2
    half_extent = max(span) / 2
    ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
    ax.set_ylim(center[1] - half_extent, center[1] + half_extent)

    def update(frame_index: int) -> list[Any]:
        current = frames[frame_index]
        centers.set_offsets(current)
        for index, (trail, disk) in enumerate(zip(trails, collision_disks)):
            if is_trajectory:
                trail.set_data(
                    frames[: frame_index + 1, index, 0],
                    frames[: frame_index + 1, index, 1],
                )
            else:
                trail.set_data([], [])
            disk.center = current[index]
        return [*trails, centers, *collision_disks]

    update(0)
    ax.set(xlabel="x", ylabel="y", title="Particle simulation")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()

    animation = None
    if is_trajectory:
        interval = 1000 * (dt if dt is not None else 0.05)
        animation = FuncAnimation(
            figure,
            update,
            frames=len(frames),
            interval=interval,
            blit=True,
        )
        # Keep the animation alive for interactive callers after this returns.
        figure._particle_animation = animation  # type: ignore[attr-defined]

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".gif" and animation is not None:
            fps = max(1, round(1 / (dt if dt is not None else 0.05)))
            animation.save(output, writer=PillowWriter(fps=fps), dpi=100)
        else:
            update(len(frames) - 1)
            figure.savefig(output, dpi=160)
    if show:
        plt.show()

    return figure, ax
