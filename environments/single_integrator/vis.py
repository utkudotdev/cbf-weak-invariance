from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .types import Parameters, State
from .sampling import default_parameters


def plot(
    x: State,
    *,
    controls: Any | None = None,
    parameters: Parameters | None = None,
    dt: float | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, FuncAnimation | None]:
    """Plot agent positions and animate time-stacked trajectories."""
    del controls  # Part of the common environment plotting interface.

    states = np.asarray(x)
    if states.shape[-1:] != (2,):
        raise ValueError("single_integrator states must have shape (..., 2)")
    positions = states[..., :2]
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
    centers = ax.scatter(frames[0, :, 0], frames[0, :, 1], s=30, c=colors, zorder=3)
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
    ax.set(xlabel="x", ylabel="y", title="Single integrator simulation")
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
    update(len(frames) - 1)

    return figure, ax, animation
