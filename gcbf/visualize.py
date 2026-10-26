"""Two-agent slices of a learned particle GCBF with interactive velocity sliders."""

import argparse
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

from environments import load_environment
from utils import int_at_least, positive_float
from .checkpoint import load_checkpoint
from .gnn import make_graph


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--extent", type=positive_float, default=2.0)
    parser.add_argument(
        "--resolution", type=partial(int_at_least, minimum=2), default=100
    )
    parser.add_argument("--ego-position", nargs=2, type=float, default=(0.0, 0.0))
    parser.add_argument("--ego-velocity", nargs=2, type=float, default=(0.0, 0.0))
    parser.add_argument("--other-velocity", nargs=2, type=float, default=(0.0, 0.0))
    parser.add_argument("--velocity-limit", type=positive_float, default=3.0)
    parser.add_argument(
        "--h-limit",
        type=positive_float,
        default=1.0,
        help="initial symmetric color range [-h-limit, h-limit] (default: 1)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)
    model, config = load_checkpoint(args.checkpoint)
    if config["environment"] != "particle":
        raise ValueError(
            "the two-agent position viewer currently supports particle only"
        )

    environment = load_environment(config["environment"])
    parameters = environment.Parameters(**config["parameters"])
    radius = config["sensing_radius"] or float("inf")

    ego_position = np.array(args.ego_position)
    x = np.linspace(
        ego_position[0] - args.extent, ego_position[0] + args.extent, args.resolution
    )
    y = np.linspace(
        ego_position[1] - args.extent, ego_position[1] + args.extent, args.resolution
    )
    xx, yy = np.meshgrid(x, y)
    points = jnp.asarray(np.stack((xx.ravel(), yy.ravel()), axis=-1))

    @eqx.filter_jit
    def evaluate(velocity):
        def one(p):
            state = jnp.concatenate(
                (jnp.stack((jnp.asarray(ego_position), p)), velocity.reshape(2, 2)),
                axis=-1,
            )
            return model(
                make_graph(
                    environment,
                    state,
                    radius,
                )
            )[:-1]  # make_graph appends a dummy node for inactive edges.

        return jax.vmap(one)(points)

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.subplots_adjust(bottom=0.35, right=0.84)
    initial = np.array([*args.ego_velocity, *args.other_velocity])
    bound = max(args.velocity_limit, float(abs(initial).max()))
    values = np.asarray(evaluate(jnp.asarray(initial)))
    image = ax.imshow(
        values[:, 0].reshape(xx.shape),
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="RdBu",
        vmin=-args.h_limit,
        vmax=args.h_limit,
    )
    fig.colorbar(image, ax=ax, label="learned h (positive = predicted safe)")
    ax.scatter(*ego_position, c="black", marker="x", label="fixed agent 0")
    ax.add_patch(
        Circle(
            ego_position,
            2 * parameters.collision_radius,
            fill=False,
            color="black",
            linestyle="--",
            label="collision boundary",
        )
    )
    ax.set(xlabel="agent 1 position x", ylabel="agent 1 position y", aspect="equal")
    ax.legend(loc="upper left")

    sliders = []
    for i, label in enumerate(("ego vx", "ego vy", "other vx", "other vy")):
        slider_ax = fig.add_axes((0.18, 0.25 - i * 0.045, 0.55, 0.022))
        sliders.append(Slider(slider_ax, label, -bound, bound, valinit=initial[i]))

    # A logarithmic control makes small barrier values easy to inspect.
    log_h_limit = np.log10(args.h_limit)
    scale_slider = Slider(
        fig.add_axes((0.18, 0.055, 0.55, 0.022)),
        "color ±h",
        min(-6.0, log_h_limit),
        max(0.0, log_h_limit),
        valinit=log_h_limit,
    )

    def update_color_scale(log_limit):
        limit = 10.0**log_limit
        image.set_clim(-limit, limit)
        scale_slider.valtext.set_text(f"±{limit:.3g}")
        fig.canvas.draw_idle()

    scale_slider.on_changed(update_color_scale)
    update_color_scale(log_h_limit)

    radio = RadioButtons(
        fig.add_axes((0.85, 0.11, 0.14, 0.12)), ("h agent 0", "h agent 1")
    )
    contour = None

    def update(_=None):
        nonlocal contour
        index = 0 if radio.value_selected == "h agent 0" else 1
        z = np.asarray(evaluate(jnp.asarray([s.val for s in sliders])))[
            :, index
        ].reshape(xx.shape)
        image.set_data(z)
        if contour is not None:
            contour.remove()
            contour = None
        if z.min() < 0 < z.max():
            contour = ax.contour(xx, yy, z, levels=[0], colors="lime", linewidths=1.5)
        ax.set_title(
            f"GCBF for agent {index}: agent 1 position slice\n"
            "green: h=0; dashed: physical collision boundary"
        )
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)
    radio.on_clicked(update)
    update()

    # Retain widgets for programmatic callers and interactive backends.
    fig._gcbf_widgets = (sliders, radio, scale_slider)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
    if not args.no_show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    main()
