"""Two-agent slices of the learned GCBF and real single-integrator safety."""

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
from gcbf.checkpoint import load_checkpoint
from gcbf.gnn import make_graph


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--extent", type=positive_float, default=2.0)
    parser.add_argument(
        "--resolution", type=partial(int_at_least, minimum=2), default=100
    )
    parser.add_argument("--ego-position", nargs=2, type=float, default=(0.0, 0.0))
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
    if config["environment"] != "single_integrator":
        raise ValueError("this viewer requires a single_integrator checkpoint")

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
    def evaluate():
        def one(p):
            state = jnp.stack((jnp.asarray(ego_position), p))
            learned = model(
                make_graph(
                    environment,
                    state,
                    radius,
                )
            )[:-1]  # make_graph appends a dummy node for inactive edges.
            real = jnp.stack(
                (
                    environment.h(state[0], state[1:], parameters),
                    environment.h(state[1], state[:1], parameters),
                )
            )
            return learned, real

        return jax.vmap(one)(points)

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.subplots_adjust(bottom=0.28, right=0.84)
    values, safety_values = map(np.asarray, evaluate())
    image = ax.imshow(
        values[:, 0].reshape(xx.shape),
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="RdBu",
        vmin=-args.h_limit,
        vmax=args.h_limit,
    )
    colorbar = fig.colorbar(
        image, ax=ax, label="learned h (positive = predicted safe)"
    )
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
    function_radio = RadioButtons(
        fig.add_axes((0.18, 0.105, 0.3, 0.075)), ("learned h", "real safety h")
    )
    contour = None

    def update(_=None):
        nonlocal contour
        index = 0 if radio.value_selected == "h agent 0" else 1
        show_real = function_radio.value_selected == "real safety h"
        selected_values = safety_values if show_real else values
        z = selected_values[:, index].reshape(xx.shape)
        image.set_data(z)
        colorbar.set_label(
            "real safety h (positive = collision-free)"
            if show_real
            else "learned h (positive = predicted safe)"
        )
        if contour is not None:
            contour.remove()
            contour = None
        if z.min() < 0 < z.max():
            contour = ax.contour(xx, yy, z, levels=[0], colors="lime", linewidths=1.5)
        function_name = "Real safety function" if show_real else "GCBF"
        ax.set_title(
            f"{function_name} for agent {index}: agent 1 position slice\n"
            "green: h=0; dashed: physical collision boundary"
        )
        fig.canvas.draw_idle()

    radio.on_clicked(update)
    function_radio.on_clicked(update)
    update()

    # Retain widgets for programmatic callers and interactive backends.
    fig._gcbf_widgets = (radio, scale_slider, function_radio)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
    if not args.no_show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    main()
