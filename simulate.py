import argparse
from pathlib import Path
from typing import Sequence

import jax
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from controllers import make_random
from environments import load_environment
from simulation import simulate
from utils import positive_float, positive_int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a configurable environment.")
    parser.add_argument(
        "--environment",
        default="double_integrator",
        help="module name below environments/ (default: double_integrator)",
    )
    parser.add_argument("--steps", type=positive_int, default=50)
    parser.add_argument("--dt", type=positive_float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", type=positive_int, default=8)
    parser.add_argument(
        "--output", type=Path, help="optional path for the rendered plot"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="do not open a plot window"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    environment = load_environment(args.environment)
    key = jax.random.key(args.seed)
    state_key, control_key = jax.random.split(key)
    state_keys = jax.random.split(state_key, args.agents)
    initial_states = jax.vmap(environment.random_state)(state_keys)
    lower, _ = environment.control_bounds()
    controller = make_random((args.agents, *lower.shape))

    parameters = environment.default_parameters()
    simulation = simulate(
        initial_states,
        environment,
        controller,
        control_key,
        parameters,
        steps=args.steps,
        dt=args.dt,
    )
    figure, _, animation = environment.plot(
        simulation.states,
        controls=simulation.controls,
        parameters=simulation.parameters,
        dt=args.dt,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".gif" and animation is not None:
            fps = max(1, round(1 / args.dt))
            animation.save(args.output, writer=PillowWriter(fps=fps), dpi=100)
        else:
            figure.savefig(args.output, dpi=160)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
