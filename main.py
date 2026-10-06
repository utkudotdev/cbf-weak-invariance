import argparse
from pathlib import Path
from typing import Sequence

import jax

from controllers import random
from environments import load_environment
from simulation import simulate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a configurable environment.")
    parser.add_argument(
        "--environment",
        default="particle",
        help="module name below environments/ (default: particle)",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agents", type=int, default=8)
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
    controller = random(initial_states.v.shape)

    parameters = environment.default_parameters()
    simulation = simulate(
        initial_states,
        environment.f,
        controller,
        control_key,
        parameters,
        steps=args.steps,
        dt=args.dt,
    )
    environment.plot(
        simulation.states,
        controls=simulation.controls,
        parameters=simulation.parameters,
        dt=args.dt,
        show=not args.no_show,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
