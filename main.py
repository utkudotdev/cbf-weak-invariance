import argparse
import importlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass(frozen=True)
class Simulation:
    states: Any
    controls: Any
    parameters: Any


def load_environment(name: str) -> ModuleType:
    """Load an environment implementing the small simulation hook interface."""
    if not name.isidentifier():
        raise ValueError(f"invalid environment name: {name!r}")

    try:
        environment = importlib.import_module(f"environments.{name}")
    except ModuleNotFoundError as error:
        if error.name == f"environments.{name}":
            raise ValueError(f"unknown environment: {name!r}") from error
        raise

    required_hooks = (
        "default_parameters",
        "initial_state",
        "random_control",
        "f",
        "plot",
    )
    missing = [hook for hook in required_hooks if not callable(getattr(environment, hook, None))]
    if missing:
        raise TypeError(
            f"environment {name!r} is missing callable hooks: {', '.join(missing)}"
        )
    return environment


@partial(jax.jit, static_argnames=("environment", "steps"))
def simulate(
    environment: ModuleType,
    *,
    steps: int,
    dt: float,
    seed: int,
) -> Simulation:
    """JIT-compile and simulate an environment with random inputs."""
    key = jax.random.key(seed)
    key, initial_key = jax.random.split(key)
    state = environment.initial_state(initial_key)
    parameters = environment.default_parameters()

    def step(state: Any, control_key: jax.Array) -> tuple[Any, tuple[Any, Any]]:
        control = environment.random_control(control_key, state)
        next_state = environment.f(state, control, parameters, dt)
        return next_state, (next_state, control)

    control_keys = jax.random.split(key, steps)
    _, (following_states, controls) = jax.lax.scan(step, state, control_keys)
    states = jax.tree.map(
        lambda initial, following: jnp.concatenate(
            (initial[jnp.newaxis], following), axis=0
        ),
        state,
        following_states,
    )
    return Simulation(states, controls, parameters)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a configurable environment.")
    parser.add_argument(
        "--environment",
        default="particle",
        help="module name below environments/ (default: particle)",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="optional path for the rendered plot")
    parser.add_argument("--no-show", action="store_true", help="do not open a plot window")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Simulation:
    args = parse_args(argv)
    environment = load_environment(args.environment)
    simulation = simulate(
        environment,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
    )
    environment.plot(
        simulation.states,
        controls=simulation.controls,
        dt=args.dt,
        show=not args.no_show,
        save_path=args.output,
    )
    return simulation


if __name__ == "__main__":
    main()
