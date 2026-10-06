import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from controllers import Controller, random
from environments import Environment, load_environment
from simulation import Simulation, simulate


NUM_ROLLOUTS = 100_000
NUM_ROLLOUTS_PER_BATCH = 4_096
MAX_ROLLOUT_LENGTH = 64
DT = 0.05
NUM_AGENTS = 5


@dataclass(frozen=True)
class Dataset:
    """Trajectories with ``hs`` shaped (rollouts, timesteps, agents)."""

    states: Any
    controls: Any
    hs: jax.Array
    parameters: Any


def make_controller(name: str, initial_state: Any) -> Controller[Any, Any, Any]:
    """Construct a controller for the sampled system state.

    Controller construction is intentionally explicit: controllers do not yet
    share enough configuration to support a useful generic factory.
    """
    if name == "random":
        try:
            control_shape = initial_state.v.shape
        except AttributeError as error:
            raise TypeError(
                "the random controller currently requires an environment state "
                "with a 'v' array whose shape is also the control shape"
            ) from error
        return random(control_shape)
    raise ValueError(f"unknown controller: {name!r}")


def rollout(
    key: jax.Array,
    environment: Environment,
    controller_name: str,
    parameters: Any,
    *,
    agents: int,
    steps: int,
    dt: float,
) -> tuple[Simulation[Any, Any, Any], jax.Array]:
    """Sample an initial state and roll out one controlled trajectory."""
    state_key, controller_key = jax.random.split(key)
    state_keys = jax.random.split(state_key, agents)
    initial_state = jax.vmap(environment.random_state)(state_keys)

    controller = make_controller(controller_name, initial_state)
    simulation = simulate(
        initial_state,
        environment.f,
        controller,
        controller_key,
        parameters,
        steps=steps,
        dt=dt,
    )

    def safety_values(states: Any) -> jax.Array:
        def agent_safety(index: jax.Array) -> jax.Array:
            other_indices = jnp.arange(agents - 1)
            other_indices = other_indices + (other_indices >= index)
            state = jax.tree.map(lambda leaf: leaf[index], states)
            others = jax.tree.map(lambda leaf: leaf[other_indices], states)
            return environment.h(state, others, parameters)

        return jax.vmap(agent_safety)(jnp.arange(agents))

    hs = jax.vmap(safety_values)(simulation.states)
    return simulation, hs


def collect_dataset(
    environment_name: str,
    controller_name: str,
    *,
    num_rollouts: int = NUM_ROLLOUTS,
    rollouts_per_batch: int = NUM_ROLLOUTS_PER_BATCH,
    rollout_length: int = MAX_ROLLOUT_LENGTH,
    dt: float = DT,
    agents: int = NUM_AGENTS,
    seed: int = 0,
) -> Dataset:
    """Collect trajectories for a named environment and controller."""
    environment = load_environment(environment_name)
    parameters = environment.default_parameters()
    rollout_keys = jax.random.split(jax.random.key(seed), num_rollouts)

    def run(key: jax.Array) -> tuple[Any, Any, jax.Array]:
        simulation, hs = rollout(
            key,
            environment,
            controller_name,
            parameters,
            agents=agents,
            steps=rollout_length,
            dt=dt,
        )
        return simulation.states, simulation.controls, hs

    states, controls, hs = jax.lax.map(
        run,
        rollout_keys,
        batch_size=rollouts_per_batch,
    )
    return Dataset(states=states, controls=controls, hs=hs, parameters=parameters)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect controlled trajectories and safety measurements."
    )
    parser.add_argument(
        "--environment",
        default="particle",
        help="module name below environments/ (default: particle)",
    )
    parser.add_argument(
        "--controller",
        default="random",
        choices=("random",),
        help="controller to use (default: random)",
    )
    parser.add_argument("--num-rollouts", type=int, default=NUM_ROLLOUTS)
    parser.add_argument(
        "--rollouts-per-batch", type=int, default=NUM_ROLLOUTS_PER_BATCH
    )
    parser.add_argument("--rollout-length", type=int, default=MAX_ROLLOUT_LENGTH)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--agents", type=int, default=NUM_AGENTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: a timestamped directory under data/)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    dataset = collect_dataset(
        args.environment,
        args.controller,
        num_rollouts=args.num_rollouts,
        rollouts_per_batch=args.rollouts_per_batch,
        rollout_length=args.rollout_length,
        dt=args.dt,
        agents=args.agents,
        seed=args.seed,
    )

    output_dir = args.output_dir or (
        Path("data") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_dir}")

    jnp.savez(
        output_dir / "dset.npz",
        states=dataset.states,
        controls=dataset.controls,
        hs=dataset.hs,
        params=dataset.parameters,
    )
    metadata = {
        "environment": args.environment,
        "controller": args.controller,
        "num_rollouts": args.num_rollouts,
        "rollouts_per_batch": args.rollouts_per_batch,
        "rollout_length": args.rollout_length,
        "dt": args.dt,
        "agents": args.agents,
        "seed": args.seed,
        "safety_convention": "h >= 0 is safe; h < 0 is a safety violation",
        "safety_axes": ["rollout", "timestep", "agent"],
    }
    with (output_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    return output_dir


if __name__ == "__main__":
    main()
