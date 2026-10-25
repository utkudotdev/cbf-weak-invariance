import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm

from controllers import make_random
from environments import Environment, load_environment
from simulation import Simulation, simulate


@dataclass(frozen=True)
class Dataset:
    """Trajectories with ``hs`` shaped (rollouts, timesteps, agents)."""

    states: Any
    controls: Any
    hs: jax.Array
    parameters: Any


def rollout(
    key: jax.Array,
    environment: Environment,
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

    controller = make_random(initial_state.v.shape)
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
    *,
    num_rollouts: int,
    rollouts_per_batch: int,
    rollout_length: int,
    dt: float,
    agents: int,
    seed: int,
) -> Dataset:
    """Collect trajectories for a named environment using random controls."""
    environment = load_environment(environment_name)
    parameters = environment.default_parameters()
    rollout_keys = jax.random.split(jax.random.key(seed), num_rollouts)

    def run(key: jax.Array) -> tuple[Any, Any, jax.Array]:
        simulation, hs = rollout(
            key,
            environment,
            parameters,
            agents=agents,
            steps=rollout_length,
            dt=dt,
        )
        return simulation.states, simulation.controls, hs

    batch_size = min(rollouts_per_batch, num_rollouts)
    num_batches = (num_rollouts + batch_size - 1) // batch_size
    # Repeat the final key to fill a partial batch, then discard padded outputs.
    indices = jnp.minimum(jnp.arange(num_batches * batch_size), num_rollouts - 1)
    key_batches = rollout_keys[indices].reshape(num_batches, batch_size)

    @scan_tqdm(num_batches, print_rate=1, desc="Collecting trajectories", unit="batch")
    def run_batch(carry: None, xs: tuple[jax.Array, jax.Array]):
        _, keys = xs
        return carry, jax.vmap(run)(keys)

    _, batches = jax.lax.scan(
        run_batch,
        None,
        (jnp.arange(num_batches), key_batches),
    )
    states, controls, hs = jax.tree.map(
        lambda leaf: leaf.reshape((num_batches * batch_size, *leaf.shape[2:]))[
            :num_rollouts
        ],
        batches,
    )
    jax.block_until_ready((states, controls, hs))
    jax.effects_barrier()
    return Dataset(states=states, controls=controls, hs=hs, parameters=parameters)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect controlled trajectories and safety measurements."
    )
    parser.add_argument(
        "--environment",
        default="particle",
        help="module name below environments/ (default: particle)",
    )
    parser.add_argument("--num-rollouts", type=positive_int, default=100_000)
    parser.add_argument("--rollouts-per-batch", type=positive_int, default=4096)
    parser.add_argument("--rollout-length", type=positive_int, default=64)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--agents", type=positive_int, default=5)
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
        "controller": "random",
        "num_rollouts": args.num_rollouts,
        "rollouts_per_batch": args.rollouts_per_batch,
        "rollout_length": args.rollout_length,
        "dt": args.dt,
        "agents": args.agents,
        "seed": args.seed,
    }
    with (output_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    print("Safety values across all trajectories, timesteps, and agents:")
    print(f"  Min: {float(jnp.min(dataset.hs)):.6g}")
    print(f"  Mean: {float(jnp.mean(dataset.hs)):.6g}")
    print(f"  Max: {float(jnp.max(dataset.hs)):.6g}")

    return output_dir


if __name__ == "__main__":
    main()
