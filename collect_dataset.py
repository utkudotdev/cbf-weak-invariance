import argparse
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
from utils import positive_float, positive_int


@dataclass(frozen=True)
class Dataset:
    """States (R, T+1, N, ...), controls (R, T, N, ...), and parameters."""

    states: Any
    controls: Any
    parameters: Any


def rollout(
    key: jax.Array,
    environment: Environment,
    parameters: Any,
    *,
    agents: int,
    steps: int,
    dt: float,
) -> Simulation[Any, Any, Any]:
    """Sample an initial state and roll out one controlled trajectory."""
    state_key, controller_key = jax.random.split(key)
    state_keys = jax.random.split(state_key, agents)
    initial_state = jax.vmap(environment.random_state)(state_keys)

    controller = make_random(initial_state.v.shape)
    simulation = simulate(
        initial_state,
        environment,
        controller,
        controller_key,
        parameters,
        steps=steps,
        dt=dt,
    )

    return simulation


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
    if min(num_rollouts, rollouts_per_batch, rollout_length, agents) < 1 or dt <= 0:
        raise ValueError("counts and dt must be positive")
    environment = load_environment(environment_name)
    parameters = environment.default_parameters()
    rollout_keys = jax.random.split(jax.random.key(seed), num_rollouts)

    def run(key: jax.Array) -> tuple[Any, Any]:
        simulation = rollout(
            key,
            environment,
            parameters,
            agents=agents,
            steps=rollout_length,
            dt=dt,
        )
        return simulation.states, simulation.controls

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
    states, controls = jax.tree.map(
        lambda leaf: leaf.reshape((num_batches * batch_size, *leaf.shape[2:]))[
            :num_rollouts
        ],
        batches,
    )
    jax.block_until_ready((states, controls))
    jax.effects_barrier()
    return Dataset(states=states, controls=controls, parameters=parameters)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect controlled trajectories.")
    parser.add_argument(
        "--environment",
        default="particle",
        help="module name below environments/ (default: particle)",
    )
    parser.add_argument("--num-rollouts", type=positive_int, default=100_000)
    parser.add_argument("--rollouts-per-batch", type=positive_int, default=4096)
    parser.add_argument("--rollout-length", type=positive_int, default=64)
    parser.add_argument("--dt", type=positive_float, default=0.05)
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

    velocities = dataset.states.v
    sample_axes = tuple(range(velocities.ndim - 1))
    speeds = jnp.linalg.norm(velocities, axis=-1)
    velocity_min, velocity_mean, velocity_max, speed_min, speed_mean, speed_max = (
        jax.device_get(
            (
                jnp.min(velocities, axis=sample_axes),
                jnp.mean(velocities, axis=sample_axes),
                jnp.max(velocities, axis=sample_axes),
                jnp.min(speeds),
                jnp.mean(speeds),
                jnp.max(speeds),
            )
        )
    )
    print(
        f"Velocity components: min={velocity_min}, mean={velocity_mean}, "
        f"max={velocity_max}"
    )
    print(
        f"Speed ||v||: min={speed_min:.6g}, mean={speed_mean:.6g}, max={speed_max:.6g}"
    )

    output_dir = args.output_dir or (
        Path("data") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_dir}")

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
    from gcbf.data import save_dataset

    save_dataset(output_dir, dataset, metadata)

    return output_dir


if __name__ == "__main__":
    main()
