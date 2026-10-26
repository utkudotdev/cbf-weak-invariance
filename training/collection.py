"""Small, fresh rollout batches for online training."""

from typing import Any

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm

from controllers import make_random
from environments import Environment
from simulation import Simulation, simulate


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
    environment: Environment,
    parameters: Any,
    *,
    num_rollouts: int,
    rollouts_per_batch: int,
    rollout_length: int,
    dt: float,
    agents: int,
    key: jax.Array,
) -> Simulation[Any, Any, Any]:
    """Collect a fresh in-memory batch using random controls."""
    if min(num_rollouts, rollouts_per_batch, rollout_length, agents) < 1 or dt <= 0:
        raise ValueError("counts and dt must be positive")
    rollout_keys = jax.random.split(key, num_rollouts)

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
    return Simulation(states=states, controls=controls, parameters=parameters)
