"""Small, fresh rollout batches for online training."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from environments import Environment
from gcbf import GCBF
from simulation import Simulation, simulate
from training.qp import QPConfig, qp_action


def rollout(
    key: jax.Array,
    model: GCBF,
    environment: Environment,
    parameters: Any,
    *,
    agents: int,
    steps: int,
    dt: float,
    sensing_radius: float,
    qp_config: QPConfig,
) -> Simulation[Any, Any, Any]:
    """Sample an initial state and roll out one controlled trajectory."""
    state_keys = jax.random.split(key, agents)
    initial_state = jax.vmap(environment.random_state)(state_keys)

    def controller(state, controller_state):
        result = qp_action(
            model, environment, state, parameters, sensing_radius, qp_config
        )
        return result.control, controller_state

    simulation = simulate(
        initial_state,
        environment,
        controller,
        None,
        parameters,
        steps=steps,
        dt=dt,
    )

    return simulation


@eqx.filter_jit
def collect_rollouts(
    model: GCBF,
    environment: Environment,
    parameters: Any,
    *,
    num_rollouts: int,
    rollouts_per_batch: int,
    rollout_length: int,
    dt: float,
    sensing_radius: float,
    qp_config: QPConfig,
    agents: int,
    key: jax.Array,
) -> Simulation[Any, Any, Any]:
    """Collect a fresh in-memory batch using the current QP controller."""
    rollout_keys = jax.random.split(key, num_rollouts)

    def run(key: jax.Array) -> tuple[Any, Any]:
        simulation = rollout(
            key,
            model,
            environment,
            parameters,
            agents=agents,
            steps=rollout_length,
            dt=dt,
            sensing_radius=sensing_radius,
            qp_config=qp_config,
        )
        return simulation.states, simulation.controls

    batch_size = min(rollouts_per_batch, num_rollouts)
    num_batches = (num_rollouts + batch_size - 1) // batch_size
    # Repeat the final key to fill a partial batch, then discard padded outputs.
    indices = jnp.minimum(jnp.arange(num_batches * batch_size), num_rollouts - 1)
    key_batches = rollout_keys[indices].reshape(num_batches, batch_size)

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
    return Simulation(states=states, controls=controls, parameters=parameters)
