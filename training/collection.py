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
    initial_state=None,
    nominal_controls=None,
) -> Simulation[Any, Any, Any]:
    """Roll out a sampled or supplied state with optional nominal controls."""
    if initial_state is None:
        state_keys = jax.random.split(key, agents)
        initial_state = jax.vmap(environment.random_state)(state_keys)

    def controller(state, index):
        nominal = None if nominal_controls is None else nominal_controls[index]
        result = qp_action(
            model,
            environment,
            state,
            parameters,
            sensing_radius,
            qp_config,
            nominal_control=nominal,
        )
        return result.control, index + 1

    simulation = simulate(
        initial_state,
        environment,
        controller,
        jnp.array(0),
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
    initial_states=None,
    nominal_controls=None,
) -> Simulation[Any, Any, Any]:
    """Collect current-policy rollouts, optionally using fixed scenario inputs."""
    rollout_keys = jax.random.split(key, num_rollouts)

    def run(index: jax.Array) -> tuple[Any, Any]:
        simulation = rollout(
            rollout_keys[index],
            model,
            environment,
            parameters,
            agents=agents,
            steps=rollout_length,
            dt=dt,
            sensing_radius=sensing_radius,
            qp_config=qp_config,
            initial_state=None if initial_states is None else initial_states[index],
            nominal_controls=(
                None if nominal_controls is None else nominal_controls[index]
            ),
        )
        return simulation.states, simulation.controls

    batch_size = min(rollouts_per_batch, num_rollouts)
    num_batches = (num_rollouts + batch_size - 1) // batch_size
    # Repeat the final scenario to fill a partial batch, then discard padding.
    indices = jnp.minimum(jnp.arange(num_batches * batch_size), num_rollouts - 1)
    index_batches = indices.reshape(num_batches, batch_size)

    def run_batch(carry, batch_indices):
        return carry, jax.vmap(run)(batch_indices)

    _, batches = jax.lax.scan(
        run_batch,
        None,
        index_batches,
    )
    states, controls = jax.tree.map(
        lambda leaf: leaf.reshape((num_batches * batch_size, *leaf.shape[2:]))[
            :num_rollouts
        ],
        batches,
    )
    return Simulation(states=states, controls=controls, parameters=parameters)
