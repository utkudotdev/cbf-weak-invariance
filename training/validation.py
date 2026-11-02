"""Fixed scenarios for evaluating physical safety under the current controller."""

import equinox as eqx
import jax
import jax.numpy as jnp

from training.batching import safety_margins


def sample_safety_scenarios(environment, count, agents, steps, key):
    """Sample initial states and bounded nominal controls once per training run."""
    state_key, control_key = jax.random.split(key)
    keys = jax.random.split(state_key, (count, agents))
    initial_states = jax.vmap(jax.vmap(environment.random_state))(keys)

    lower, upper = environment.control_bounds()
    nominal_controls = jax.random.uniform(
        control_key,
        (count, steps, agents, *lower.shape),
        minval=lower,
        maxval=upper,
    )

    return initial_states, nominal_controls


@eqx.filter_jit
def safety_metrics(environment, simulation, dt):
    """Summarize physical margins over all agents and times, including t=0.

    Violation magnitude is max(0, -h), in the environment's safety-margin units.
    A trajectory is unsafe if any agent has a negative margin at any time.
    Total violation time sums dt over unsafe intervals across trajectories,
    using each interval's starting state and counting simultaneous agents once.
    """
    margins = safety_margins(environment, simulation.states, simulation.parameters)
    violations = jnp.maximum(-margins, 0.0)
    unsafe = margins < 0

    return dict(
        safety_violation_mean=violations.mean(),
        safety_violation_worst=violations.max(),
        safety_violation_time_total=unsafe[:, :-1].any(axis=-1).sum() * dt,
        safety_unsafe_fraction=unsafe.mean(),
        safety_unsafe_trajectory_fraction=unsafe.any(axis=(1, 2)).mean(),
    )
