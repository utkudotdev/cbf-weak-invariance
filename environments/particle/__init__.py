from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Float


@register_dataclass
@dataclass
class Parameters:
    mass: float
    collision_radius: float


@register_dataclass
@dataclass
class State:
    p: Float[jax.Array, "*batch 2"]
    v: Float[jax.Array, "*batch 2"]


type Control = Float[jax.Array, "*batch 2"]


def f(x: State, u: Control, params: Parameters, dt: float) -> State:
    return State(p=x.p + dt * x.v, v=x.v + dt * u / params.mass)


def h(x: State, others: State, params: Parameters) -> jax.Array:
    """Return the main agent's minimum collision margin against other agents.

    ``x`` contains one agent and ``others`` has an additional penultimate
    agent dimension, excluding the main agent. Any leading batch dimensions
    are preserved. Nonnegative values are safe; no other agents gives infinity.
    """
    separation = x.p[..., jnp.newaxis, :] - others.p
    squared_distance = jnp.sum(separation**2, axis=-1)
    minimum_separation = 2 * params.collision_radius
    pairwise_margin = squared_distance - minimum_separation**2
    return jnp.min(pairwise_margin, axis=-1, initial=jnp.inf)


def default_parameters() -> Parameters:
    """Return the parameters used by the example simulation."""
    return Parameters(mass=1.0, collision_radius=0.2)


def random_state(key: jax.Array) -> State:
    """Sample a stationary particle for the example runner."""
    position = jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0)
    return State(p=position, v=jnp.zeros_like(position))


# Import this last so vis can import State without creating a circular import.
from .vis import plot  # noqa: E402


__all__ = [
    "Control",
    "Parameters",
    "State",
    "default_parameters",
    "f",
    "h",
    "plot",
    "random_state",
]
