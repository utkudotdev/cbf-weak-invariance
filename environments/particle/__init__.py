from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Float


@register_dataclass
@dataclass
class Parameters:
    mass: float


@register_dataclass
@dataclass
class State:
    p: Float[jax.Array, "*batch 2"]
    v: Float[jax.Array, "*batch 2"]


type Control = Float[jax.Array, "*batch 2"]


def f(x: State, u: Control, params: Parameters, dt: float) -> State:
    return State(p=x.p + dt * x.v, v=x.v + dt * u / params.mass)


def default_parameters() -> Parameters:
    """Return the parameters used by the example simulation."""
    return Parameters(mass=1.0)


def initial_state(key: jax.Array) -> State:
    """Sample a stationary particle for the example runner."""
    position = jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0)
    return State(p=position, v=jnp.zeros_like(position))


def random_control(key: jax.Array, x: State) -> Control:
    """Sample an independent standard-normal force in each direction."""
    return jax.random.normal(key, x.v.shape)


# Import this last so vis can import State without creating a circular import.
from .vis import plot  # noqa: E402


__all__ = [
    "Control",
    "Parameters",
    "State",
    "default_parameters",
    "f",
    "initial_state",
    "plot",
    "random_control",
]
