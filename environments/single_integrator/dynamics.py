import jax
import jax.numpy as jnp
from jaxtyping import Float

from .types import Control, Parameters, State


def control_bounds() -> tuple[Control, Control]:
    return jnp.array([-1, -1]), jnp.array([1, 1])


def f(x: State, params: Parameters) -> State:
    """Single-integrator drift: p_dot = 0."""
    return jnp.zeros_like(x)


def g(x: State, params: Parameters) -> Float[jax.Array, "2 2"]:
    """Single-integrator control matrix: p_dot = u."""
    return jnp.eye(2, dtype=x.dtype)
