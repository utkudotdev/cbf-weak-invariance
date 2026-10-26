import jax
import jax.numpy as jnp
from jaxtyping import Float

from .types import Control, Parameters, State


def control_bounds() -> tuple[Control, Control]:
    """Return componentwise force limits (lower, upper)."""
    return jnp.array([-0.001, -0.001]), jnp.array([0.001, 0.001])


def f(x: State, params: Parameters) -> State:
    """Single-particle drift: p_dot = v, v_dot = 0."""
    return jnp.array([x[2], x[3], 0.0, 0.0])


def g(x: State, params: Parameters) -> Float[jax.Array, "4 2"]:
    """Single-particle control matrix: p_dot = 0, v_dot = u / mass."""
    return jnp.array(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=x.dtype
    ) / params.mass
