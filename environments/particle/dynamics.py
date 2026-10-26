import jax.numpy as jnp

from .types import Control, Parameters, State


def control_bounds() -> tuple[Control, Control]:
    """Return componentwise force limits (lower, upper)."""
    return jnp.array([-0.0, -0.0]), jnp.array([0.0, 0.0])


def f(x: State, u: Control, params: Parameters, dt: float) -> State:
    return State(p=x.p + dt * x.v, v=x.v + dt * u / params.mass)
