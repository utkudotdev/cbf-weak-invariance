from collections.abc import Callable
from typing import Any

import jax


type Controller[SystemStateT, ControlT, ControllerStateT] = Callable[
    [SystemStateT, ControllerStateT], tuple[ControlT, ControllerStateT]
]


def make_random(
    shape: tuple[int, ...],
    std: float = 1.0,
) -> Controller[Any, jax.Array, jax.Array]:
    """Create a zero-mean Gaussian controller with a fixed output shape."""
    if not std >= 0:
        raise ValueError("standard deviation must be nonnegative")

    def controller(_, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        key, control_key = jax.random.split(key)
        control = std * jax.random.normal(control_key, shape)
        return control, key

    return controller


__all__ = ["make_random"]
