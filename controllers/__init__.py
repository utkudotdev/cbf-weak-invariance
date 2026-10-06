from collections.abc import Callable
from typing import Any

import jax


type Controller[SystemStateT, ControlT, ControllerStateT] = Callable[
    [SystemStateT, ControllerStateT], tuple[ControlT, ControllerStateT]
]


def random(
    shape: tuple[int, ...],
) -> Controller[Any, jax.Array, jax.Array]:
    """Create a standard-normal random controller with a fixed output shape."""

    def controller(_, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        key, control_key = jax.random.split(key)
        control = jax.random.normal(control_key, shape)
        return control, key

    return controller


__all__ = ["random"]
