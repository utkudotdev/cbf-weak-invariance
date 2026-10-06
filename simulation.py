from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from controllers import Controller


@register_dataclass
@dataclass(frozen=True)
class Simulation[StateT, ControlT, ParametersT]:
    states: StateT
    controls: ControlT
    parameters: ParametersT


@partial(jax.jit, static_argnames=("dynamics", "controller", "steps"))
def simulate[StateT, ControlT, ParametersT, ControllerStateT](
    initial_states: StateT,
    dynamics: Callable[[StateT, ControlT, ParametersT, float], StateT],
    controller: Controller[StateT, ControlT, ControllerStateT],
    initial_controller_state: ControllerStateT,
    parameters: ParametersT,
    *,
    steps: int,
    dt: float,
) -> Simulation[StateT, ControlT, ParametersT]:
    """Simulate a system under state feedback using a JIT-compiled scan."""

    def step(
        carry: tuple[StateT, ControllerStateT], _: None
    ) -> tuple[
        tuple[StateT, ControllerStateT],
        tuple[StateT, ControlT],
    ]:
        state, controller_state = carry
        control, next_controller_state = controller(state, controller_state)
        next_state = dynamics(state, control, parameters, dt)
        return (next_state, next_controller_state), (next_state, control)

    _, (following_states, controls) = jax.lax.scan(
        step,
        (initial_states, initial_controller_state),
        xs=None,
        length=steps,
    )
    states = jax.tree.map(
        lambda initial, following: jnp.concatenate(
            (initial[jnp.newaxis], following), axis=0
        ),
        initial_states,
        following_states,
    )
    return Simulation(states, controls, parameters)


__all__ = ["Simulation", "simulate"]
