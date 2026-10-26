from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from controllers import Controller
from environments import Environment, dynamics


@register_dataclass
@dataclass(frozen=True)
class Simulation[StateT, ControlT, ParametersT]:
    states: StateT
    controls: ControlT
    parameters: ParametersT


def euler_step(environment: Environment, state, control, parameters, dt: float):
    """Advance one agent's dynamics by one explicit Euler step."""
    velocity = dynamics(environment, state, control, parameters)
    return state + dt * velocity


@partial(jax.jit, static_argnames=("environment", "controller", "steps"))
def simulate[StateT, ControlT, ParametersT, ControllerStateT](
    initial_states: StateT,
    environment: Environment,
    controller: Controller[StateT, ControlT, ControllerStateT],
    initial_controller_state: ControllerStateT,
    parameters: ParametersT,
    *,
    steps: int,
    dt: float,
) -> Simulation[StateT, ControlT, ParametersT]:
    """Simulate bounded state feedback, recording the controls actually applied."""
    lower, upper = environment.control_bounds()

    def step(
        carry: tuple[StateT, ControllerStateT], _: None
    ) -> tuple[
        tuple[StateT, ControllerStateT],
        tuple[StateT, ControlT],
    ]:
        state, controller_state = carry
        control, next_controller_state = controller(state, controller_state)
        control = jnp.clip(control, lower, upper)
        next_state = jax.vmap(
            lambda x, u: euler_step(environment, x, u, parameters, dt)
        )(state, control)
        return (next_state, next_controller_state), (next_state, control)

    _, (following_states, controls) = jax.lax.scan(
        step,
        (initial_states, initial_controller_state),
        xs=None,
        length=steps,
    )
    states = jnp.concatenate((initial_states[jnp.newaxis], following_states), axis=0)
    return Simulation(states, controls, parameters)


__all__ = ["Simulation", "euler_step", "simulate"]
