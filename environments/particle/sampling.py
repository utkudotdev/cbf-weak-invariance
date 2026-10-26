import jax
import jax.numpy as jnp

from .types import Parameters, State


def default_parameters() -> Parameters:
    return Parameters(mass=1.0, collision_radius=0.2)


def random_state(key: jax.Array) -> State:
    position = jax.random.uniform(key, (2,), minval=-3.0, maxval=3.0)
    velocity = jax.random.uniform(key, (2,), minval=-5.0, maxval=5.0)
    return jnp.concatenate((position, velocity))
