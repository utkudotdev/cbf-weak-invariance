import jax

from .types import Parameters, State


def default_parameters() -> Parameters:
    return Parameters(collision_radius=0.2)


def random_state(key: jax.Array) -> State:
    return jax.random.uniform(key, (2,), minval=-3.0, maxval=3.0)
