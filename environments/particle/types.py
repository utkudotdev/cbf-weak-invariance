from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Float


@register_dataclass
@dataclass
class Parameters:
    mass: float
    collision_radius: float


@register_dataclass
@dataclass
class State:
    p: Float[jax.Array, "*batch 2"]
    v: Float[jax.Array, "*batch 2"]


type Control = Float[jax.Array, "*batch 2"]
