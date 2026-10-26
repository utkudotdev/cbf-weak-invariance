from dataclasses import dataclass

import jax
from jax.tree_util import register_dataclass
from jaxtyping import Float


@register_dataclass
@dataclass
class Parameters:
    mass: float
    collision_radius: float


# Components: [position_x, position_y, velocity_x, velocity_y].
type State = Float[jax.Array, "*batch 4"]
type Control = Float[jax.Array, "*batch 2"]
