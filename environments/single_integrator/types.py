from dataclasses import dataclass

import jax
from jax.tree_util import register_dataclass
from jaxtyping import Float


@register_dataclass
@dataclass
class Parameters:
    collision_radius: float


# Components: [position_x, position_y].
type State = Float[jax.Array, "*batch 2"]
type Control = Float[jax.Array, "*batch 2"]
