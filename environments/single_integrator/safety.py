import jax
import jax.numpy as jnp

from .types import Parameters, State


def h(x: State, others: State, params: Parameters) -> jax.Array:
    """Minimum squared separation margin between collision disks."""
    separation = x[..., jnp.newaxis, :] - others
    squared_distance = jnp.sum(separation**2, axis=-1)
    minimum_separation = 2 * params.collision_radius
    pairwise_margin = squared_distance - minimum_separation**2
    return jnp.min(pairwise_margin, axis=-1, initial=jnp.inf)
