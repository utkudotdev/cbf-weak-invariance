"""Translation-invariant features for directed edges j -> i."""

import jax
import jax.numpy as jnp
from jaxtyping import Float

from .types import State


def squared_distance(receiver: State, sender: State) -> Float[jax.Array, "*batch"]:
    """Squared Euclidean position distance, broadcasting leading dimensions."""
    return jnp.sum((sender - receiver) ** 2, axis=-1)


def edge_feature(receiver: State, sender: State):
    """Return relative position p_j - p_i."""
    return sender - receiver


def node_feature(state: State):
    """Agent type encoding, shared with the particle environment."""
    return jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (*state.shape[:-1], 3))
