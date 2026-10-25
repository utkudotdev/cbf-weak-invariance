"""Translation-invariant state reduction for directed edges j -> i."""

import jax.numpy as jnp

from .types import State


def edge_feature(receiver: State, sender: State):
    """Return [p_j - p_i, v_i, v_j], broadcasting leading dimensions.

    Here v_i and v_j denote absolute particle velocities, not the node-type
    features denoted by v_i and v_j in GCBF+ Section IV-A. Those node features
    are provided separately by node_feature().
    """
    dp, vi, vj = jnp.broadcast_arrays(sender.p - receiver.p, receiver.v, sender.v)
    return jnp.concatenate((dp, vi, vj), axis=-1)


def node_feature(state: State):
    """Paper's agent/goal/obstacle type encoding; here all nodes are agents."""
    return jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (*state.p.shape[:-1], 3))
