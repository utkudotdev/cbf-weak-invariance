"""Equinox single-hop attention GCBF, following GCBF+ Eq. (18).

https://arxiv.org/html/2401.14554v4#S4.SS1
Edge lists support arbitrary connectivity; padding masks keep construction JIT-safe.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
from jaxtyping import Float, PRNGKeyArray

from environments import Environment


def make_graph[StateT](
    environment: Environment,
    state: StateT,
    sensing_radius: float,
) -> jraph.GraphsTuple:
    """Build directed non-self edges using the environment's features and distance.

    Reserve N*(N-1) slots and redirect inactive edges to a final dummy node,
    following the authors' implementation. Shapes stay fixed under JIT/vmap.
    The GNN returns one value per node; callers discard the final dummy value.
    """
    nodes = environment.node_feature(state)
    n = nodes.shape[0]
    receivers = jnp.repeat(jnp.arange(n), n - 1)
    neighbors = jnp.broadcast_to(jnp.arange(n - 1), (n, n - 1))
    senders = (neighbors + (neighbors >= jnp.arange(n)[:, None])).reshape(-1)

    receiver_states = jax.tree.map(lambda x: x[receivers], state)
    sender_states = jax.tree.map(lambda x: x[senders], state)
    edges = environment.edge_feature(receiver_states, sender_states)
    distance_squared = environment.squared_distance(receiver_states, sender_states)
    edge_mask = distance_squared < sensing_radius**2

    # Inactive edges become dummy self-loops, so segment_softmax never includes
    # them in a real receiver's normalization. Zero their unused features too.
    nodes = jnp.concatenate((nodes, jnp.zeros_like(nodes[:1])), axis=0)
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=jnp.where(edge_mask[:, None], edges, 0.0),
        senders=jnp.where(edge_mask, senders, n),
        receivers=jnp.where(edge_mask, receivers, n),
        n_node=jnp.array([n + 1], dtype=jnp.int32),
        n_edge=jnp.array([edges.shape[0]], dtype=jnp.int32),
        globals=None,
    )


class GCBF(eqx.Module):
    encoder: eqx.nn.MLP
    gate: eqx.nn.MLP
    value: eqx.nn.MLP
    head: eqx.nn.MLP

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        width: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        keys = jax.random.split(key, 4)

        self.encoder = eqx.nn.MLP(
            2 * node_dim + edge_dim,
            width,
            width,
            depth,
            activation=jax.nn.relu,
            key=keys[0],
        )
        self.gate = eqx.nn.MLP(
            width,
            1,
            width,
            depth,
            activation=jax.nn.relu,
            key=keys[1],
        )
        self.value = eqx.nn.MLP(
            width,
            width,
            width,
            depth,
            activation=jax.nn.relu,
            key=keys[2],
        )
        self.head = eqx.nn.MLP(
            width,
            1,
            width,
            depth,
            activation=jax.nn.relu,
            final_activation=jnp.tanh,
            key=keys[3],
        )

    def attention(
        self, graph: jraph.GraphsTuple
    ) -> tuple[
        Float[jax.Array, "nodes hidden_dim"],
        Float[jax.Array, "edges"],
    ]:
        n = graph.nodes.shape[0]
        receiver = graph.nodes[graph.receivers]
        sender = graph.nodes[graph.senders]
        z = jnp.concatenate((receiver, sender, graph.edges), axis=-1)
        q = jax.vmap(self.encoder)(z)
        logits = jax.vmap(self.gate)(q)[:, 0]

        weights = jraph.segment_softmax(
            logits,
            segment_ids=graph.receivers,
            num_segments=n,
        )

        values = jax.vmap(self.value)(q)
        aggregate = jraph.segment_sum(
            weights[:, None] * values,
            graph.receivers,
            num_segments=n,
        )

        return aggregate, weights

    def __call__(self, graph: jraph.GraphsTuple) -> Float[jax.Array, "nodes"]:
        """Return a barrier value for every node, including any dummy nodes."""
        aggregate, _ = self.attention(graph)
        return jax.vmap(self.head)(aggregate)[:, 0]
