"""Fixed-capacity graph batches and trajectory labels."""

import jax
import jax.numpy as jnp

from gcbf import make_graph, trajectory_labels


def evaluate(model, environment, states, sensing_radius):
    graphs = jax.vmap(lambda state: make_graph(environment, state, sensing_radius))(
        states
    )
    batch_size, nodes_per_graph, node_dim = graphs.nodes.shape
    offsets = jnp.arange(batch_size)[:, None] * nodes_per_graph

    # Vectorized equivalent of jraph.batch for equal-size graphs. One
    # disjoint graph gives the GPU larger matrix/reduction operations.
    graph = graphs._replace(
        nodes=graphs.nodes.reshape((-1, node_dim)),
        edges=graphs.edges.reshape((-1, graphs.edges.shape[-1])),
        senders=(graphs.senders + offsets).reshape(-1),
        receivers=(graphs.receivers + offsets).reshape(-1),
        n_node=graphs.n_node.reshape(-1),
        n_edge=graphs.n_edge.reshape(-1),
    )
    # Each constituent graph ends with its own dummy node.
    return model(graph).reshape((batch_size, nodes_per_graph))[:, :-1]


def safety_margins(environment, states, parameters):
    """Evaluate environment safety at training time, excluding each agent itself."""

    def graph_margins(state):
        n = jax.tree.leaves(state)[0].shape[0]

        def one(i):
            indices = jnp.arange(n - 1)
            indices += indices >= i
            return environment.h(
                jax.tree.map(lambda x: x[i], state),
                jax.tree.map(lambda x: x[indices], state),
                parameters,
            )

        return jax.vmap(one)(jnp.arange(n))

    return jax.jit(jax.vmap(jax.vmap(graph_margins)))(states)


def transitions(environment, simulation, horizon):
    margins = safety_margins(environment, simulation.states, simulation.parameters)
    safe, unsafe = trajectory_labels(margins, horizon)

    def flatten(x):
        return x.reshape((-1, *x.shape[2:]))

    return (
        jax.tree.map(lambda x: flatten(x[:, :-1]), simulation.states),
        flatten(safe[:, :-1]),
        flatten(unsafe[:, :-1]),
    )
