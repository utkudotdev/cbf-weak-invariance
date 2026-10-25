"""Numeric, pickle-free trajectory storage, with environment PyTree templates."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from environments import load_environment


def save_dataset(directory, dataset, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for name in ("states", "controls", "parameters"):
        for i, leaf in enumerate(jax.tree.leaves(getattr(dataset, name))):
            arrays[f"{name}_{i}"] = np.asarray(leaf)

    np.savez(directory / "dset.npz", **arrays)
    (directory / "metadata.json").write_text(
        json.dumps(dict(metadata, format_version=2), indent=2)
    )


def load_dataset(directory):
    from collect_dataset import Dataset

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    environment = load_environment(metadata["environment"])

    if metadata.get("format_version") != 2:
        raise ValueError("unsupported dataset format; recollect using collect_dataset.py")

    templates = (
        environment.random_state(jax.random.key(0)),
        jnp.zeros((2,)),
        environment.default_parameters(),
    )
    trees = []
    with np.load(directory / "dset.npz", allow_pickle=False) as archive:
        for name, template in zip(("states", "controls", "parameters"), templates):
            leaves, structure = jax.tree.flatten(template)
            trees.append(
                jax.tree.unflatten(
                    structure,
                    [jnp.asarray(archive[f"{name}_{i}"]) for i in range(len(leaves))],
                )
            )

    dataset = Dataset(*trees)
    leaves = jax.tree.leaves(dataset.states)
    r, t, n = leaves[0].shape[:3]
    if t < 2 or any(x.shape[:3] != (r, t, n) for x in leaves):
        raise ValueError("state leaves must share (rollouts, steps+1, agents) axes")
    if any(x.shape[:3] != (r, t - 1, n) for x in jax.tree.leaves(dataset.controls)):
        raise ValueError("controls must align with state transitions")
    if metadata["dt"] <= 0:
        raise ValueError("dataset dt must be positive")

    return dataset, metadata, environment


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
