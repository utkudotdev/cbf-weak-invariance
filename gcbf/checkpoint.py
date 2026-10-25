import json
from pathlib import Path

import equinox as eqx
import jax

from .gnn import GCBF


def save_checkpoint(directory, model, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(directory / "model.eqx", model)
    (directory / "config.json").write_text(json.dumps(config, indent=2))


def load_checkpoint(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    model = GCBF(**config["model"], key=jax.random.key(0))
    return eqx.tree_deserialise_leaves(directory / "model.eqx", model), config
