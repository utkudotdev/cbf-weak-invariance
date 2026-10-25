"""Train an offline GCBF certificate from recorded multi-agent trajectories."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from gcbf import GCBF, LossConfig, cbf_loss, make_graph, trajectory_labels
from gcbf.checkpoint import save_checkpoint
from gcbf.data import load_dataset, safety_margins
from utils import fraction, nonnegative_float, positive_float, positive_int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gcbf"))
    parser.add_argument("--epochs", type=positive_int, default=50)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--horizon", type=positive_int, default=16)
    parser.add_argument("--width", type=positive_int, default=128)
    parser.add_argument("--depth", type=positive_int, default=2)
    parser.add_argument("--learning-rate", type=positive_float, default=3e-4)
    parser.add_argument("--sensing-radius", type=positive_float, default=float("inf"))
    parser.add_argument("--margin", type=positive_float, default=0.02)
    parser.add_argument("--alpha", type=positive_float, default=1.0)
    parser.add_argument("--derivative-weight", type=nonnegative_float, default=0.2)
    parser.add_argument("--validation-fraction", type=fraction, default=0.2)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    dataset, metadata, environment = load_dataset(args.dataset)
    r, t = jax.tree.leaves(dataset.states)[0].shape[:2]
    if args.horizon >= t:
        raise ValueError(f"horizon must be <= rollout length ({t - 1})")

    margins = safety_margins(environment, dataset.states, dataset.parameters)
    safe, unsafe = trajectory_labels(margins, args.horizon)
    print(
        f"Labels: safe={float(safe[:, :-1].mean()):.1%}, "
        f"unsafe={float(unsafe[:, :-1].mean()):.1%}"
    )
    if not bool(safe.any()) or not bool(unsafe[:, :-1].any()):
        print("Warning: dataset lacks a label class; collect more varied trajectories.")

    # Keep whole rollouts together to prevent train/validation leakage.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(r)
    validation_count = (
        max(1, round(r * args.validation_fraction))
        if r > 1 and args.validation_fraction
        else 0
    )
    validation_count = min(validation_count, r - 1)

    def transitions(indices):
        def flatten(x):
            return x.reshape((-1, *x.shape[2:]))

        current = jax.tree.map(lambda x: flatten(x[indices, :-1]), dataset.states)
        following = jax.tree.map(lambda x: flatten(x[indices, 1:]), dataset.states)
        return (
            current,
            following,
            flatten(safe[indices, :-1]),
            flatten(unsafe[indices, :-1]),
        )

    training = transitions(order[validation_count:])
    validation = transitions(order[:validation_count]) if validation_count else None

    example = jax.tree.map(lambda x: x[0], training[0])
    graph = make_graph(
        environment,
        example,
        args.sensing_radius,
    )
    model_config = dict(
        node_dim=graph.nodes.shape[-1],
        edge_dim=graph.edges.shape[-1],
        width=args.width,
        depth=args.depth,
    )
    model = GCBF(**model_config, key=jax.random.key(args.seed))

    optimizer = optax.chain(
        optax.clip_by_global_norm(2.0), optax.adam(args.learning_rate)
    )
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_config = LossConfig(args.margin, args.alpha, args.derivative_weight)

    def evaluate(model, states):
        graphs = jax.vmap(
            lambda state: make_graph(environment, state, args.sensing_radius)
        )(states)
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

    @eqx.filter_jit
    def objective(model, batch):
        state, following, safe_mask, unsafe_mask = batch
        h = evaluate(model, state)
        h_next = evaluate(model, following)
        return cbf_loss(h, h_next, safe_mask, unsafe_mask, metadata["dt"], loss_config)

    @eqx.filter_jit
    def step(model, optimizer_state, data, indices, metric_totals):
        # Fuse minibatch gathers with the update instead of dispatching a gather
        # for every state/label leaf from Python.
        batch = jax.tree.map(lambda x: x[indices], data)
        (_, metrics), grads = eqx.filter_value_and_grad(objective, has_aux=True)(
            model, batch
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state, model)
        if metric_totals is not None:
            metrics = jax.tree.map(
                lambda total, value: total + value, metric_totals, metrics
            )
        return eqx.apply_updates(model, updates), optimizer_state, metrics

    @eqx.filter_jit
    def validation_step(model, data, indices, loss_total):
        batch = jax.tree.map(lambda x: x[indices], data)
        return loss_total + objective(model, batch)[0]

    def batch_indices(count, shuffle=False):
        indices = rng.permutation(count) if shuffle else np.arange(count)
        for start in range(0, count, args.batch_size):
            yield indices[start : start + args.batch_size]

    training_pairs = jax.tree.leaves(training)[0].shape[0]
    validation_pairs = (
        jax.tree.leaves(validation)[0].shape[0] if validation is not None else 0
    )
    training_steps = (training_pairs + args.batch_size - 1) // args.batch_size
    validation_steps = (validation_pairs + args.batch_size - 1) // args.batch_size
    print(
        f"Training on {jax.devices()[0]}: {training_pairs:,} state pairs/epoch, "
        f"{training_steps:,} updates/epoch, batch size {args.batch_size:,}; "
        f"{validation_pairs:,} validation pairs"
    )

    config = dict(
        model=model_config,
        environment=metadata["environment"],
        parameters={k: float(v) for k, v in vars(dataset.parameters).items()},
        sensing_radius=args.sensing_radius
        if np.isfinite(args.sensing_radius)
        else None,
        dt=metadata["dt"],
        loss=loss_config._asdict(),
        horizon=args.horizon,
        dataset=str(args.dataset.resolve()),
        seed=args.seed,
        training_rollouts=order[validation_count:].tolist(),
        validation_rollouts=order[:validation_count].tolist(),
        training=dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            pairs_per_epoch=training_pairs,
            updates_per_epoch=training_steps,
            validation_pairs=validation_pairs,
        ),
        mode="offline_recorded_transitions",
    )
    history = []
    best = float("inf")

    for epoch in range(args.epochs):
        started = perf_counter()
        metric_totals = None
        for indices in batch_indices(training_pairs, shuffle=True):
            model, optimizer_state, metric_totals = step(
                model, optimizer_state, training, indices, metric_totals
            )

        # Fetch metrics once per epoch. float(device_array) inside the minibatch
        # loop forces a GPU synchronization and prevents asynchronous dispatch.
        totals = jax.device_get(metric_totals)
        train_seconds = perf_counter() - started
        row = dict(
            epoch=epoch + 1,
            **{key: float(value) / training_steps for key, value in totals.items()},
            train_seconds=train_seconds,
            pairs_per_second=training_pairs / train_seconds,
        )

        if validation is not None:
            validation_started = perf_counter()
            loss_total = jnp.array(0.0)
            for indices in batch_indices(validation_pairs):
                loss_total = validation_step(model, validation, indices, loss_total)
            row["validation"] = float(loss_total) / validation_steps
            row["validation_seconds"] = perf_counter() - validation_started

        score = row.get("validation", row["total"])
        if score < best:
            best = score
            save_checkpoint(args.output_dir, model, config)

        history.append(row)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={row['total']:.5f}"
            + (f" validation={row['validation']:.5f}" if validation is not None else "")
            + f" train={train_seconds:.1f}s ({row['pairs_per_second']:,.0f} pairs/s)"
        )

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best checkpoint: {args.output_dir.resolve()}")
    return args.output_dir


if __name__ == "__main__":
    main()
