"""Online collection, minibatch optimization, and checkpointing."""

import json
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from environments import load_environment
from gcbf import GCBF, LossConfig, cbf_loss, make_graph
from gcbf.checkpoint import save_checkpoint
from training.batching import evaluate, transitions
from training.collection import collect_dataset


def train(args):
    if args.horizon > args.rollout_length:
        raise ValueError("horizon must be <= rollout length")
    environment = load_environment(args.environment)
    parameters = environment.default_parameters()
    rng = np.random.default_rng(args.seed)
    validation_count = (
        max(1, round(args.num_rollouts * args.validation_fraction))
        if args.num_rollouts > 1 and args.validation_fraction
        else 0
    )
    validation_count = min(validation_count, args.num_rollouts - 1)

    def collect_epoch(epoch):
        simulation = collect_dataset(
            environment,
            parameters,
            num_rollouts=args.num_rollouts,
            rollouts_per_batch=args.rollouts_per_batch,
            rollout_length=args.rollout_length,
            dt=args.dt,
            agents=args.agents,
            key=jax.random.fold_in(jax.random.key(args.seed + 1), epoch),
        )

        # Split whole rollouts before flattening, so labels cannot leak across splits.
        def subset(start, stop):
            states = jax.tree.map(lambda x: x[start:stop], simulation.states)
            controls = jax.tree.map(lambda x: x[start:stop], simulation.controls)
            return transitions(
                environment,
                type(simulation)(states, controls, parameters),
                args.horizon,
            )

        training = subset(validation_count, args.num_rollouts)
        validation = subset(0, validation_count) if validation_count else None
        velocities = simulation.states.v
        speeds = jnp.linalg.norm(velocities, axis=-1)
        axes = tuple(range(velocities.ndim - 1))
        stats = jax.device_get(
            (
                velocities.min(axis=axes),
                velocities.mean(axis=axes),
                velocities.max(axis=axes),
                speeds.min(),
                speeds.mean(),
                speeds.max(),
            )
        )
        print(f"Velocity components: min={stats[0]}, mean={stats[1]}, max={stats[2]}")
        print(
            f"Speed ||v||: min={stats[3]:.6g}, mean={stats[4]:.6g}, max={stats[5]:.6g}"
        )
        return training, validation

    collection_started = perf_counter()
    training, validation = collect_epoch(0)
    initial_collection_seconds = perf_counter() - collection_started
    example = jax.tree.map(lambda x: x[0], training[0])
    graph = make_graph(
        environment,
        example,
        args.sensing_radius,
    )
    model_config = {
        "node_dim": graph.nodes.shape[-1],
        "edge_dim": graph.edges.shape[-1],
        "width": args.width,
        "depth": args.depth,
    }
    model = GCBF(**model_config, key=jax.random.key(args.seed))

    optimizer = optax.chain(
        optax.clip_by_global_norm(2.0), optax.adam(args.learning_rate)
    )
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_config = LossConfig(args.margin, args.alpha, args.derivative_weight)

    @eqx.filter_jit
    def objective(model, batch):
        state, following, safe_mask, unsafe_mask = batch
        h = evaluate(model, environment, state, args.sensing_radius)
        h_next = evaluate(model, environment, following, args.sensing_radius)
        return cbf_loss(h, h_next, safe_mask, unsafe_mask, args.dt, loss_config)

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

    config = {
        "model": model_config,
        "environment": args.environment,
        "parameters": {k: float(v) for k, v in vars(parameters).items()},
        "sensing_radius": args.sensing_radius
        if np.isfinite(args.sensing_radius)
        else None,
        "dt": args.dt,
        "loss": loss_config._asdict(),
        "horizon": args.horizon,
        "seed": args.seed,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "pairs_per_epoch": training_pairs,
            "updates_per_epoch": training_steps,
            "validation_pairs": validation_pairs,
        },
        "mode": "online_random_rollouts",
    }
    history = []
    best = float("inf")

    for epoch in range(args.epochs):
        collection_started = perf_counter()
        if epoch:
            training, _ = collect_epoch(epoch)
        collection_seconds = (
            perf_counter() - collection_started if epoch else initial_collection_seconds
        )
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
            collection_seconds=collection_seconds,
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
