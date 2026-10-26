"""Online collection, minibatch optimization, and checkpointing."""

import json
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from environments import load_environment
from gcbf import GCBF, LossConfig, make_graph
from gcbf.checkpoint import save_checkpoint
from training.batching import transitions
from training.collection import collect_rollouts, print_statistics
from training.optimization import make_steps
from training.qp import QPConfig


def train(args):
    if args.horizon > args.rollout_length:
        raise ValueError("horizon must be <= rollout length")

    validation_count = round(args.num_rollouts * args.validation_fraction)
    training_count = args.num_rollouts - validation_count

    if validation_count < 1 or training_count < 1:
        raise ValueError(
            "num_rollouts and validation_fraction must yield at least one "
            "training rollout and one validation rollout"
        )

    environment = load_environment(args.environment)
    parameters = environment.default_parameters()
    rng = np.random.default_rng(args.seed)

    training_key, validation_key = jax.random.split(jax.random.key(args.seed + 1))

    qp_config = QPConfig(
        args.alpha,
        args.margin,
        args.slack_linear,
        args.slack_quadratic,
        args.qp_tolerance,
        args.qp_max_iterations,
    )

    def collect_dataset(model, num_rollouts, key):
        simulation = collect_rollouts(
            model,
            environment,
            parameters,
            num_rollouts=num_rollouts,
            rollouts_per_batch=args.rollouts_per_batch,
            rollout_length=args.rollout_length,
            dt=args.dt,
            sensing_radius=args.sensing_radius,
            qp_config=qp_config,
            agents=args.agents,
            key=key,
        )

        data = transitions(environment, simulation, args.horizon)

        print_statistics(simulation)
        print(
            f"Labels: safe={float(data[1].mean()):.1%}, "
            f"unsafe={float(data[2].mean()):.1%}"
        )

        return data

    example = jax.vmap(environment.random_state)(
        jax.random.split(jax.random.key(args.seed + 2), args.agents)
    )
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

    optimizer = optax.adam(args.learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_config = LossConfig(
        margin=args.margin,
        alpha=args.alpha,
        derivative_weight=args.derivative_weight,
        safe_weight=args.safe_weight,
        unsafe_weight=args.unsafe_weight,
    )

    step, validation_step = make_steps(
        environment,
        parameters,
        optimizer,
        loss_config,
        qp_config,
        args.dt,
        args.sensing_radius,
    )

    # Hold out the initial validation rollouts for comparable checkpoint scores.
    validation = collect_dataset(model, validation_count, validation_key)

    def batches(data, shuffle=False):
        count = data[0].shape[0]
        indices = rng.permutation(count) if shuffle else np.arange(count)

        for start in range(0, count, args.batch_size):
            selected = indices[start : start + args.batch_size]
            yield tuple(x[selected] for x in data)

    training_pairs = training_count * args.rollout_length
    validation_pairs = validation_count * args.rollout_length
    training_steps = (training_pairs + args.batch_size - 1) // args.batch_size
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
        "qp": qp_config._asdict(),
        "seed": args.seed,
        "training": {
            "epochs": args.epochs,
            "num_rollouts": args.num_rollouts,
            "training_rollouts": training_count,
            "validation_rollouts": validation_count,
            "rollouts_per_batch": args.rollouts_per_batch,
            "rollout_length": args.rollout_length,
            "agents": args.agents,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "pairs_per_epoch": training_pairs,
            "updates_per_epoch": training_steps,
            "validation_pairs": validation_pairs,
        },
        "mode": "online_qp_rollouts",
    }
    history = []
    best = float("inf")

    for epoch in range(args.epochs):
        collection_started = perf_counter()
        training = collect_dataset(
            model, training_count, jax.random.fold_in(training_key, epoch)
        )
        collection_seconds = perf_counter() - collection_started

        started = perf_counter()
        metric_totals = None
        for batch in batches(training, shuffle=True):
            model, optimizer_state, metric_totals = step(
                model, optimizer_state, batch, metric_totals
            )

        # Fetch metrics once per epoch. float(device_array) inside the minibatch
        # loop forces a GPU synchronization and prevents asynchronous dispatch.
        totals = jax.device_get(metric_totals)
        train_seconds = perf_counter() - started
        row = dict(
            epoch=epoch + 1,
            collection_seconds=collection_seconds,
            **{key: float(value) / training_pairs for key, value in totals.items()},
            train_seconds=train_seconds,
            pairs_per_second=training_pairs / train_seconds,
        )

        validation_started = perf_counter()
        loss_total = jnp.array(0.0)

        for batch in batches(validation):
            loss_total = validation_step(model, batch, loss_total)

        row["validation"] = float(loss_total) / validation_pairs
        row["validation_seconds"] = perf_counter() - validation_started

        if row["validation"] < best:
            best = row["validation"]
            save_checkpoint(args.output_dir, model, config)

        history.append(row)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={row['total']:.5f}"
            f" validation={row['validation']:.5f}"
            f" derivative_ok={row['derivative_satisfied']:.1%}"
            f" qp_ok={row['qp_converged']:.1%} slack={row['qp_slack']:.4g}"
            f" collect={collection_seconds:.1f}s"
            f" train={train_seconds:.1f}s ({row['pairs_per_second']:,.0f} pairs/s)"
        )

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best checkpoint: {args.output_dir.resolve()}")
    return args.output_dir
