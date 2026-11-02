"""Train a GCBF with fresh trajectories and general/unsafe replay."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from environments import load_environment
from gcbf import GCBF, LossConfig, make_graph
from gcbf.checkpoint import save_checkpoint
from training.batching import transitions
from training.collection import collect_rollouts
from training.optimization import make_steps
from training.qp import QPConfig
from training.replay import ReplaySampler
from training.validation import safety_metrics, sample_safety_scenarios
from utils import nonnegative_float, positive_float, positive_int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--environment", default="double_integrator")
    parser.add_argument(
        "--num-rollouts",
        type=positive_int,
        default=64,
        help="fresh training trajectories per epoch, excluding validation",
    )
    parser.add_argument("--rollouts-per-batch", type=positive_int, default=64)
    parser.add_argument("--rollout-length", type=positive_int, default=64)
    parser.add_argument("--dt", type=positive_float, default=0.05)
    parser.add_argument("--agents", type=positive_int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gcbf"))
    parser.add_argument("--epochs", type=positive_int, default=50)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument(
        "--buffer-size",
        type=positive_int,
        default=100_000,
        help="general replay capacity in graph states; unsafe capacity is half",
    )
    parser.add_argument("--horizon", type=positive_int, default=16)
    parser.add_argument("--width", type=positive_int, default=128)
    parser.add_argument("--depth", type=positive_int, default=2)
    parser.add_argument("--learning-rate", type=positive_float, default=3e-4)
    parser.add_argument("--sensing-radius", type=positive_float, default=float("inf"))
    parser.add_argument("--margin", type=positive_float, default=0.02)
    parser.add_argument("--alpha", type=positive_float, default=1.0)
    parser.add_argument("--derivative-weight", type=nonnegative_float, default=0.2)
    parser.add_argument("--safe-weight", type=nonnegative_float, default=1.0)
    parser.add_argument("--unsafe-weight", type=nonnegative_float, default=1.0)
    parser.add_argument(
        "--loss-validation-trajectories",
        type=positive_int,
        default=16,
        help="trajectories collected once for fixed validation-loss samples",
    )
    parser.add_argument(
        "--safety-validation-trajectories",
        type=positive_int,
        default=64,
        help="fixed scenarios rolled out under the current controller each epoch",
    )
    parser.add_argument("--slack-linear", type=positive_float, default=1000.0)
    parser.add_argument("--slack-quadratic", type=positive_float, default=10.0)
    parser.add_argument("--qp-tolerance", type=positive_float, default=1e-3)
    parser.add_argument("--qp-max-iterations", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args(argv)


def train(args):
    if args.horizon > args.rollout_length:
        raise ValueError("horizon must be <= rollout length")

    validation_count = args.loss_validation_trajectories
    training_count = args.num_rollouts

    environment = load_environment(args.environment)
    parameters = environment.default_parameters()
    shuffle_key = jax.random.key(args.seed)
    replay_key = jax.random.key(args.seed + 3)
    replay = ReplaySampler(args.buffer_size)

    training_key, validation_key = jax.random.split(jax.random.key(args.seed + 1))
    safety_key = jax.random.key(args.seed + 4)
    safety_initial_states, safety_nominal_controls = sample_safety_scenarios(
        environment,
        args.safety_validation_trajectories,
        args.agents,
        args.rollout_length,
        safety_key,
    )

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

        return transitions(environment, simulation, args.horizon)

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

    def batches(data, shuffle=False):
        nonlocal shuffle_key
        count = data[0].shape[0]
        if shuffle:
            shuffle_key, batch_key = jax.random.split(shuffle_key)
            indices = jax.random.permutation(batch_key, count)
        else:
            indices = jnp.arange(count)

        for start in range(0, count, args.batch_size):
            selected = indices[start : start + args.batch_size]
            yield tuple(x[selected] for x in data)

    fresh_pairs = training_count * args.rollout_length
    maximum_training_pairs = 3 * fresh_pairs
    validation_pairs = validation_count * args.rollout_length
    maximum_training_steps = (
        maximum_training_pairs + args.batch_size - 1
    ) // args.batch_size
    print(
        f"Training on {jax.devices()[0]}: {fresh_pairs:,} fresh state pairs/epoch, "
        f"up to {maximum_training_pairs:,} with replay, "
        f"batch size {args.batch_size:,}; "
        f"{validation_pairs:,} loss validation pairs; "
        f"{args.safety_validation_trajectories:,} safety validation trajectories"
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
            "loss_validation_trajectories": validation_count,
            "safety_validation_trajectories": args.safety_validation_trajectories,
            "rollouts_per_batch": args.rollouts_per_batch,
            "rollout_length": args.rollout_length,
            "agents": args.agents,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "fresh_pairs_per_epoch": fresh_pairs,
            "maximum_pairs_per_epoch": maximum_training_pairs,
            "maximum_updates_per_epoch": maximum_training_steps,
            "buffer_size": replay.general.capacity,
            "unsafe_buffer_size": replay.unsafe.capacity,
            "validation_pairs": validation_pairs,
        },
        "mode": "online_qp_rollouts_with_replay",
    }
    history = []

    def status_line(position):
        return tqdm(
            total=0,
            position=position,
            bar_format="{desc}",
            dynamic_ncols=True,
            disable=None,
            leave=False,
        )

    with (
        tqdm(
            total=args.epochs,
            unit="epoch",
            position=0,
            dynamic_ncols=True,
            disable=None,
        ) as progress,
        status_line(1) as loss_status,
        status_line(2) as data_status,
        status_line(3) as timing_status,
        status_line(4) as safety_status,
    ):
        # Hold out the initial validation rollouts for comparable checkpoint scores.
        validation = collect_dataset(model, validation_count, validation_key)
        validation_safe = float(validation[1].mean())
        validation_unsafe = float(validation[2].mean())

        for epoch in range(args.epochs):
            collection_started = perf_counter()
            fresh = collect_dataset(
                model, training_count, jax.random.fold_in(training_key, epoch)
            )
            training = replay.mix(fresh, jax.random.fold_in(replay_key, epoch))
            training_pairs = training[0].shape[0]
            training_steps = (training_pairs + args.batch_size - 1) // args.batch_size
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
                training_pairs=training_pairs,
                training_updates=training_steps,
                fresh_safe_fraction=float(fresh[1].mean()),
                fresh_unsafe_fraction=float(fresh[2].mean()),
                safe_fraction=float(training[1].mean()),
                unsafe_fraction=float(training[2].mean()),
                validation_safe_fraction=validation_safe,
                validation_unsafe_fraction=validation_unsafe,
                unsafe_graph_fraction=float(training[2].any(axis=-1).mean()),
                replay_size=len(replay.general),
                unsafe_replay_size=len(replay.unsafe),
                collection_seconds=collection_seconds,
                **{key: float(value) / training_pairs for key, value in totals.items()},
                train_seconds=train_seconds,
                pairs_per_second=training_pairs / train_seconds,
            )

            validation_started = perf_counter()
            loss_total = jnp.array(0.0)

            for batch in batches(validation):
                loss_total = validation_step(model, batch, loss_total)

            row["validation_loss"] = float(loss_total) / validation_pairs
            row["validation_seconds"] = perf_counter() - validation_started

            safety_started = perf_counter()
            safety_simulation = collect_rollouts(
                model,
                environment,
                parameters,
                num_rollouts=args.safety_validation_trajectories,
                rollouts_per_batch=args.rollouts_per_batch,
                rollout_length=args.rollout_length,
                dt=args.dt,
                sensing_radius=args.sensing_radius,
                qp_config=qp_config,
                agents=args.agents,
                key=safety_key,
                initial_states=safety_initial_states,
                nominal_controls=safety_nominal_controls,
            )
            rollout_metrics = safety_metrics(environment, safety_simulation, args.dt)
            row.update({key: float(value) for key, value in rollout_metrics.items()})
            row["safety_validation_seconds"] = perf_counter() - safety_started

            history.append(row)
            loss_status.set_description_str(
                f"loss={row['loss']:.5f} val={row['validation_loss']:.5f}"
                f" deriv={row['derivative_satisfied']:.0%}"
                f" qp={row['qp_converged']:.0%} slack={row['qp_slack']:.2g}"
            )
            data_status.set_description_str(
                f"safe/unsafe={row['fresh_safe_fraction']:.0%}/"
                f"{row['fresh_unsafe_fraction']:.0%}"
                f" mixed_unsafe={row['unsafe_fraction']:.0%}"
                f" replay={row['replay_size']:,}/{row['unsafe_replay_size']:,}"
                f" pairs={training_pairs:,}"
            )
            timing_status.set_description_str(
                f"collect={collection_seconds:.1f}s train={train_seconds:.1f}s"
                f" ({row['pairs_per_second']:,.0f} pairs/s)"
            )
            safety_status.set_description_str(
                f"safety violation mean={row['safety_violation_mean']:.3g}"
                f" worst={row['safety_violation_worst']:.3g}"
                f" violation_time={row['safety_violation_time_total']:.2f}s"
                f" unsafe={row['safety_unsafe_fraction']:.1%}"
                f" trajectories={row['safety_unsafe_trajectory_fraction']:.1%}"
            )
            progress.update(1)

    save_checkpoint(
        args.output_dir, model, config | {"checkpoint_epoch": args.epochs}
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Final checkpoint: {args.output_dir.resolve()}")
    return args.output_dir


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
