"""GCBF+ Eq. (21), evaluated on controlled transitions."""

from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp


class LossConfig(NamedTuple):
    margin: float
    alpha: float
    derivative_weight: float
    safe_weight: float = 1.0
    unsafe_weight: float = 1.0


def masked_mean(values, mask):
    return jnp.sum(jnp.where(mask, values, 0.0)) / jnp.maximum(jnp.sum(mask), 1)


def cbf_loss(h, h_next, safe, unsafe, dt, config: LossConfig):
    """Safe/unsafe hinge means plus derivative hinge over *all* samples."""

    # Eq. (21) in the paper uses sums. Following the authors' implementation,
    # we instead average each labeled class separately and the derivative term
    # over all samples. This changes their relative weighting as class counts
    # vary.
    safe_loss = masked_mean(jax.nn.relu(config.margin - h), safe)
    unsafe_loss = masked_mean(jax.nn.relu(config.margin + h), unsafe)

    residual = (h_next - h) / dt + config.alpha * h
    derivative_loss = jnp.mean(jax.nn.relu(config.margin - residual))

    total = (
        config.safe_weight * safe_loss
        + config.unsafe_weight * unsafe_loss
        + config.derivative_weight * derivative_loss
    )

    return total, dict(
        loss=total,
        safe_loss=safe_loss,
        unsafe_loss=unsafe_loss,
        derivative_loss=derivative_loss,
        derivative_satisfied=jnp.mean(residual >= 0),
    )


def objective(model, batch, environment, parameters, loss_config, qp_config, dt, radius):
    """Evaluate the barrier loss and QP metrics on controlled transitions."""
    # Training helpers import the public gcbf API, so load them at call time.
    from simulation import euler_step
    from training.batching import evaluate
    from training.qp import qp_action

    states, safe, unsafe = batch
    control_model = jax.tree.map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
        model,
    )
    results = jax.vmap(
        lambda state: qp_action(
            control_model, environment, state, parameters, radius, qp_config
        )
    )(states)

    # Choose controls with a frozen certificate; retain gradients through both
    # barrier evaluations without differentiating through the QP solver.
    controls = jax.lax.stop_gradient(results.control)
    following = jax.vmap(
        jax.vmap(lambda x, u: euler_step(environment, x, u, parameters, dt))
    )(states, controls)

    h = evaluate(model, environment, states, radius)
    h_next = evaluate(model, environment, following, radius)
    loss, metrics = cbf_loss(h, h_next, safe, unsafe, dt, loss_config)
    metrics.update(
        qp_converged=results.converged.mean(),
        qp_slack=results.slack.mean(),
        qp_violation=results.violation.mean(),
    )

    return loss, metrics


@partial(jax.jit, static_argnames=("horizon",))
def trajectory_labels(margins, horizon: int):
    """Label (rollout, time, agent) arrays using fully observed future windows.

    Safe iff t..t+horizon are collision-free. Unsafe iff colliding *now*.
    Future collisions and incomplete tails are unlabeled, not unsafe.
    """
    # Pad with false so incomplete windows are never labeled safe.
    padded = jnp.pad(margins >= 0, ((0, 0), (0, horizon), (0, 0)))

    safe = jax.lax.reduce_window(
        padded,
        True,
        jax.lax.bitwise_and,
        window_dimensions=(1, horizon + 1, 1),
        window_strides=(1, 1, 1),
        padding="VALID",
    )

    return safe, margins < 0
