"""GCBF+ Eq. (21), evaluated on recorded transitions (no learned policy)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LossConfig(NamedTuple):
    margin: float
    alpha: float
    derivative_weight: float


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

    total = safe_loss + unsafe_loss + config.derivative_weight * derivative_loss

    return total, dict(
        total=total,
        safe=safe_loss,
        unsafe=unsafe_loss,
        derivative=derivative_loss,
        derivative_satisfied=jnp.mean(residual >= 0),
    )


def trajectory_labels(margins, horizon: int):
    """Label (rollout, time, agent) arrays using fully observed future windows.

    Safe iff t..t+horizon are collision-free. Unsafe iff colliding *now*.
    Future collisions and incomplete tails are unlabeled, not unsafe.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")

    t = margins.shape[1]
    safe = margins >= 0

    # Pad with false so incomplete windows are never labeled safe.
    padded = jnp.pad(safe, ((0, 0), (0, horizon), (0, 0)))
    for k in range(1, horizon + 1):
        safe = safe & padded[:, k : k + t]
    return safe, margins < 0
