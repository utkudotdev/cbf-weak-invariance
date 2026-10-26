"""QP-controlled next states and differentiable barrier updates."""

import equinox as eqx
import jax

from gcbf import cbf_loss
from simulation import euler_step
from training.batching import evaluate
from training.qp import qp_action


def make_steps(environment, parameters, optimizer, loss_config, qp_config, dt, radius):
    def controlled_batch(model, batch):
        states, safe, unsafe = batch

        results = jax.vmap(
            lambda state: qp_action(
                model, environment, state, parameters, radius, qp_config
            )
        )(states)

        # The QP chooses the action using the current certificate, but is not
        # differentiated. Both h(x) and h(x_next) below retain model gradients.
        controls = jax.lax.stop_gradient(results.control)
        following = jax.vmap(
            jax.vmap(lambda x, u: euler_step(environment, x, u, parameters, dt))
        )(states, controls)

        metrics = {
            "qp_converged": results.converged.mean(),
            "qp_slack": results.slack.mean(),
            "qp_violation": results.violation.mean(),
        }

        return (states, following, safe, unsafe), metrics

    def objective(model, batch):
        states, following, safe, unsafe = batch
        h = evaluate(model, environment, states, radius)
        h_next = evaluate(model, environment, following, radius)
        return cbf_loss(h, h_next, safe, unsafe, dt, loss_config)

    @eqx.filter_jit
    def step(model, optimizer_state, batch, metric_totals):
        batch_size = batch[0].shape[0]
        batch, qp_metrics = controlled_batch(model, batch)

        (_, metrics), grads = eqx.filter_value_and_grad(objective, has_aux=True)(
            model, batch
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state, model)

        metrics.update(qp_metrics)
        # Weight partial minibatches by their actual number of graph states.
        metrics = jax.tree.map(lambda x: x * batch_size, metrics)

        if metric_totals is not None:
            metrics = jax.tree.map(lambda a, b: a + b, metric_totals, metrics)

        return eqx.apply_updates(model, updates), optimizer_state, metrics

    @eqx.filter_jit
    def validation_step(model, batch, loss_total):
        batch_size = batch[0].shape[0]
        batch, _ = controlled_batch(model, batch)

        return loss_total + objective(model, batch)[0] * batch_size

    return step, validation_step
