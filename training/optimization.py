"""QP-controlled next states and differentiable barrier updates."""

import equinox as eqx
import jax

from gcbf.loss import objective


def make_steps(environment, parameters, optimizer, loss_config, qp_config, dt, radius):
    def batch_objective(model, batch):
        return objective(
            model, batch, environment, parameters, loss_config, qp_config, dt, radius
        )

    @eqx.filter_jit
    def step(model, optimizer_state, batch, metric_totals):
        batch_size = batch[0].shape[0]
        (_, metrics), grads = eqx.filter_value_and_grad(batch_objective, has_aux=True)(
            model, batch
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state, model)

        # Weight partial minibatches by their actual number of graph states.
        metrics = jax.tree.map(lambda x: x * batch_size, metrics)

        if metric_totals is not None:
            metrics = jax.tree.map(lambda a, b: a + b, metric_totals, metrics)

        return eqx.apply_updates(model, updates), optimizer_state, metrics

    @eqx.filter_jit
    def validation_step(model, batch, loss_total):
        batch_size = batch[0].shape[0]
        return loss_total + batch_objective(model, batch)[0] * batch_size

    return step, validation_step
