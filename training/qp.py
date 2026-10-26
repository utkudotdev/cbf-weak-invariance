from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import qpax
from jax.scipy.linalg import block_diag
from jaxtyping import Float

from environments import Environment
from gcbf import GCBF, make_graph


class QPConfig(NamedTuple):
    alpha: float
    margin: float
    slack_linear: float
    slack_quadratic: float
    tolerance: float
    max_iterations: int


class QPData(eqx.Module):
    hessian: Float[jax.Array, "variables variables"]
    linear: Float[jax.Array, "variables"]  # noqa: F821
    constraint: Float[jax.Array, "agents variables"]
    rhs: Float[jax.Array, "agents"]  # noqa: F821
    lower: Float[jax.Array, "variables"]  # noqa: F821
    upper: Float[jax.Array, "variables"]  # noqa: F821


class QPResult(eqx.Module):
    control: Any
    slack: Float[jax.Array, "agents"]  # noqa: F821
    violation: Float[jax.Array, ""]
    converged: jax.Array


def control_template(environment: Environment, agents: int):
    """Broadcast single-agent bounds to the joint control arrays."""
    lower, upper = environment.control_bounds()

    return (
        jnp.broadcast_to(lower, (agents, *lower.shape)),
        jnp.broadcast_to(upper, (agents, *upper.shape)),
    )


def lie_derivative(h, f):
    """Return x -> Dh(x) f(x), for a vector field or a matrix of fields."""

    def derivative(x):
        return jnp.tensordot(jax.jacrev(h)(x), f(x), axes=x.ndim)

    return derivative


def make_qp(
    model: GCBF,
    environment: Environment,
    state: Any,
    parameters: Any,
    sensing_radius: float,
    config: QPConfig,
):
    """Build min .5||u||² + .5*w₂||r||² + w₁ sum(r), r >= 0.

    Constraints: Lf h + Lg h u + alpha*h + r >= margin, plus hard control bounds.
    Lie derivatives use the continuous-time control-affine vector field f + g u.
    All agents' controls are solved jointly, including cross-agent derivatives.
    """
    agents = state.shape[0]
    lower, upper = control_template(environment, agents)
    control_shape = lower.shape

    lower, upper = lower.ravel(), upper.ravel()
    zero = jnp.zeros_like(lower)

    def barrier(x):
        return model(make_graph(environment, x, sensing_radius))[:-1]

    def joint_f(x):
        return jax.vmap(environment.f, in_axes=(0, None))(x, parameters)

    def joint_g(x):
        matrices = jax.vmap(environment.g, in_axes=(0, None))(x, parameters)

        return block_diag(*matrices).reshape(*x.shape, -1)

    h = barrier(state)
    lie_f_h = lie_derivative(barrier, joint_f)(state)
    lie_g_h = lie_derivative(barrier, joint_g)(state)

    num_controls = lower.size
    hessian = jnp.diag(
        jnp.concatenate(
            (jnp.ones(num_controls), jnp.full(agents, config.slack_quadratic))
        )
    )
    linear = jnp.concatenate((zero, jnp.full(agents, config.slack_linear)))

    constraint = -jnp.concatenate((lie_g_h, jnp.eye(agents)), axis=1)

    rhs = lie_f_h + config.alpha * h - config.margin

    data = QPData(
        hessian,
        linear,
        constraint,
        rhs,
        jnp.concatenate((lower, jnp.zeros(agents))),
        jnp.concatenate((upper, jnp.full(agents, jnp.inf))),
    )

    return data, lambda u: u.reshape(control_shape)


class QPSolution(eqx.Module):
    x: Float[jax.Array, "variables"]  # noqa: F821
    converged: jax.Array
    primal_residual: Float[jax.Array, ""]
    dual_residual: Float[jax.Array, ""]


def inequalities(data: QPData):
    """Convert box bounds to Gx <= h without any infinite right-hand sides."""
    variables = data.linear.size
    controls = variables - data.rhs.size
    upper = data.upper[:controls]

    identity = jnp.eye(variables)
    upper_rows = jnp.where(jnp.isfinite(upper)[:, None], identity[:controls], 0.0)
    lower_rows = jnp.where(jnp.isfinite(data.lower)[:, None], -identity, 0.0)

    matrix = jnp.concatenate((data.constraint, upper_rows, lower_rows))
    rhs = jnp.concatenate(
        (
            data.rhs,
            jnp.where(jnp.isfinite(upper), upper, 1.0),
            jnp.where(jnp.isfinite(data.lower), -data.lower, 1.0),
        )
    )

    return matrix, rhs


def solve_qp(data: QPData, config: QPConfig) -> QPSolution:
    matrix, rhs = inequalities(data)
    variables = data.linear.size

    x, _, dual, _, converged, _ = qpax.solve_qp(
        data.hessian,
        data.linear,
        jnp.zeros((0, variables)),
        jnp.zeros(0),
        matrix,
        rhs,
        backend="i",
        solver_tol=config.tolerance,
        max_iter=config.max_iterations,
    )

    primal = jnp.maximum(jnp.max(matrix @ x - rhs), 0.0)
    dual_residual = jnp.max(jnp.abs(data.hessian @ x + data.linear + matrix.T @ dual))

    return QPSolution(x, converged, primal, dual_residual)


def qp_action(
    model: GCBF,
    environment: Environment,
    state: Any,
    parameters: Any,
    sensing_radius: float,
    config: QPConfig,
) -> QPResult:
    data, recover_control = make_qp(
        model, environment, state, parameters, sensing_radius, config
    )
    solution = solve_qp(data, config)

    agents = data.rhs.size
    finite = jnp.all(jnp.isfinite(solution.x))

    # Enforce the hard bounds even when the solver terminates within tolerance.
    # A failed solve falls back to bounded zero control and is reported in metrics.
    x = jnp.where(finite, solution.x, jnp.zeros_like(solution.x))
    x = jnp.clip(x, data.lower, data.upper)

    violation = jnp.maximum(jnp.max(data.constraint @ x - data.rhs), 0.0)
    converged = finite & solution.converged & (violation <= config.tolerance)

    return QPResult(recover_control(x[:-agents]), x[-agents:], violation, converged)
