import importlib
from typing import Any, Protocol, cast


class Environment(Protocol):
    """Structural interface implemented by environment modules."""

    def default_parameters(self) -> Any: ...

    def random_state(self, key: Any) -> Any: ...

    def edge_feature(self, receiver: Any, sender: Any) -> Any: ...

    def node_feature(self, state: Any) -> Any: ...

    def squared_distance(self, receiver: Any, sender: Any) -> Any: ...

    def control_bounds(self) -> tuple[Any, Any]:
        """Return componentwise lower and upper control limits."""
        ...

    def f(self, state: Any, parameters: Any) -> Any:
        """Single-agent continuous-time drift, shaped (state_dim,)."""
        ...

    def g(self, state: Any, parameters: Any) -> Any:
        """Single-agent control matrix, shaped (state_dim, control_dim)."""
        ...

    def h(self, state: Any, other_states: Any, parameters: Any) -> Any:
        """Return the main agent's safety margin against the other agents.

        In GCBF+ terms, this function gives S_(N,i)."""
        ...

    def plot(
        self,
        states: Any,
        *,
        controls: Any,
        parameters: Any,
        dt: float,
    ) -> Any: ...


def dynamics(
    environment: Environment, state: Any, control: Any, parameters: Any
) -> Any:
    """Evaluate the single-agent vector field f(x) + g(x) u."""
    return (
        environment.f(state, parameters)
        + environment.g(state, parameters) @ control
    )


def load_environment(name: str) -> Environment:
    """Load and validate an environment module from this package."""
    if not name.isidentifier():
        raise ValueError(f"invalid environment name: {name!r}")

    module_name = f"{__name__}.{name}"
    try:
        environment = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise ValueError(f"unknown environment: {name!r}") from error
        raise

    required_hooks = (
        "default_parameters",
        "random_state",
        "edge_feature",
        "node_feature",
        "squared_distance",
        "f",
        "g",
        "control_bounds",
        "h",
        "plot",
    )
    missing = [
        hook
        for hook in required_hooks
        if not callable(getattr(environment, hook, None))
    ]
    if missing:
        raise TypeError(
            f"environment {name!r} is missing callable hooks: {', '.join(missing)}"
        )
    return cast(Environment, environment)


__all__ = ["Environment", "dynamics", "load_environment"]
