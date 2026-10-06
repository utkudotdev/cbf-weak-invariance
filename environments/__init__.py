import importlib
from pathlib import Path
from typing import Any, Protocol, cast


class Environment(Protocol):
    """Structural interface implemented by environment modules."""

    def default_parameters(self) -> Any: ...

    def random_state(self, key: Any) -> Any: ...

    def f(
        self, state: Any, control: Any, parameters: Any, dt: float
    ) -> Any: ...

    def h(self, state: Any, other_states: Any, parameters: Any) -> Any:
        """Return the main agent's safety margin against the other agents."""
        ...

    def plot(
        self,
        states: Any,
        *,
        controls: Any,
        parameters: Any,
        dt: float,
        show: bool,
        save_path: str | Path | None,
    ) -> Any: ...


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
        "f",
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


__all__ = ["Environment", "load_environment"]
