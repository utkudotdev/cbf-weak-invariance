from .types import Control, Parameters, State


def f(x: State, u: Control, params: Parameters, dt: float) -> State:
    return State(p=x.p + dt * x.v, v=x.v + dt * u / params.mass)
