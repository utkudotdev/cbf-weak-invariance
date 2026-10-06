# Control Barrier Functions for Weakly Invariant Systems

This project explores using weak invariance in dynamical systems to accelerate training for multi-agent CBFs.

## Running the particle example

Run a random-input simulation and display its trajectories with:

```sh
uv run python main.py
```

The runner exposes the simulation settings as command-line options. For
example, this saves a deterministic run without opening a window:

```sh
uv run python main.py --steps 500 --dt 0.02 --seed 7 \
  --output particle.png --no-show
```

Use `uv run python main.py --help` for the full list of options.

The simulation loop uses `jax.lax.scan` and is JIT-compiled. The environment
and step count are static arguments because they select Python code or
determine array shapes; `dt` and `seed` can change without recompiling for the
same shapes.

New environments can be selected with `--environment NAME`. A module at
`environments/NAME/__init__.py` needs to expose five hooks:
`default_parameters()`, `initial_state(key)`,
`random_control(key, state)`, `f(state, control, parameters, dt)`, and
`plot(states, *, controls, dt, show, save_path)`.
