# Control Barrier Functions for Weakly Invariant Systems

This project explores using weak invariance in dynamical systems to accelerate
training for multi-agent CBFs.

## Running the particle example

Run a random-input simulation and display its trajectories with:

```sh
uv run python main.py
```

The runner exposes the simulation settings as command-line options. For
example, this saves a deterministic run without opening a window:

```sh
uv run python main.py --steps 200 --dt 0.05 --agents 5 --seed 7 \
  --output particle.gif --no-show
```

Use `uv run python main.py --help` for the full list of options.
