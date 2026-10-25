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

## Collect trajectories and train a GCBF

```sh
uv run python collect_dataset.py --num-rollouts 512 --rollouts-per-batch 64 \
  --rollout-length 64 --agents 5 --output-dir data/particles
uv run python train.py data/particles --epochs 50 --horizon 16 \
  --output-dir runs/particles
```

Collection saves states `(rollouts, steps+1, agents, ...)`, controls
`(rollouts, steps, agents, ...)`, parameters, and metadata.

`train.py` writes the best checkpoint (`model.eqx`, `config.json`) and per-epoch
metrics (`history.json`). Width, depth, loss parameters, learning rate, batch size,
and seed are configurable; run `--help` for options.

## Inspect a learned barrier

The particle viewer fixes agent 0's position and sweeps agent 1 over an x/y grid.
Four sliders vary the agents' absolute velocity components. Radio buttons select
whose barrier to display. Green shows the learned zero contour when it crosses
the slice; the dashed circle shows the physical collision boundary. Positive
values are the model's predicted safe region. Each image is a **two-agent slice**;
it does not summarize neighborhoods with additional agents.

```sh
uv run python -m gcbf.visualize runs/particles --ego-position 0 0 \
  --ego-velocity 0.5 0 --other-velocity -0.5 0
```
