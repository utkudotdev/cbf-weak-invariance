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

## Train a GCBF

```sh
uv run --extra gpu python train.py --num-rollouts 64 --rollout-length 64 --agents 5 \
  --epochs 50 --horizon 16 --output-dir runs/particles
```

Omit `--extra gpu` for CPU training.

Training collects a fresh, small batch of trajectories each epoch in memory.
Each rollout uses the current barrier's QP controller. Whole rollouts are split
before labeling/flattening; the initial validation rollouts are held fixed across
epochs. Training rollouts are refreshed using the latest barrier.
There is no separate collection command or dataset file. The `training/` package
handles collection, batching, optimization, and checkpointing.

`train.py` writes the best checkpoint (`model.eqx`, `config.json`) and per-epoch
metrics (`history.json`). Run `--help` for all options.

For each minibatch, the current barrier solves a centralized QP for all agents,
then the environment computes `x_next = f(x, u_qp, parameters, dt)`. The derivative
loss evaluates the learned barrier at both `x` and this recomputed `x_next`.
Gradients reach both barrier evaluations; QP actions are treated as constants.
There is no learned policy yet, and recorded next states are not used in the loss.

The QP minimizes `0.5 ||u||² + 0.5 w₂ ||r||² + w₁ sum(r)` subject to
`Lf h + Lg h u + alpha h + r >= margin`, nonnegative per-agent slack `r`, and
hard environment control bounds. Zero control is the nominal action (no goal
controller yet). The Lie derivatives use the full multi-agent barrier Jacobian
and the Euler vector field `(f(x,u,dt) - x) / dt`; this assumes control-affine
Euler dynamics, as in the particle environment. The loss still measures the
actual finite-step residual, so it need not equal the continuous-time QP bound.
Slack makes an infeasible barrier constraint solvable; it is not a safety guarantee.
`qp_converged`, `qp_slack`, and `qp_violation` are logged with the loss. Nonfinite
solves fall back to bounded zero control and count as failures.

The solver is [qpax](https://github.com/qpax-solver/qpax) with its implicit (`i`)
backend, selected from GPU benchmarks. Tolerance, iteration limit, and slack
penalties are CLI options. The network and loss configuration remain explicit.
The QP can only affect trajectories when `environment.control_bounds()` allows
a nonzero range of actions.

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
