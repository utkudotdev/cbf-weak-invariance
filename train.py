"""Train a GCBF with fresh trajectories collected each epoch."""

import argparse
from pathlib import Path

from training.loop import train
from utils import fraction, nonnegative_float, positive_float, positive_int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--environment", default="particle")
    parser.add_argument("--num-rollouts", type=positive_int, default=64)
    parser.add_argument("--rollouts-per-batch", type=positive_int, default=64)
    parser.add_argument("--rollout-length", type=positive_int, default=64)
    parser.add_argument("--dt", type=positive_float, default=0.05)
    parser.add_argument("--agents", type=positive_int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gcbf"))
    parser.add_argument("--epochs", type=positive_int, default=50)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--horizon", type=positive_int, default=16)
    parser.add_argument("--width", type=positive_int, default=128)
    parser.add_argument("--depth", type=positive_int, default=2)
    parser.add_argument("--learning-rate", type=positive_float, default=3e-4)
    parser.add_argument("--sensing-radius", type=positive_float, default=float("inf"))
    parser.add_argument("--margin", type=positive_float, default=0.02)
    parser.add_argument("--alpha", type=positive_float, default=1.0)
    parser.add_argument("--derivative-weight", type=nonnegative_float, default=0.2)
    parser.add_argument("--safe-weight", type=nonnegative_float, default=1.0)
    parser.add_argument("--unsafe-weight", type=nonnegative_float, default=1.0)
    parser.add_argument("--validation-fraction", type=fraction, default=0.2)
    parser.add_argument("--slack-linear", type=positive_float, default=1000.0)
    parser.add_argument("--slack-quadratic", type=positive_float, default=10.0)
    parser.add_argument("--qp-tolerance", type=positive_float, default=1e-3)
    parser.add_argument("--qp-max-iterations", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args(argv)


def main(argv=None):
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
