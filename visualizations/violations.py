"""Plot safety violations over training, optionally comparing multiple runs."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


METRICS = {
    "mean": ("safety_violation_mean", "Mean violation", False),
    "time": ("safety_violation_time_total", "Total violation time", False),
    "unsafe": ("safety_unsafe_fraction", "Unsafe agent states", True),
    "trajectories": (
        "safety_unsafe_trajectory_fraction",
        "Trajectories with violations",
        True,
    ),
}


def read_history(path, metrics, x_axis):
    history_path = path / "history.json" if path.is_dir() else path
    history = json.loads(history_path.read_text())
    if not isinstance(history, list) or not history:
        raise ValueError(f"{history_path}: expected a nonempty list of epoch records")

    x = []
    values = {metric: [] for metric in metrics}
    updates = 0

    for index, row in enumerate(history, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{history_path}: record {index} must be an object")

        x_key = "training_updates" if x_axis == "steps" else "epoch"
        if x_key not in row:
            raise ValueError(f"{history_path}: record {index} is missing {x_key}")

        if x_axis == "steps":
            updates += row[x_key]
            x.append(updates)
        else:
            x.append(row[x_key])

        for metric in metrics:
            key = METRICS[metric][0]
            if key not in row:
                raise ValueError(
                    f"{history_path}: record {index} is missing {key}. "
                    "Use a run that logged safety validation metrics."
                )

            value = float(row[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{history_path}: {key} must be finite and nonnegative")

            values[metric].append(value)

    return x, values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "histories", nargs="+", type=Path, help="run directories or history.json files"
    )
    parser.add_argument("--labels", nargs="+", help="one legend label per run")
    parser.add_argument(
        "--metrics", nargs="+", choices=METRICS, default=["mean", "time"]
    )
    parser.add_argument(
        "--x-axis",
        choices=("steps", "epoch"),
        default="steps",
        help="cumulative optimizer updates (default) or epoch number",
    )
    parser.add_argument("--output", type=Path, help="save the plot, e.g. violations.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)

    if args.labels is not None and len(args.labels) != len(args.histories):
        parser.error("--labels must provide exactly one label per run")

    labels = args.labels or [
        path.name if path.is_dir() else path.parent.name for path in args.histories
    ]
    try:
        runs = [read_history(path, args.metrics, args.x_axis) for path in args.histories]
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    fig, axes = plt.subplots(
        len(args.metrics),
        1,
        figsize=(9, 3 * len(args.metrics)),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for ax, metric in zip(axes[:, 0], args.metrics):
        _, title, percentage = METRICS[metric]
        for label, (x, values) in zip(labels, runs):
            ax.plot(x, values[metric], label=label, linewidth=1.5, marker=".")

        ylabel = "Fraction" if percentage else "max(0, −h)"
        if metric == "time":
            ylabel = "Simulated seconds (sum across trajectories)"

        ax.set(title=title, ylabel=ylabel)
        ax.set_ylim(bottom=0)
        if percentage:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.set_ylim(top=1)

        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()

    axes[-1, 0].set_xlabel(
        "Training steps (cumulative optimizer updates)"
        if args.x_axis == "steps"
        else "Epoch"
    )
    fig.suptitle("Safety validation — lower is better")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)

    if not args.no_show:
        plt.show()

    return fig, axes[:, 0]


if __name__ == "__main__":
    main()
