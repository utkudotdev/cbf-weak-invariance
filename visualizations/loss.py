"""Plot training and validation losses from a run directory or history.json."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="run directory or history.json")
    parser.add_argument("--output", type=Path, help="save the plot, e.g. loss.png")
    parser.add_argument("--log-scale", action="store_true", help="logarithmic loss axis")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)

    history_path = args.history
    if history_path.is_dir():
        history_path = history_path / "history.json"

    history = json.loads(history_path.read_text())
    if not isinstance(history, list) or not history:
        raise ValueError("history must contain a nonempty list of epoch records")

    epochs = [row["epoch"] for row in history]
    # Support histories written before the loss metrics were renamed.
    training = [row["loss"] if "loss" in row else row["total"] for row in history]
    validation = [
        row["validation_loss"] if "validation_loss" in row else row["validation"]
        for row in history
    ]

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(epochs, training, label="Training loss", linewidth=1.5)
    ax.plot(epochs, validation, label="Validation loss", linewidth=1.5)
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    if args.log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
    if not args.no_show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    main()
