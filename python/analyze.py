"""
Practical 1: load UART-logged displacement data and plot y(t).

Expected data format in ../data/data.txt (two columns, space-separated):
    t[s]  y[m]

Example:
    0.0000 0.0000
    0.1000 0.0109
    ...
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.txt"
OUTPUT_DIR = Path(__file__).resolve().parent


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Save PuTTY session output to data/data.txt first."
        )

    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected two columns (t y) in {path}, got shape {data.shape}"
        )

    t = data[:, 0]
    y = data[:, 1]
    return t, y


def plot_displacement(t: np.ndarray, y: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, y, "o-", color="tab:blue", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (m)")
    ax.set_title("Залежність переміщення від часу")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")
    plt.show()


def main() -> None:
    t, y = load_data(DATA_PATH)
    print(f"Loaded {len(t)} samples from {DATA_PATH}")
    print(np.column_stack((t, y)))

    plot_displacement(t, y, OUTPUT_DIR / "displacement_plot.png")

    # TODO (завдання):
    # 1. Обчисліть швидкість і прискорення чисельними похідними.
    # 2. Побудуйте графіки v(t) та a(t), збережіть як .png у цій папці.


if __name__ == "__main__":
    main()
