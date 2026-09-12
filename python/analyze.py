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

    # Helper function to filter lines before numpy sees them
    def filter_valid_lines(filepath: Path):
        with open(filepath, 'r') as f:
            for line in f:
                # Split the line into words/tokens
                parts = line.split()
                if not parts:  # Skip empty lines
                    continue
                
                # Check if the first word can be converted to a float (number)
                try:
                    float(parts[0])
                    yield line  # If successful, pass this line to numpy
                except ValueError:
                    pass  # If it fails (e.g., text), skip this line

    # Pass the filtered lines directly into loadtxt instead of the file path
    data = np.loadtxt(filter_valid_lines(path))
    
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
    
def plot_speed(t: np.ndarray, y: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, y, "o-", color="tab:blue", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Залежність швидкості від часу")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")
    plt.show()
    

def plot_accel(t: np.ndarray, y: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, y, "o-", color="tab:blue", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accel (m/s²)")
    ax.set_title("Залежність прискорення від часу")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")
    plt.show()
    

def calculate_speed(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    speed = []
    for idx, el in enumerate(t):
        if idx == 0:
            speed.append((y[idx+1]-y[idx])/(t[idx+1]-t[idx]))
        elif idx == t.size-1:
            speed.append((y[idx]-y[idx-1])/(t[idx]-t[idx-1]))
        else:
            speed.append((y[idx+1]-y[idx-1])/(t[idx+1]-t[idx-1]))
    return np.array(speed)


def calculate_accel(t: np.ndarray, speed: np.ndarray) -> np.ndarray:
    accel = []
    for idx, el in enumerate(t):
        if idx == 0:
            accel.append((speed[idx+1]-speed[idx])/(t[idx+1]-t[idx]))
        elif idx == t.size-1:
            accel.append((speed[idx]-speed[idx-1])/(t[idx]-t[idx-1]))
        else:
            accel.append((speed[idx+1]-speed[idx-1])/(t[idx+1]-t[idx-1]))
    return np.array(accel)
        


def main() -> None:
    t, y = load_data(DATA_PATH)
    print(f"Loaded {len(t)} samples from {DATA_PATH}")
    print(np.column_stack((t, y)))

    plot_displacement(t, y, OUTPUT_DIR / "displacement_plot.png")
    speed = calculate_speed(t, y)
    accel = calculate_accel(t, speed)
    plot_speed(t, speed, OUTPUT_DIR / "speed_plot.png")
    plot_accel(t, accel, OUTPUT_DIR / "accel_plot.png")
    
    #using np.gradient
    
    speed_gradient = np.gradient(y, t, edge_order=2)
    accel_gradient = np.gradient(speed_gradient, t, edge_order=2)
    plot_speed(t, speed_gradient, OUTPUT_DIR / "speed_gradient_plot.png")
    plot_accel(t, accel_gradient, OUTPUT_DIR / "accel_gradient_plot.png")
    print(speed)
    print(speed_gradient)



if __name__ == "__main__":
    main()
