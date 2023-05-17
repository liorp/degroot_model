from typing import List
from matplotlib import pyplot as plt
import numpy as np


def draw_energy(iterations: int, energies: List[int], p: str) -> plt.Figure:
    it = np.arange(iterations + 1)
    log_energies = np.log2(energies)
    log_times = np.log2(it)
    f = plt.figure("Total l_{p} Energy (log) vs Time (log)")
    plt.title("Total l_{p} Energy (log) vs Time (log)")
    plt.scatter(log_times, log_energies)
    a, b = np.polyfit(log_times, log_energies, 1)
    plt.plot(
        log_times,
        a * it + b,
        "r--",
        label=f"y={a:.2f}x+{b:.2f}",
    )
    plt.legend(fontsize=9)
    plt.xlabel("Time (log)")
    plt.ylabel(f"l_{p} Energy (log)")
    return f
