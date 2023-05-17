from typing import List
from matplotlib import pyplot as plt
import numpy as np


def draw_energy(
    iterations: int, energies: List[int], p: str, color: str = "r"
) -> plt.Figure:
    it = np.arange(iterations + 1)
    log_energies = np.log(energies[1:])
    log_times = np.log(it[1:])
    plt.scatter(log_times, log_energies, c=color)
    a, b = np.polyfit(log_times, log_energies, 1)
    plt.plot(
        log_times,
        a * log_times + b,
        f"{color}--",
        label=f"l_{p} y={a:.2f}x+{b:.2f}",
    )
    plt.legend(fontsize=9)
    plt.xlabel("Time (log)")
    plt.ylabel("Energy (log)")
