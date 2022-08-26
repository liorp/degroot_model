from typing import List
from matplotlib import pyplot as plt
import numpy as np


def draw_energy(iterations: int, energies: List[int]):
    it = np.arange(iterations + 1)
    log_energies = np.log2(energies)
    plt.figure("Total Energy (log) vs Time")
    plt.scatter(it, log_energies)
    a, b, c = np.polyfit(it, log_energies, 2)
    plt.plot(
        it,
        a * it**2 + b * it + c,
        "r--",
        label=f"y={a:.2f}x^2+{b:.2f}x+{c:.2f}",
    )
    plt.legend(fontsize=9)
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.show()
