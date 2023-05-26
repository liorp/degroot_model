from typing import List
from matplotlib import pyplot as plt
import numpy as np
import json


THRESHOLDS = (0.01, 1e6)


def draw_energy(
    energies: List[int],
    p: str,
    color: str = "r",
    stubborn: bool = False,
) -> int:
    """
    Returns the fitted exponent `a` of the regression function `E(t)=e^b*t^a` or `E(t)=E_infty+e^b*t^a` depending on stubborn flag (see below).
    Note that it removes extreme values in order for the fit to work.
    """
    t, e = remove_extreme_values(
        np.array(np.arange(1, len(energies))), np.array(energies[1:])
    )
    log_times = np.log(t)
    log_energies = np.log(e)
    plt.scatter(t, e, c=color, label="Data")

    if not stubborn:
        # Linear regression E(t)=e^b*t^a => log(E(t))=a*log(t)+b
        a, b = np.polyfit(log_times, log_energies, 1)
        x, y = np.exp(log_times), np.exp(a * log_times + b)
        label = f"l_{p} E(t)=e^{b:.2f}*t^{a:.2f}"
    else:
        # Regression E(t)=E_infty+e^b*t^a => log(E(t)-E_infty)=a*log(t)+b
        # We assume that E_infty is the last value of the energy
        E_infty = e[-1] - 0.01
        a, b = np.polyfit(log_times, np.log(e - E_infty), 1)
        x, y = t, E_infty + np.exp(a * log_times + b)
        label = f"l_{p} E(t)={E_infty:.2f}+e^{b:.2f}*t^{a:.2f}"

    plt.loglog(
        x,
        y,
        f"{color}--",
        label=label,
    )
    plt.legend(fontsize=9)
    plt.xlabel("Time")
    plt.ylabel("Energy")
    return a


def remove_extreme_values(x, y, thresholds=THRESHOLDS):
    mask = np.logical_and(y >= thresholds[0], y <= thresholds[1])

    x_filtered = x[mask]
    y_filtered = y[mask]
    return x_filtered, y_filtered


def dump_p_vs_a(p_vs_a_not_stubborn, p_vs_a_stubborn):
    """
    This functions dumps p vs a for both stubborn and not stubborn cases in json format
    """
    with open("data/p_vs_a_not_stubborn.json", "w") as f:
        json.dump(p_vs_a_not_stubborn, f)
    with open("data/p_vs_a_stubborn.json", "w") as f:
        json.dump(p_vs_a_stubborn, f)


def draw_p_vs_a():
    """
    This functions draws p vs a for both stubborn and not stubborn cases
    """
    with open("data/p_vs_a_not_stubborn.json", "r") as f:
        p_vs_a_not_stubborn = json.load(f)
    with open("data/p_vs_a_stubborn.json", "r") as f:
        p_vs_a_stubborn = json.load(f)
    plt.figure("p vs a")
    plt.title("p vs a")
    plt.plot(
        [x[0] for x in p_vs_a_not_stubborn],
        [x[1] for x in p_vs_a_not_stubborn],
        label="Not Stubborn",
    )
    plt.plot(
        [x[0] for x in p_vs_a_stubborn],
        [x[1] for x in p_vs_a_stubborn],
        label="Stubborn",
    )
    plt.legend()
    plt.savefig("images/p_vs_a.png", bbox_inches="tight", pad_inches=0)
    plt.show()
