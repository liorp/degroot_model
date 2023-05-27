from matplotlib import pyplot as plt
import numpy as np
import json
import pandas as pd


THRESHOLDS = (0.01, 1e6)


def draw_energy(
    energies: list[float],
    fitted_energies: list[float],
    label: str,
    color: str = "r",
) -> int:
    plt.figure("Total Energy vs Time (log log plot)")
    plt.title("Total Energy vs Time (log log plot)")
    plt.scatter(np.arange(len(energies)), energies, c=color, label="Data")

    plt.loglog(
        np.arange(len(fitted_energies)),
        fitted_energies,
        f"{color}--",
        label=label,
    )
    plt.legend(fontsize=9)
    plt.xlabel("Time")
    plt.ylabel("Energy")


def fit_energy_to_time(
    energies: list[int],
    p: float,
    N: int,
    stubborn: bool = False,
) -> dict:
    t, e = remove_extreme_values(
        np.array(np.arange(1, len(energies))), np.array(energies[1:])
    )
    log_times = np.log(t)
    log_energies = np.log(e)

    if not stubborn:
        # Linear regression E(t)=e^b*t^a => log(E(t))=a*log(t)+b
        a, b = np.polyfit(log_times, log_energies, 1)
        x, y = np.exp(log_times), np.exp(a * log_times + b)
        label = f"l_{p} E(t)=e^{b:.2f}*t^{a:.2f}"
    else:
        # Regression E(t)=E_infty+e^b*t^a => log(E(t)-E_infty)=a*log(t)+b
        # We assume that E_infty is the last value of the energy
        E_infty = e[-1] - 0.01
        # We only fit the part of the data between round(N / 2) : round(len(e) / 2)
        # because the energy is not stable at the beginning and at the end
        r = slice(int(N / 3), int(len(t) / 1.5))
        a, b = np.polyfit(log_times[r], np.log(e[r] - E_infty), 1)
        x, y = t, E_infty + np.exp(a * log_times + b)
        label = f"l_{p} E(t)={E_infty:.2f}+e^{b:.2f}*t^{a:.2f}"
    return {
        "a": a,
        "b": b,
        "x": x,
        "y": y,
        "label": label,
    }


def remove_extreme_values(x, y, thresholds=THRESHOLDS):
    mask = np.logical_and(y >= thresholds[0], y <= thresholds[1])

    x_filtered = x[mask]
    y_filtered = y[mask]
    return x_filtered, y_filtered


def export_dataframe(df, filename):
    """Export the DataFrame to a CSV file."""
    df.to_csv(filename, index=False)


def import_dataframe(filename):
    """Import the DataFrame from a CSV file."""
    df = pd.read_csv(filename)
    return df


def dump_p_vs_a(p_vs_a_not_stubborn, p_vs_a_stubborn):
    p_vs_a_not_stubborn.to_csv("data/p_vs_a_not_stubborn.csv", index=False)
    p_vs_a_stubborn.to_csv("data/p_vs_a_stubborn.csv", index=False)


def draw_p_vs_a(p_vs_a_not_stubborn=None, p_vs_a_stubborn=None):
    if p_vs_a_not_stubborn is None or p_vs_a_stubborn is None:
        p_vs_a_not_stubborn = pd.read_csv("data/p_vs_a_not_stubborn.csv")
        p_vs_a_stubborn = pd.read_csv("data/p_vs_a_stubborn.csv")
    plt.figure("p vs a")
    plt.title("p vs a")

    plt.scatter(p_vs_a_not_stubborn["p"], p_vs_a_not_stubborn["a"], c="b", label="Data")
    plt.scatter(p_vs_a_stubborn["p"], p_vs_a_stubborn["a"], c="r", label="Data")

    plt.plot(
        p_vs_a_not_stubborn["p"],
        p_vs_a_not_stubborn["a"],
        "b--",
        label="Not Stubborn",
    )
    plt.plot(p_vs_a_stubborn["p"], p_vs_a_stubborn["a"], "r--", label="Stubborn")
    plt.legend()
    plt.xlabel("p")
    plt.ylabel("a")
    plt.savefig("images/p_vs_a.png", bbox_inches="tight", pad_inches=0)
    plt.show()
