from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


THRESHOLDS = (0.001, 1e6)


def draw_energy(
    energies: list[float],
    fitted_energies: list[float],
    label: str,
) -> int:
    plt.figure("Total Energy vs Time (log log plot)")
    plt.title("Total Energy vs Time (log log plot)")

    plt.loglog(
        np.arange(1, len(fitted_energies) + 1),
        fitted_energies,
        "b--",
        label=label,
    )
    plt.scatter(np.arange(1, len(energies) + 1), energies, c="r", label="Data")
    plt.legend(fontsize=9)
    plt.xlabel("Time")
    plt.ylabel("Energy")


def fit_energy_to_time(
    energies: list[int],
    p: float,
    stubborn: bool = False,
) -> dict:
    t, e = remove_extreme_values(
        np.array(np.arange(1, len(energies))), np.array(energies[1:])
    )
    log_times = np.log(t)
    log_energies = np.log(e)
    x = np.arange(1, len(energies))

    if not stubborn:
        # Linear regression E(t)=e^b*t^a => log(E(t))=a*log(t)+b
        a, b = np.polyfit(log_times, log_energies, 1)
        y = np.exp(b) * np.float_power(x, a)
        label = f"l_{p} E(t)=e^{b:.2f}*t^{a:.2f}"
    else:
        # Regression E(t)=E_infty+e^b*t^a => log(E(t)-E_infty)=a*log(t)+b
        # In order to obtain initial guess for the parameters, we fit the data with polyfit
        # We assume that E_infty is the last value of the energy
        E_infty = e[-1]

        # We only fit the part of the data
        # because the energy is not stable at the beginning and at the end
        r = slice(int(len(t) / 50), int(len(t) / 1.25))
        a, b = np.polyfit(log_times[r], np.log(e[r] - E_infty), 1)

        # We use the initial guess to fit the data using nonlinear least squares with `curve_fit`
        popt, pcov = curve_fit(stubborn_energy, x, e, [a, b, E_infty])
        a, b, E_infty = popt

        y = E_infty + np.exp(b) * np.float_power(x, a)
        label = f"l_{p} E(t)={E_infty:.2f}+e^{b:.2f}*t^{a:.2f}"
    # Add first energy value
    # x, y = np.concatenate((np.array([0]), np.array(x))), np.concatenate(
    #     (np.array([energies[0]]), np.array(y))
    # )
    return {
        "a": a,
        "b": b,
        "x": x,
        "y": y,
        "label": label,
    }


def stubborn_energy(x, a, b, E_infty):
    return E_infty + np.exp(b) * np.float_power(x, a)


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
