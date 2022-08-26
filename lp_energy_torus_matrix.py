from typing import Tuple
import numpy as np
from scipy import optimize
from functools import reduce
from itertools import starmap
from matplotlib import pyplot as plt
import networkx as nx
from SyncDegrootModel import SyncDegrootModel
import seaborn as sns
from celluloid import Camera
import multiprocessing as mp
from logger import logger
from utils import draw_energy


ITERATIONS = 50
P = 10
N = 7
PROCESSES = 1
STUBBORN_AGENTS = {(2, 5): 0.7, (1, 1): 0}


def _get_neighbours(
    mat: np.ndarray, idx: Tuple[int, int]
) -> Tuple[int, int, int, int, int, int, int, int]:
    columns = mat.shape[1]
    rows = mat.shape[0]
    neighbours = [
        mat[idx[0], (idx[1] + 1) % columns],  # right
        mat[(idx[0] + 1) % rows, (idx[1] + 1) % columns],  # up right
        mat[(idx[0] + 1) % rows, idx[1]],  # up
        mat[(idx[0] + 1) % rows, (idx[1] - 1) % columns],  # up left
        mat[idx[0], (idx[1] - 1) % columns],  # left
        mat[(idx[0] - 1) % rows, (idx[1] - 1) % columns],  # down left
        mat[(idx[0] - 1) % rows, idx[1]],  # down
        mat[(idx[0] - 1) % rows, (idx[1] + 1) % columns],  # down right
    ]
    return neighbours


def _calculate_next_energy(
    mat: np.ndarray, idx: Tuple[int, int], p: int = P
) -> Tuple[Tuple[int, int], int]:
    if idx in STUBBORN_AGENTS:
        return idx, STUBBORN_AGENTS[idx]
    neighbours = _get_neighbours(mat, idx)
    energy_function = lambda x: np.sum(
        [np.float_power(np.abs(x - A_i), p) for A_i in neighbours]
    )
    result = optimize.minimize_scalar(energy_function)
    x_min = result.x
    return idx, x_min
    # For 2-degroot model:
    # return idx, sum(neighbours) / len(neighbours)


def degroot_iteration(mat: np.ndarray, p: int = P) -> np.ndarray:
    temp = mat.copy()
    if PROCESSES == 1:
        result = starmap(
            _calculate_next_energy,
            [(mat, idx, p) for idx, _ in np.ndenumerate(temp)],
        )
    else:
        with mp.Pool(processes=PROCESSES) as pool:
            result = pool.starmap(
                _calculate_next_energy,
                [(mat, idx, p) for idx, _ in np.ndenumerate(temp)],
            )
    for idx, x_min in result:
        temp[idx] = x_min
    return temp


def _get_current_energy(
    mat: np.ndarray, idx: Tuple[int, int], elem: int, p: int = P
) -> int:
    neighbours = _get_neighbours(mat, idx)
    energy_function = lambda x: np.sum(
        [np.float_power(np.abs(x - A_i), p) for A_i in neighbours]
    )
    result = energy_function(elem)
    return result


def get_matrix_energy(mat: np.ndarray, p: int = P) -> int:
    if PROCESSES == 1:
        result = starmap(
            _get_current_energy,
            [(mat, idx, e, p) for idx, e in np.ndenumerate(mat)],
        )
    else:
        with mp.Pool(processes=PROCESSES) as pool:
            result = pool.starmap(
                _get_current_energy,
                [(mat, idx, e, p) for idx, e in np.ndenumerate(mat)],
            )
    return sum(result)


def main():
    matrix = 10 * np.random.rand(N, N)
    fig, (ax, cbar_ax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [10, 1]})
    camera = Camera(fig)
    ax.text(
        0.5, 1.01, f"t=1 n={N} p={P} stubborn={STUBBORN_AGENTS}", transform=ax.transAxes
    )
    sns.heatmap(
        matrix, cmap="magma", annot=False, vmin=0, vmax=10, ax=ax, cbar_ax=cbar_ax
    )
    camera.snap()

    energies = []
    energy = get_matrix_energy(matrix)
    energies.append(energy)
    logger.info(f"Initial Energy: f{energy}")

    n = degroot_iteration(matrix)
    ax.text(
        0.5, 1.01, f"t=2 n={N} p={P} stubborn={STUBBORN_AGENTS}", transform=ax.transAxes
    )
    plt.draw()
    sns.heatmap(n, cmap="magma", annot=False, vmin=0, vmax=10, ax=ax, cbar_ax=cbar_ax)
    camera.snap()

    for i in range(ITERATIONS):
        n = degroot_iteration(n)
        ax.text(
            0.5,
            1.01,
            f"t={3+i} n={N} p={P} stubborn={STUBBORN_AGENTS}",
            transform=ax.transAxes,
        )
        plt.draw()
        sns.heatmap(
            n, cmap="magma", annot=False, vmin=0, vmax=10, ax=ax, cbar_ax=cbar_ax
        )
        camera.snap()

        energy = get_matrix_energy(n)
        energies.append(energy)
        logger.debug(f"{i+1} Energy: f{energy}")

    ax.text(
        0.5,
        1.01,
        f"t={4+i} n={N} p={P} stubborn={STUBBORN_AGENTS}",
        transform=ax.transAxes,
    )
    plt.draw()
    sns.heatmap(n, cmap="magma", annot=False, vmin=0, vmax=10, ax=ax, cbar_ax=cbar_ax)
    camera.snap()
    # animation = camera.animate()
    # animation.save(f"animation_N{N}_P{P}_I{ITERATIONS}_STUBBORN{len(STUBBORN_AGENTS)}.mp4")
    plt.show()

    draw_energy(ITERATIONS, energies)


if __name__ == "__main__":
    main()
