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


P = 15
N = 100
ITERATIONS = np.floor(N / 2).astype(int)
PROCESSES = 1
STUBBORN_AGENTS = {
    (np.floor(N) - 1, np.floor(N) - 1): 0,
    (np.floor(N / 2), np.floor(N / 2)): 10,
}
STEPS_SNAPSHOT = 3
CHECK_MINIMAL_ENERGY = False


def energy_function_flat(x: np.array, p: int = P) -> int:
    # Calculate energy of flattened matrix
    mat = x.reshape(N, N)
    return get_matrix_energy(mat, p)


def _get_neighbours(
    mat: np.ndarray, idx: Tuple[int, int]
) -> Tuple[int, int, int, int, int, int, int, int]:
    # Get opinions of neighbours of idx
    columns = mat.shape[1]
    rows = mat.shape[0]
    right = (idx[0], (idx[1] + 1) % columns)
    left = (idx[0], (idx[1] - 1) % columns)
    up = ((idx[0] + 1) % rows, idx[1])
    down = ((idx[0] - 1) % rows, idx[1])
    up_right = ((idx[0] + 1) % rows, (idx[1] + 1) % columns)
    up_left = ((idx[0] + 1) % rows, (idx[1] - 1) % columns)
    down_right = ((idx[0] - 1) % rows, (idx[1] + 1) % columns)
    down_left = ((idx[0] - 1) % rows, (idx[1] - 1) % columns)
    directions = [right, left, up, down, up_right, up_left, down_right, down_left]

    neighbours = [STUBBORN_AGENTS.get(d, mat[d]) for d in directions]
    return neighbours


def _calculate_next_energy(
    mat: np.ndarray, idx: Tuple[int, int], p: int = P
) -> Tuple[Tuple[int, int], int]:
    if idx in STUBBORN_AGENTS:
        return idx, STUBBORN_AGENTS[idx]
    neighbours = _get_neighbours(mat, idx)
    energy_function = lambda x: np.sum(
        np.float_power(np.abs(np.subtract(x, neighbours)), p)
    )
    result = optimize.minimize_scalar(energy_function)
    x_min = result.x
    return idx, x_min
    # For 2-degroot model:
    # return idx, sum(neighbours) / len(neighbours)


def degroot_iteration(mat: np.ndarray, p: int = P) -> np.ndarray:
    # Calculate next energy for each element in matrix
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


def _get_vertex_energy(
    mat: np.ndarray, idx: Tuple[int, int], elem: int, p: int = P
) -> int:
    neighbours = _get_neighbours(mat, idx)
    energy_function = lambda x: np.sum(
        np.float_power(np.abs(np.subtract(x, neighbours)), p)
    )
    result = energy_function(elem)
    return result


def get_matrix_energy(mat: np.ndarray, p: int = P) -> int:
    if PROCESSES == 1:
        result = starmap(
            _get_vertex_energy,
            [(mat, idx, e, p) for idx, e in np.ndenumerate(mat)],
        )
    else:
        with mp.Pool(processes=PROCESSES) as pool:
            result = pool.starmap(
                _get_vertex_energy,
                [(mat, idx, e, p) for idx, e in np.ndenumerate(mat)],
            )
    return sum(result)


def main():
    logger.info("Starting simulation")

    matrix = 10 * np.random.rand(N, N)
    logger.info("Generated matrix")

    if CHECK_MINIMAL_ENERGY:
        # Draw minimal energy solution for this graph
        minimum_graph_energy = optimize.minimize(
            energy_function_flat,
            [10 * np.random.rand() for i in range(N * N)],
            constraints=[
                optimize.LinearConstraint(
                    [k == i * N + j for k in range(N * N)],
                    STUBBORN_AGENTS[(i, j)],
                    STUBBORN_AGENTS[(i, j)],
                )
                for (i, j) in STUBBORN_AGENTS
            ],
            method="trust-constr",
        )
        logger.info(f"Minimum Graph Energy: {minimum_graph_energy.fun}")
        plt.figure("Minimum Graph State")
        plt.title(f"Minimum Graph State (E={minimum_graph_energy.fun})")
        sns.heatmap(
            minimum_graph_energy.x.reshape(N, N),
            cmap="magma",
            annot=True,
            vmin=0,
            vmax=10,
        )

    fig, (ax, cbar_ax) = plt.subplots(
        ncols=2, gridspec_kw={"width_ratios": [10, 1]}, num="Process"
    )
    camera = Camera(fig)
    ax.text(
        0.5, 1.01, f"t=0 n={N} p={P} stubborn={STUBBORN_AGENTS}", transform=ax.transAxes
    )
    sns.heatmap(
        matrix,
        cmap="magma",
        annot=False,
        vmin=0,
        vmax=10,
        ax=ax,
        cbar_ax=cbar_ax,
    )
    camera.snap()

    energies = []
    energy = get_matrix_energy(matrix)
    energies.append(energy)
    logger.info(f"Initial Energy: f{energy}")

    for i in range(ITERATIONS):
        matrix = degroot_iteration(matrix)
        if i % STEPS_SNAPSHOT == 0:
            ax.text(
                0.5,
                1.01,
                f"t={i+1} n={N} p={P} stubborn={STUBBORN_AGENTS}",
                transform=ax.transAxes,
            )
            plt.draw()
            sns.heatmap(
                matrix,
                cmap="magma",
                annot=False,
                vmin=0,
                vmax=10,
                ax=ax,
                cbar_ax=cbar_ax,
            )
            camera.snap()

        energy = get_matrix_energy(matrix)
        energies.append(energy)
        logger.debug(f"{i+1} Energy: f{energy}")

    animation = camera.animate()
    animation.save(
        f"pdegroot_torus_N{N}_P{P}_I{ITERATIONS}_STUBBORN{len(STUBBORN_AGENTS)}.mp4"
    )
    plt.show()

    if CHECK_MINIMAL_ENERGY:
        plt.figure("Final State")
        plt.title(f"Final State (E={energy})")
        sns.heatmap(
            matrix,
            cmap="magma",
            annot=True,
            vmin=0,
            vmax=10,
        )

        f = draw_energy(ITERATIONS, energies)
        plt.show()


if __name__ == "__main__":
    main()
