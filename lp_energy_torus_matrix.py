import itertools
import random
import multiprocessing as mp
from typing import List, Tuple

import numpy as np
from scipy import optimize
from itertools import starmap
from matplotlib import pyplot as plt
import seaborn as sns
from celluloid import Camera
from logger import logger
from utils import draw_energy
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# Parameters

P_OPTIONS = [1.1, 1.5, 2, 5, 7, 10, 50, np.inf]
N = 50
ITERATIONS = 100
PROCESSES = 1
STUBBORN_AGENTS_OPTIONS = [
    {},
    {
        (np.floor(N) - 10, np.floor(N) - 10): 0,
        (np.floor(N / 2 - 4), np.floor(N / 2 - 4)): 10,
    },
]
STEPS_SNAPSHOT = 2
CHECK_MINIMAL_ENERGY = False
ANIMATION = False
TIMES_FOR_PLOT = [1, 5, 10, 50, 100]

Neighbour = tuple[tuple[int, int], tuple[int, int]]
SmallWorldNeighbours = list[Neighbour]


def _generate_small_world_neighbours(N: int = N, k: int = 5) -> SmallWorldNeighbours:
    # Get neighbours for small world network - we only allow one out of every k nodes to have a single random friend
    values = list(range(0, N - 1))

    # Generate and shuffle all possible combinations of length 2
    shuffled_combinations = zip(
        random.sample(list(itertools.combinations(values, 2)), N // k),
        random.sample(list(itertools.combinations(values, 2)), N // k),
    )

    # Convert shuffled combinations into list
    neighbours = [tuple(comb) for comb in shuffled_combinations]
    return neighbours


SMALL_WORLD_NEIGHBOURS_OPTIONS = [[], _generate_small_world_neighbours()]

# Functions


def _get_neighbours(
    mat: np.ndarray,
    idx: Tuple[int, int],
    stubborn_agents: dict,
    small_world_neighbours: SmallWorldNeighbours,
) -> tuple[int, int, int, int, int, int, int, int]:
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

    if small_world_neighbours:
        # Add small world neighbours to directions list if idx is in them (i.e. idx is a small world node)
        directions.extend(
            [n[1] if n[0] == idx else n[0] for n in small_world_neighbours if idx in n]
        )

    neighbours = tuple(stubborn_agents.get(d, mat[d]) for d in directions)
    return neighbours


def _calculate_next_energy(
    mat: np.ndarray,
    idx: Tuple[int, int],
    p: int,
    stubborn_agents: dict,
    small_world_neighbours: SmallWorldNeighbours,
) -> Tuple[Tuple[int, int], int]:
    if idx in stubborn_agents:
        return idx, stubborn_agents[idx]
    neighbours = _get_neighbours(mat, idx, stubborn_agents, small_world_neighbours)

    if p == np.inf:
        energy_function = lambda x: np.max(np.abs(np.subtract(x, neighbours)))
    else:
        energy_function = lambda x: np.sum(
            np.float_power(np.abs(np.subtract(x, neighbours)), p)
        )

    result = optimize.minimize_scalar(energy_function)
    x_min = result.x
    return idx, x_min
    # For 2-degroot model:
    # return idx, sum(neighbours) / len(neighbours)


def degroot_iteration(
    mat: np.ndarray,
    p: int,
    stubborn_agents: dict,
    small_world_neighbours: SmallWorldNeighbours,
) -> np.ndarray:
    # Calculate next energy for each element in matrix
    temp = mat.copy()
    if PROCESSES == 1:
        result = starmap(
            _calculate_next_energy,
            [
                (mat, idx, p, stubborn_agents, small_world_neighbours)
                for idx, _ in np.ndenumerate(temp)
            ],
        )
    else:
        with mp.Pool(processes=PROCESSES) as pool:
            result = pool.starmap(
                _calculate_next_energy,
                [
                    (mat, idx, p, stubborn_agents, small_world_neighbours)
                    for idx, _ in np.ndenumerate(temp)
                ],
            )
    for idx, x_min in result:
        temp[idx] = x_min
    return temp


def _get_vertex_energy(
    mat: np.ndarray,
    idx: Tuple[int, int],
    elem: int,
    p: int,
    stubborn_agents: dict,
    small_world_neighbours: SmallWorldNeighbours,
) -> int:
    neighbours = _get_neighbours(mat, idx, stubborn_agents, small_world_neighbours)

    if p == np.inf:
        energy_function = lambda x: np.max(np.abs(np.subtract(x, neighbours)))
    else:
        energy_function = lambda x: np.sum(
            np.float_power(np.abs(np.subtract(x, neighbours)), p)
        )

    result = energy_function(elem)
    return result


def get_matrix_energy(
    mat: np.ndarray,
    p: int,
    stubborn_agents: dict,
    small_world_neighbours: SmallWorldNeighbours,
) -> int:
    if PROCESSES == 1:
        result = starmap(
            _get_vertex_energy,
            [
                (mat, idx, e, p, stubborn_agents, small_world_neighbours)
                for idx, e in np.ndenumerate(mat)
            ],
        )
    else:
        with mp.Pool(processes=PROCESSES) as pool:
            result = pool.starmap(
                _get_vertex_energy,
                [
                    (mat, idx, e, p, stubborn_agents, small_world_neighbours)
                    for idx, e in np.ndenumerate(mat)
                ],
            )
    if p == np.inf:
        return max(result)
    else:
        return sum(result)


def get_matrix_energies(
    mat: np.ndarray,
    p: int,
    energies: dict[str, int],
    stubborn_agents: dict,
    small_world_neighbours: SmallWorldNeighbours,
) -> Tuple[int, int, int]:
    # Returns p energy, 2 energy, inf energy
    # It also appends the energies to the energies dict for convenience
    results = tuple(
        get_matrix_energy(mat, _p, stubborn_agents, small_world_neighbours)
        for _p in [p, 2, np.inf]
    )
    energies["p"].append(results[0])
    energies["2"].append(results[1])
    energies["inf"].append(results[2])
    return results


def degroot_simulation(
    matrix: np.ndarray,
    p: int,
    stubborn_agents: dict,
    small_world_neighbours: SmallWorldNeighbours,
) -> dict:
    logger.info(
        f"Starting simulation for p={p}, stubborn {len(stubborn_agents)}, small world {len(small_world_neighbours) > 0}"
    )
    fig, axes = plt.subplots(nrows=1, ncols=len(TIMES_FOR_PLOT) + 1, figsize=(25, 3))
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(
        f"Simulating p-DeGroot for n={N} p={p} stubborn={stubborn_agents} small world {len(small_world_neighbours) > 0} in times {TIMES_FOR_PLOT}"
    )
    im1 = sns.heatmap(
        matrix,
        cmap="magma",
        annot=False,
        vmin=0,
        vmax=10,
        ax=axes[0],
        cbar=False,
    )
    divider1 = make_axes_locatable(axes[0])
    cax1 = divider1.append_axes("left", size="10%", pad=0.3)
    cax1.yaxis.tick_left()
    cax1.yaxis.set_ticks_position("left")
    cax1.yaxis.set_label_position("left")
    plt.colorbar(im1.get_children()[0], cax=cax1)

    if ANIMATION:
        # GIF animation
        fig, (ax, cbar_ax) = plt.subplots(
            ncols=2, gridspec_kw={"width_ratios": [10, 1]}, num="Process"
        )
        camera = Camera(fig)
        ax.text(
            0,
            1.01,
            f"t=0 n={N} p={p} stubborn={stubborn_agents}",
            transform=ax.transAxes,
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

    energies = {"p": [], "2": [], "inf": []}
    energy_p, energy_2, energy_inf = get_matrix_energies(
        matrix, p, energies, stubborn_agents, small_world_neighbours
    )
    logger.info(
        f"Initial Energies: f{energy_p:.2f}, f{energy_2:.2f}, f{energy_inf:.2f}"
    )

    temp_matrix = matrix.copy()
    for i in range(ITERATIONS):
        temp_matrix = degroot_iteration(
            temp_matrix, p, stubborn_agents, small_world_neighbours
        )
        if i % STEPS_SNAPSHOT == 0 and ANIMATION:
            # GIF animation
            ax.text(
                0,
                1.01,
                f"t={i+1} n={N} p={p} stubborn={stubborn_agents}",
                transform=ax.transAxes,
            )
            plt.draw()
            sns.heatmap(
                temp_matrix,
                cmap="magma",
                annot=False,
                vmin=0,
                vmax=10,
                ax=ax,
                cbar_ax=cbar_ax,
            )
            camera.snap()
        if i + 1 in TIMES_FOR_PLOT:
            sns.heatmap(
                temp_matrix,
                cmap="magma",
                annot=False,
                vmin=0,
                vmax=10,
                ax=axes[TIMES_FOR_PLOT.index(i + 1) + 1],
                cbar=False,
            )

        energy_p, energy_2, energy_inf = get_matrix_energies(
            temp_matrix, p, energies, stubborn_agents, small_world_neighbours
        )
        logger.debug(
            f"{i+1} Energies: f{energy_p:.2f}, f{energy_2:.2f}, f{energy_inf:.2f}"
        )

    if ANIMATION:
        animation = camera.animate()
        animation.save(
            f"animations/pdegroot_torus_N{N}_p{p}_I{ITERATIONS}_STUBBORN{len(stubborn_agents)}_SMALL_WORLD_{len(small_world_neighbours) > 0}".replace(
                ".", "_"
            )
            + ".mp4"
        )
    plt.savefig(
        f"images/pdegroot_torus_N{N}_p{p}_I{ITERATIONS}_STUBBORN{len(stubborn_agents)}_SMALL_WORLD_{len(small_world_neighbours) > 0}".replace(
            ".", "_"
        )
        + ".png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()

    draw_energies(energies, p, stubborn_agents, small_world_neighbours)


def draw_energies(
    energies: dict[str, list],
    p: int,
    stubborn_agents: dict,
    small_world_neighbours: SmallWorldNeighbours,
):
    plt.figure("Total Energy (log) vs Time (log)")
    plt.title("Total Energy (log) vs Time (log)")
    draw_energy(ITERATIONS, energies["p"], p, "r")
    draw_energy(ITERATIONS, energies["2"], "2", "g")
    draw_energy(ITERATIONS, energies["inf"], "inf", "b")
    plt.savefig(
        f"images/energy_pdegroot_torus_N{N}_p_{p}_I{ITERATIONS}_STUBBORN{len(stubborn_agents)}_SMALL_WORLD_{len(small_world_neighbours) > 0}".replace(
            ".", "_"
        )
        + ".png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()


def energy_function_flat(
    x: np.array,
    p: int,
    stubborn_agents: dict,
    small_world_neighbours: SmallWorldNeighbours,
) -> int:
    # Calculate energy of flattened matrix
    mat = x.reshape(N, N)
    return get_matrix_energy(mat, p, stubborn_agents, small_world_neighbours)


def check_minimal_energy(stubborn_agents: dict):
    # Draw minimal energy solution for this graph
    minimum_graph_energy = optimize.minimize(
        energy_function_flat,
        [10 * np.random.rand() for i in range(N * N)],
        constraints=[
            optimize.LinearConstraint(
                [k == i * N + j for k in range(N * N)],
                stubborn_agents[(i, j)],
                stubborn_agents[(i, j)],
            )
            for (i, j) in stubborn_agents
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


def main():
    logger.info("Starting simulation")

    matrix = 10 * np.random.rand(N, N)
    logger.info("Generated matrix")

    for p, small_world_neighbours, stubborn_agents in itertools.product(
        P_OPTIONS, SMALL_WORLD_NEIGHBOURS_OPTIONS, STUBBORN_AGENTS_OPTIONS
    ):
        if CHECK_MINIMAL_ENERGY:
            check_minimal_energy(stubborn_agents)
        degroot_simulation(matrix, p, stubborn_agents, small_world_neighbours)


if __name__ == "__main__":
    main()
