import numpy as np
from scipy import optimize
from functools import reduce
from matplotlib import pyplot as plt
import networkx as nx
from SyncDegrootModel import SyncDegrootModel

ITERATIONS = 10

def main():
    degroot = SyncDegrootModel(nodes=30, edge_probability=1, P=20, initial_opinions_distribution=lambda: np.random.uniform(5, 50))
    energies = []
    energy = degroot.get_graph_energy()
    energies.append(energy)
    print(f"Initial Energy: f{energy}")

    for i in range(ITERATIONS):
        degroot.one_step_degroot_update()
        energy = degroot.get_graph_energy()
        energies.append(energy)
        print(f"{i+1} Energy: f{energy}")

    # plt.figure("Energy")
    # labels = nx.get_node_attributes(degroot._graph, "energy")
    # nx.draw_networkx(degroot._graph, labels=labels)

    # plt.figure("Opinion")
    # labels = nx.get_node_attributes(degroot._graph, "opinion")
    # nx.draw_networkx(degroot._graph, labels=labels)

    plt.figure("Total Energy (log) vs Time")
    plt.scatter(range(ITERATIONS + 1), np.log10(energies))
    m, b = np.polyfit(range(ITERATIONS + 1), np.log10(energies), 1)
    plt.plot(range(ITERATIONS + 1), m*range(ITERATIONS + 1) + b, "r--", label=f"y={m:.2f}x+{b:.2f}")
    plt.legend(fontsize=9)
    plt.xlabel("Time")
    plt.ylabel("Energy")
    
    plt.show()


if __name__ == "__main__":
    main()
