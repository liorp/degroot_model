import numpy as np
from scipy import optimize
from functools import reduce
from matplotlib import pyplot as plt
import networkx as nx
from SyncDegrootModel import SyncDegrootModel

def main():
    degroot = SyncDegrootModel(nodes=10, edge_probability=0.8)
    for i in range(1000):
        degroot.one_step_degroot_update()
    labels = nx.get_node_attributes(degroot._graph, 'opinion')
    nx.draw_networkx(degroot._graph, labels=labels)
    plt.show()

if __name__ == "__main__":
    main()
