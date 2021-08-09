from collections.abc import Callable
import numpy as np
import networkx as nx
from scipy import optimize
from functools import reduce


class SyncDegrootModel:
    _graph: nx.Graph
    
    def __init__(self, nodes=100, edge_probability=0.5, P=2, initial_opinions_distribution=np.random.uniform) -> None:
        self._graph = nx.fast_gnp_random_graph(nodes, edge_probability)
        self._energy = lambda A: lambda x: reduce(lambda s, A_i: s + np.abs(x - A_i)**P, A, 0) # A is the neighbors' opinions
        self._init_opinions(initial_opinions_distribution)

    def _init_opinions(self, distribution: Callable[[], np.double]):
        for n in self._graph.nodes:
            self._graph.nodes[n]["opinion"] = distribution()

    def one_step_degroot_update(self):
        for n in self._graph.nodes:
            result = optimize.minimize_scalar(self._energy([self._graph.nodes[i]["opinion"] for i in self._graph.adj[n]]))
            x_min = result.x
            self._graph.nodes[n]["next_opinion"] = x_min
        for n in self._graph.nodes:
            self._graph.nodes[n]["next_opinion"] = self._graph.nodes[n]["opinion"]