from collections.abc import Callable
import numpy as np
import networkx as nx
from scipy import optimize
from functools import reduce


class SyncDegrootModel:
    _graph: nx.Graph
    
    def __init__(self, nodes=100, edge_probability=0.5, P=2, initial_opinions_distribution=np.random.uniform) -> None:
        self._graph = nx.fast_gnp_random_graph(nodes, edge_probability)
        if P == np.inf:
            self._energy = lambda A: lambda x: np.max([np.abs(x - A_i) for A_i in A]) # A is the neighbors' opinions
        else:
            self._energy = lambda A: lambda x: reduce(lambda s, A_i: np.add(s, np.float_power(np.abs(x - A_i), P)), A, 0) # A is the neighbors' opinions
        self._init_opinions(initial_opinions_distribution)

    def _init_opinions(self, distribution: Callable[[], np.double]):
        for n in self._graph.nodes:
            self._graph.nodes[n]["opinion"] = distribution()
        for n in self._graph.nodes:
            self._graph.nodes[n]["energy"] = self._energy([self._graph.nodes[i]["opinion"] for i in self._graph.adj[n]])(self._graph.nodes[n]["opinion"])

    def one_step_degroot_update(self):
        for n in self._graph.nodes:
            result = optimize.minimize_scalar(self._energy([self._graph.nodes[i]["opinion"] for i in self._graph.adj[n]]))
            x_min = result.x
            self._graph.nodes[n]["next_opinion"] = x_min
        for n in self._graph.nodes:
            self._graph.nodes[n]["energy"] = self._energy([self._graph.nodes[i]["next_opinion"] for i in self._graph.adj[n]])(self._graph.nodes[n]["next_opinion"])
            self._graph.nodes[n]["opinion"] = self._graph.nodes[n]["next_opinion"]

    def get_graph_energy(self) -> float:
        labels = nx.get_node_attributes(self._graph, 'energy')
        return sum([labels[l] for l in labels])
