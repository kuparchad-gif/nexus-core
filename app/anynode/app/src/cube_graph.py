import networkx as nx
import numpy as np
from math import pi as PI, sin
from typing import List

PHI = (1 + 5 ** 0.5) / 2

class MetatronCube:
    def __init__(self):
        self.graph = self._create_graph()

    def _create_graph(self) -> nx.Graph:
        G = nx.Graph()
        nodes = range(13)
        G.add_nodes_from(nodes)

        for i in range(13):
            for j in range(i + 1, 13):
                G.add_edge(i, j)

        return G

    def fuse(self, feature_vector: List[float], iterations: int = 5, delta_threshold: float = 1e-5) -> List[float]:
        if len(feature_vector) != 13:
            raise ValueError("Feature vector must have 13 elements.")

        for i, weight in enumerate(feature_vector):
            self.graph.nodes[i]['weight'] = weight

        for iteration in range(iterations):
            new_weights = {}
            for node in self.graph.nodes:
                neighbors = list(self.graph.neighbors(node))
                neighbor_weights = [self.graph.nodes[n]['weight'] for n in neighbors]

                new_weight = self.graph.nodes[node]['weight']
                for neighbor_weight in neighbor_weights:
                    new_weight += neighbor_weight * PHI * sin(2 * PI * iteration / iterations)

                new_weights[node] = new_weight / (len(neighbors) + 1)

            delta = sum(abs(new_weights[i] - self.graph.nodes[i]['weight']) for i in range(13))
            if delta < delta_threshold:
                break

            for node, weight in new_weights.items():
                self.graph.nodes[node]['weight'] = weight

        return [self.graph.nodes[i]['weight'] for i in range(13)]
