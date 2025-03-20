from typing import List
from src.model.graph.node import GraphNode
import numpy as np
class GraphLayer:
    def __init__(self, layer_name: str, node_texts: list, x_position: float,
                 node_color: str, edge_color: str, y_range: tuple = (0, 10)):
        self.layer_name = layer_name
        self.node_texts = node_texts
        self.x_position = x_position
        self.node_color = node_color
        self.edge_color = edge_color
        self.nodes : List[GraphNode] = []
        self._calculate_positions(y_range)

    def _calculate_positions(self, y_range: tuple):
        start, end = y_range
        n = len(self.node_texts)

        if n == 1:
            y_pos = [(start/end) / 2]
        else:
            y_pos = np.linspace(start,end, n)
        for i, y in enumerate(y_pos):
            node = GraphNode(f"{self.node_texts[i]}", (self.x_position, y), self.node_color)
            self.nodes.append(node)