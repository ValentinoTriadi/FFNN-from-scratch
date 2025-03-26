from typing import List
from model.graph.node import GraphNode
import numpy as np

class GraphLayer:
    """
    GraphLayer is a model that represent the layer in FFNN. it can be an Input Layer, Hidden Layer, and even Output Layer
    Parameter: 
    - index : The layer index (start from 0)
    - neuron_num : The amount of neuron or node need to be created
    - text_pre_header : The Header of each node name (E.g: Input, Hidden, Output)
    - x_position : Position of the layer relative to graph view
    - y_range : This is the difference in vertical range between each node
    """
    bias_idx = 1
    def __init__(self, index : int, neuron_num: int, text_pre_header: str, x_position: float,
                 node_color: str, edge_color: str, y_range: tuple = (0, 10)):
        self.index = index
        self.text_pre_header = text_pre_header
        self.x_position = x_position
        self.node_color = node_color
        self.edge_color = edge_color
        self.neuron_num = neuron_num
        self.nodes: List[GraphNode] = []
        self._calculate_positions(y_range)

    def _calculate_positions(self, y_range: tuple):
        start, end = y_range
        y_pos = np.linspace(start, end, self.neuron_num)
    
        for i in range(self.neuron_num):
            pos = (self.x_position, y_pos[i])
            title = self.text_pre_header
            idx = i 
            
            # Handle Bias
            if(self.text_pre_header != "Output" and i == 0):
                title = "Bias"
                idx = GraphLayer.bias_idx
                GraphLayer.bias_idx += 1
            layer_idx = self.index
            if(self.text_pre_header == "Output" or self.text_pre_header == "Input"):
                layer_idx = ""
            node = GraphNode(f"{title}{layer_idx}-{idx+1}", pos, self.node_color)
            self.nodes.append(node)
    