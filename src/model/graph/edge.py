from src.model.graph.node import *
class GraphEdge:
    def __init__(self, input_node : GraphNode, output_node : GraphNode, line_color : str):
        self.input_node = input_node
        self.output_node = output_node
        self.color = line_color
