from model.graph.node import *
class GraphEdge:
    """
    GraphEdge is model for an Edge that save data between 2 nodes. These edge used for saving these: 
    - Weight
    - Gradient Weight
    also can visualize the connection between nodes
    """
    def __init__(self,  input_node : GraphNode, output_node : GraphNode, line_color : str, weight : str, grad : str, label : str):
        self.input_node = input_node
        self.output_node = output_node
        self.color = line_color
        self.weight = weight
        self.gradient_weight = grad
        self.label = label