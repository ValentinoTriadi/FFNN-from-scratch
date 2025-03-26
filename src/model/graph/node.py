import numpy as np
class GraphNode:
    """
    GraphNode is a linked list model representing one neuron in FFNN Neural network
    Parameter: 
    - text : the text written in the node
    - pos : position of the node in the viewport
    - color : the color of the node
    """
    def __init__(self, text : str, pos: tuple, color: str):
        self.text = text
        self.pos = np.array(pos, dtype=float)
        self.color = color