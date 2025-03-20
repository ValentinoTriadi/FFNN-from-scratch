import numpy as np
class GraphNode:
    def __init__(self, text : str, pos: tuple, color: str):
        self.text = text
        self.pos = np.array(pos, dtype=float)
        self.color = color
        self.adjacent = []
        self.next = None