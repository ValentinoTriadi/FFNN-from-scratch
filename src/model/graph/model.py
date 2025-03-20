
from typing import List
from src.model.graph.node import GraphNode
from src.model.graph.layer import GraphLayer
from src.model.graph.edge import GraphEdge
from src.utils.colorHelper import ColorHelper
class GraphModel:
    def __init__(self):
        self.head = None
        self.nodes : List[GraphNode] = []
        self.edges : List[GraphEdge]= []
        self.layers : List[GraphLayer] = []

    def add_layer(self, layer: GraphLayer):
        self.layers.append(layer)
        for node in layer.nodes:
            self.add_node(node)

    def add_node(self, node : GraphNode):
        if self.head is None:
            self.head = node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = node
        self.nodes.append(node)
    
    def create_fully_connected_edges(self):
        for i in range(len(self.layers) - 1):
            current_layer : GraphLayer  = self.layers[i]
            next_layer : GraphLayer = self.layers[i + 1]
            for node_from in current_layer.nodes:
                for node_to in next_layer.nodes:
                    edge = GraphEdge(node_from, node_to, current_layer.edge_color)
                    self.edges.append(edge)
                    node_from.adjacent.append(node_to)

    @classmethod
    def create_from_layers(cls, layers_data : list[list[str]], x_spacing : float = 10, y_range: tuple=(0,10)) -> 'GraphModel':
        model = cls()
        
        layer_colors = ColorHelper.generate_colors(len(layers_data), '')
            
        
        for i, node_labels in enumerate(layers_data):
            x_position = i * x_spacing 
            layer_name = f"Layer {i}"
            node_color = layer_colors[i]
            edge_color = layer_colors[i]
            layer = GraphLayer(layer_name, node_labels, x_position, node_color, edge_color, y_range)
            model.add_layer(layer)
        model.create_fully_connected_edges()
        return model
        