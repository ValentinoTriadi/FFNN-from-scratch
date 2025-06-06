
from typing import List
from model.graph.node import GraphNode
from model.graph.layer import GraphLayer
from model.graph.edge import GraphEdge
from config.graphConfig import GraphConfig
from utils.colorHelper import ColorHelper
import numpy as np
class GraphModel:
    """
    GraphModel is the main model which are the connector between the ML model and the views
    Parameter: 
    - neuron_list : List of integer that specify the amount of neuron each layers have
    - weight_neuron_data : List of NpArray that save the weight of each edge
    - weight_grads_data : List of NpArray that save the gradient weight of each edge
    """
    def __init__(self,  neuron_list : list[int], weight_neuron_data : list[list[list[str]]], weight_grads_data : list[list[list[str]]], training_loss : list[float], validation_loss : list[float]):
        self.nodes : List[GraphNode] = []
        self.edges : List[GraphEdge]= []
        self.layers : List[GraphLayer] = []
        self.weights: List[np.ndarray] = [] 
        self.gradien_weight : List[np.ndarray] = []
        self.training_loss = training_loss
        self.validation_loss = validation_loss
        
        self.x_spacing = GraphConfig.LAYER_SPACING
        self.y_range = GraphConfig.LAYER_Y_RANGE
        
        self.layer_colors = ColorHelper.generate_colors(len(neuron_list), 'dark')
        
        self._initiate_layers(neuron_list)
        self._save_weight_gradient_weight(weight_neuron_data, weight_grads_data)
        self._create_fully_connected_edges()
        

    # Save the weight and gradient weight for easier access        
    def _save_weight_gradient_weight(self, weight_neuron_data : list
    [list[list[str]]], weight_grads_data : list[list[list[str]]]):
        if isinstance(weight_neuron_data, np.ndarray):
            if weight_neuron_data.ndim == 3:
                self.weights = [weight_neuron_data[i] for i in range(weight_neuron_data.shape[0])]
                self.gradien_weight = [weight_grads_data[i] for i in range(weight_grads_data.shape[0])]
            else:
                self.weights = weight_neuron_data
                self.gradien_weight = weight_grads_data
        else:
            self.weights = weight_neuron_data 
            self.gradien_weight = weight_grads_data

    # Create each layer with the given number of neurons.
    def _initiate_layers(self, neuron_list : list[int]):
        for i, num_neurons in enumerate(neuron_list):
            
            x_position = i * self.x_spacing 
            neuron_pre_name = ""
            if(i == 0):
                neuron_pre_name = f"Input"
            elif(i == (len(neuron_list)-1)):
                neuron_pre_name = f"Output"
            else:
                neuron_pre_name = f"Hidden"
            
            node_color = self.layer_colors[i]
            edge_color = self.layer_colors[i]
            layer = GraphLayer(i , num_neurons, neuron_pre_name,x_position, node_color, edge_color, self.y_range)
            self._add_layer(layer)

    def _add_layer(self, layer: GraphLayer):
        self.layers.append(layer)
        for node in layer.nodes:
            self._add_node(node)

    def _add_node(self, node : GraphNode):
        self.nodes.append(node)
    
    def _create_fully_connected_edges(self):
        for layer_index in range(len(self.layers) - 1):
            current_layer : GraphLayer  = self.layers[layer_index]
            next_layer : GraphLayer = self.layers[layer_index + 1]
            
            bias_counter = 1
            for i,node_from in enumerate(current_layer.nodes):
                for j, node_to in enumerate(next_layer.nodes):   
                    
                    from_label = ""
                    to_label = ""
                    if("Input" in current_layer.text_pre_header):
                        from_label = f"I{i+1}"
                    else:
                        from_label = f"H{current_layer.index + 1}-{j+1}"
                    if(i == len(current_layer.nodes)-1):
                        from_label = f"B{bias_counter}"
                    if("Output" in next_layer.text_pre_header):
                        to_label = f"O{j+1}"
                    else:
                        to_label = f"H{next_layer.index + 1}-{j+1}"
                    edge_label = f"W[{from_label}][{to_label}]"
                    edge = GraphEdge(node_from, node_to, current_layer.edge_color, self.weights[layer_index][i][j],self.gradien_weight[layer_index][i][j],edge_label)
                    self.edges.append(edge)

    def getAllEdgesFromNode(self, node) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.input_node == node]
    
    def getNode(self, node_index : int):
        return self.nodes[node_index]