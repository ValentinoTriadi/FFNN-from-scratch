from src.model.graph.model import GraphModel

from src.view.draggableGraph import DraggableGraphItem
import pyqtgraph as pg
import numpy as np
from src.config.graphConfig import GraphConfig
from src.utils.colorHelper import ColorHelper
class GraphView(pg.GraphicsLayoutWidget):
    def __init__(self, graph_model: GraphModel):
        super().__init__()
        self.graph_model = graph_model
        self.initUI()
        self.show()
    
    

    def initUI(self):
        self.setBackground(GraphConfig.BACKGROUND_COLOR)    
        viewbox = pg.ViewBox(lockAspect=True)
        self.addItem(viewbox)
        
        self.graph_item = DraggableGraphItem()
        viewbox.addItem(self.graph_item)
    
        pos_list = [node.pos for node in self.graph_model.nodes]
        pos = np.array(pos_list)
    
        adj_list = []
        for edge in self.graph_model.edges:
            i = self.graph_model.nodes.index(edge.input_node)
            j = self.graph_model.nodes.index(edge.output_node)
            adj_list.append((i, j))
        adj = np.array(adj_list)
    
        node_colors = [node.color for node in self.graph_model.nodes]

        edges_color = [edge.color for edge in self.graph_model.edges]
        pen = ColorHelper.create_lines_array(edges_color,GraphConfig.LINE_SIZE)
    
        self.graph_item.setGraphData(pos=pos, adj=adj, pen=pen, size=GraphConfig.NODE_SIZE, symbol='o', symbolBrush=node_colors)
        
        textItems = []
        for i, node in enumerate(self.graph_model.nodes):
            textItem = pg.TextItem(text=node.text, color=GraphConfig.TEXT_COLOR)
            textItem.setPos(node.pos[0], node.pos[1])
            viewbox.addItem(textItem)
            textItems.append(textItem)
        self.graph_item.initTextItems(pos,textItems)