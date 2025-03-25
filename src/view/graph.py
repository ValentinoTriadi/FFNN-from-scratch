from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from model.graph.model import GraphModel
from view.draggableGraph import DraggableGraphItem
from config.graphConfig import GraphConfig
from utils.colorHelper import ColorHelper


class GraphWidget(QWidget):
    def __init__(self, graph_model: GraphModel, parent=None):
        super().__init__(parent)
        self.graph_model = graph_model
        self.initUI()

    def initUI(self):

        self.setStyleSheet("background-color: {};".format(GraphConfig.BACKGROUND_COLOR))
        pg.setConfigOptions(antialias=False)

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.graphics_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_layout)

        viewbox = pg.ViewBox(lockAspect=True)
        self.graphics_layout.addItem(viewbox)

        self.graph_item = DraggableGraphItem(self.graph_model)

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
        pen = ColorHelper.create_lines_array(edges_color, GraphConfig.LINE_SIZE)

        self.graph_item.setGraphData(
            pos=pos,
            adj=adj,
            pen=pen,
            size=GraphConfig.NODE_SIZE,
            symbol="o",
            symbolBrush=node_colors,
        )

        textItems = []
        for node in self.graph_model.nodes:
            textItem = pg.TextItem(text=node.text, color=GraphConfig.TEXT_COLOR)
            textItem.setPos(node.pos[0], node.pos[1])
            viewbox.addItem(textItem)
            textItems.append(textItem)
        self.graph_item.initTextItems(pos, textItems)
