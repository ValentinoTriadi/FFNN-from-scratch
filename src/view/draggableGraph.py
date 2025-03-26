import pyqtgraph as pg
import numpy as np
from typing import List
from config.graphConfig import GraphConfig
from PyQt6.QtWidgets import (
    QScrollArea, QGraphicsProxyWidget, QGraphicsItem
)
from PyQt6.QtCore import QPointF
from view.paginatedTable import PaginatedTableWidget
from model.graph.model import GraphModel
from scipy.spatial import cKDTree

class GraphPG(pg.GraphItem):
    """
    GraphPG are the graphics item that paint all the node, edge, and text in the viewport.
    Parameter : 
    - model : All the neural netweork data
    """
    def __init__(self, model : GraphModel):
        super().__init__()
        self.draggedNode = None
        self._pos = None
        self._adj = None
        self._pen = None
        self._size = None
        self._symbol = None
        self._symbolBrush = None
        self.graphModel = model
        self.textItems: List[str] = []
        self.setAcceptHoverEvents(True)
        self.maxEdgeDistance = 15
        self._clickPos = None  # to record the initial click position
        
        # Create Table (Using pooling system for faster update)        
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)        

        self.paginatedTable = PaginatedTableWidget(data=[], rows_per_page=GraphConfig.WEIGHT_TABLE_ROWS)
        self.paginatedTable.closeRequested.connect(self.hidePopup)
        
        self.scrollArea.setWidget(self.paginatedTable)

        # Prepare the popup proxy; hide initially. Used for popup shown when double clicking a node
        self.popupProxy = QGraphicsProxyWidget()
        self.popupProxy.setWidget(self.scrollArea)
        self.popupProxy.hide()

        self.popupNodeIndex = None

    def hidePopup(self):
        self.popupProxy.hide()

    def setGraphData(self, pos: np.ndarray, adj: np.ndarray, pen, size: float, symbol: str, symbolBrush):
        self._pos = pos
        self.kd_tree = cKDTree(pos)
        self._adj = adj
        self._pen = pen
        self._size = size
        self._symbol = symbol
        self._symbolBrush = symbolBrush

        filtered_adj = []
        filtered_indices = []
        for idx, edge in enumerate(adj):
            i, j = edge
            p1, p2 = pos[i], pos[j]
            dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if dist < self.maxEdgeDistance:
                filtered_adj.append(edge)
                filtered_indices.append(idx)

        filtered_adj = np.array(filtered_adj, dtype=int)
        filtered_indices = np.array(filtered_indices, dtype=int)

        if isinstance(pen, np.ndarray):
            pen = pen[filtered_indices]
        else:
            pen = [pen[i] for i in filtered_indices]

        super().setData(
            pos=pos,
            adj=filtered_adj,
            pen=pen,
            size=size,
            symbol=symbol,
            symbolBrush=symbolBrush
        )

    def initTextItems(self, pos, textItems):
        self.textItems.clear()
        if textItems is not None:
            for i, textItem in enumerate(textItems):
                textItem.setPos(pos[i, 0], pos[i, 1])
                textItem.setAnchor((0.5, 0.5))
                textItem.setZValue(100)
                
                self.textItems.append(textItem)
                textItem.setHtml(
                    f'<span style="color{GraphConfig.TEXT_COLOR}; font-size:{GraphConfig.TEXT_SIZE}px;">'
                    f'{textItem.textItem.toPlainText()}</span>'
                )

    def updateTextItems(self, textItems):
        viewbox = self._viewBox()
        visible_rect = viewbox.mapToScene(viewbox.viewport().rect()).boundingRect()
        
        if textItems:
            for i, textItem in enumerate(textItems):
                node_pos = self._pos[i]
                node_point = QPointF(node_pos[0], node_pos[1])
                if visible_rect.contains(node_point):
                    textItem.setVisible(True)
                    textItem.setPos(node_pos[0], node_pos[1])
                    textItem.setAnchor((0.5, 0.5))
                    textItem.setHtml(
                        f'<span style="color{GraphConfig.TEXT_COLOR}; font-size:{GraphConfig.TEXT_SIZE}px;">'
                        f'{textItem.textItem.toPlainText()}</span>'
                    )
                else:
                    textItem.setVisible(False)

    def mousePressEvent(self, ev):
        viewbox = self._viewBox()
        pos_view = viewbox.mapSceneToView(ev.scenePos())
        if self._pos is not None:
            # Query the KD-tree for the nearest node
            dist, node_index = self.kd_tree.query([pos_view.x(), pos_view.y()])
            if dist < 1:  # if click is close enough to a node
                self.draggedNode = node_index
                self._clickPos = pos_view
                ev.accept()
                return
        ev.ignore()

    def mouseDoubleClickEvent(self, ev):
        """
        On a double-click event, find the node that was clicked and show the popup.
        """
        viewbox = self._viewBox()
        pos_view = viewbox.mapSceneToView(ev.scenePos())
        if self._pos is not None:
            distances, node_index = self.kd_tree.query([pos_view.x(), pos_view.y()])
            if distances < 2:  # within hit radius
                self.showScrollablePanel(node_index)
                ev.accept()
                return
        ev.ignore()

    def mouseReleaseEvent(self, ev):
        self.draggedNode = None
        self._clickPos = None
        ev.accept()

    def showScrollablePanel(self,node_index: int):
        """
        Updates the persistent paginated table with new data for the clicked node,
        then positions the popup near that node.
        """
        print("Double click event received for node:", node_index)
        self.popupNodeIndex = node_index

        new_data = []
        selected_node = self.graphModel.getNode(node_index)
        edges = self.graphModel.getAllEdgesFromNode(selected_node)
        for edge in edges:
            weight_val = f"{edge.weight:.3f}"
            gradient_val = f"{edge.gradient_weight:.3f}"
            new_data.append((edge.label, gradient_val,weight_val))
        
        self.paginatedTable.updateData(new_data, selected_node.color, selected_node.text)

        if not self.popupProxy.scene():
            self._viewBox().scene().addItem(self.popupProxy)
        
        viewbox = self._viewBox()
        node_pos = self._pos[node_index]
        scene_point = viewbox.mapViewToScene(QPointF(node_pos[0], node_pos[1]))

        popup_size = self.scrollArea.sizeHint()
        x = scene_point.x()
        y = scene_point.y() - popup_size.height() / 2
        views = viewbox.scene().views()
        view = views[0]

        viewport_rect = view.mapToScene(view.viewport().rect()).boundingRect()

        # Clamp x within the viewport.
        if x + popup_size.width() > viewport_rect.right():
            x -= ((x + popup_size.width()) - viewport_rect.right() + GraphConfig.MARGIN_RIGHT)
        if x < viewport_rect.left():
            x = 0

        # Clamp y within the viewport.
        if y + popup_size.height() > viewport_rect.bottom():
            y = viewport_rect.bottom() - popup_size.height()
        if y < viewport_rect.top():
            y = viewport_rect.top()

        self.scrollArea.adjustSize()
        self.popupProxy.setPos(x, y)
        self.popupProxy.show()
