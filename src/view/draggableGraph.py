import pyqtgraph as pg
import numpy as np
from typing import List
from src.config.graphConfig import GraphConfig
class DraggableGraphItem(pg.GraphItem):
    def __init__(self):
        super().__init__()
        self.draggedNode = None
        self._pos = None
        self._adj = None
        self._pen = None
        self._size = None
        self._symbol = None
        self._symbolBrush = None
        self.textItems: List[str] = []
        self.setAcceptHoverEvents(True)
    
    def setGraphData(self, pos : np.ndarray,adj: np.ndarray, pen ,size : float, symbol: str, symbolBrush):
        self._pos = pos
        self._adj = adj
        self._pen = pen
        self._size = size
        self._symbol = symbol
        self._symbolBrush = symbolBrush
        super().setData(pos=pos, adj=adj, pen=pen, size=size, symbol=symbol, symbolBrush=symbolBrush)

    def initTextItems(self, pos, textItems):
        self.textItems.clear()
        if textItems is not None:
            for i, textItem in enumerate(textItems):
                textItem.setPos(pos[i, 0], pos[i, 1])
                textItem.setAnchor((0.5, 0.5))
                textItem.setZValue(100)            
                self.textItems.append(textItem)
                textItem.setHtml(f'<span style="color{GraphConfig.TEXT_COLOR}; font-size:{GraphConfig.TEXT_SIZE}px;">{textItem.textItem.toPlainText()}</span>')

    def updateTextItems(self, textItems):
        if textItems:
            for i, textItem in enumerate(textItems):
                textItem.setPos(self._pos[i, 0], self._pos[i, 1])
                textItem.setAnchor((0.5, 0.5))
                textItem.setHtml(f'<span style="color{GraphConfig.TEXT_COLOR}; font-size:{GraphConfig.TEXT_SIZE}px;">{textItem.textItem.toPlainText()}</span>')

    def mousePressEvent(self, ev):
        viewbox = self._viewBox()  
        pos_view = viewbox.mapSceneToView(ev.scenePos())
        if self._pos is not None:
            distances = np.linalg.norm(self._pos - np.array([pos_view.x(), pos_view.y()]), axis=1)
            node_index = np.argmin(distances)
            if distances[node_index] < 1:  
                self.draggedNode = node_index
                ev.accept()
                return
        ev.ignore()
    
    def mouseMoveEvent(self, ev):
        if self.draggedNode is not None:
            viewbox = self._viewBox()
            pos_view = viewbox.mapSceneToView(ev.scenePos())
            self._pos[self.draggedNode] = [pos_view.x(), pos_view.y()]
            self.setGraphData(self._pos, self._adj, self._pen, self._size, self._symbol, self._symbolBrush)
            if hasattr(self, 'textItems'):
                self.updateTextItems(self.textItems)
            ev.accept()
        else:
            ev.ignore()
    
    def mouseReleaseEvent(self, ev):
        self.draggedNode = None
        ev.accept()