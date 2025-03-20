
from src.config.graphConfig import GraphConfig
from PyQt6.QtWidgets import  QMainWindow, QWidget, QVBoxLayout
from src.view.graph import GraphView, GraphModel
from PyQt6.QtWidgets import QApplication, QLabel
import sys


class GUI(QMainWindow):
    def __init__(self,graph_model : GraphModel):
        super().__init__()
        self.setWindowTitle("Graph Visualization (Auto x-position)")
        self.setGeometry(100, 100, 1000, 1080)
        self.setStyleSheet(F"background-color: {GraphConfig.BACKGROUND_COLOR};")        
        self.graph_model = graph_model
        self._initUI()
        self.show()

    def _initUI(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.setCentralWidget(widget)

        self.graph_view = GraphView(graph_model=self.graph_model)
        layout.addWidget(self.graph_view)
        