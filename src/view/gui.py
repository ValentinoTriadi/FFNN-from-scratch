from config.graphConfig import GraphConfig
from PyQt6.QtWidgets import (
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QTabWidget,
    QScrollArea,
    QLabel,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from .graph import GraphWidget, GraphModel
from typing import List
from .distributionTabs import DistributionTabs


class GUI(QMainWindow):
    """
    GUI Entry Point. this will hold all the widgets on the higher level
    """
    def __init__(self, graph_model: GraphModel, selected_distribution_layer : list[int]):
        super().__init__()
        self.setWindowTitle("Graph Visualization (Auto x-position)")
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet(f"background-color: {GraphConfig.BACKGROUND_COLOR};")

        # Create centrall widget, this control between the left and right side
        central_widget = QWidget()

        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Left Panel Initiation
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(0)
        self.left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.main_layout.addWidget(self.left_panel)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # Graph Visualization
        self.tab_widget.addTab(GraphWidget(graph_model), "Graph Structure")

        # Weight Distribution
        self.tab_widget.addTab(
            DistributionTabs.CreateWeightDistribution(graph_model, selected_distribution_layer),
            "Weight Distribution",
        )
        
        # Gradient Weight Distribution
        self.tab_widget.addTab(DistributionTabs.CreateGradientWeightDistribution(graph_model, selected_distribution_layer), "Gradient Weight Distribution")

        self.left_layout.addWidget(self.tab_widget)
