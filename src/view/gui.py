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

        # self.left_layout = QVBoxLayout(self.left_panel)
        # self.main_layout.addWidget(self.left_panel)

        # Right Panel Initiation
        # self.right_panel = QWidget()
        # self.right_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        # self.right_layout = QVBoxLayout(self.right_panel)
        # self.right_layout.setContentsMargins(10, 10, 10, 10)
        # self.right_layout.setSpacing(10)

        # self.right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # self.main_layout.addWidget(self.right_panel, stretch=1)

        # Stacking widget for view Switching
        # self.view_stack = QStackedWidget()
        # self.left_layout.addWidget(self.view_stack)

        # Init View Widgets
        # self.graph_widget = GraphWidget(graph_model)
        # self.other = QWidget()

        # self.view_stack.addWidget(self.graph_widget)
        # self.view_stack.addWidget(self.other)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.tab_widget.addTab(GraphWidget(graph_model), "Graph Structure")
        self.tab_widget.addTab(
            DistributionTabs.CreateWeightDistribution(graph_model, selected_distribution_layer),
            "Weight Distribution",
        )
        self.tab_widget.addTab(DistributionTabs.CreateGradientWeightDistribution(graph_model, selected_distribution_layer), "Gradient Weight Distribution")

        self.left_layout.addWidget(self.tab_widget)

        # self._initLeftTab(['Graph', 'Other'])
        # self._initController()

    def _initLayout(self):
        self.main_layout = QHBoxLayout(self.app)
        self.left_layout = QVBoxLayout(self.main_layout)
        self.right_layout = QVBoxLayout(self.right_layout)

    def _initController(self):
        save_button = QPushButton("Save")
        load_button = QPushButton("Load")
        self.right_layout.addWidget(save_button)
        self.right_layout.addWidget(load_button)

    def _initLeftTab(self, tabs: List[str]):
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.TabPosition.North)

        # Connect the currentChanged signal to a handler
        tab_widget.currentChanged.connect(self._handleTabChange)

        for tab_name in tabs:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)

            label = QLabel(f"Content for {tab_name}")
            content_layout.addWidget(label)

            scroll_area.setWidget(content_widget)
            tab_widget.addTab(scroll_area, tab_name)

        self.left_layout.addWidget(tab_widget)

    def _handleTabChange(self, index: int):
        self.view_stack.setCurrentIndex(index)

    def _initView(self, view: QWidget):

        # self.setCentralWidget(self.app)
        self.left_layout.addWidget(view)
