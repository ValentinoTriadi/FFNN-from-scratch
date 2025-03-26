import math
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel
from PyQt6.QtCore import QTimer
from model.graph.model import GraphModel

class SinglePlotDistribution(QWidget):
    def __init__(self, layer_data: list[tuple[int,np.ndarray]], distribution_mode: str = 'gaussian', parent=None):
        """
        Plot layers data into chart or plot. we at least gave 3 plot which is scatter, histogram, and Gaussian Curve
        Parameters:
          layer_data: List of tuples (layer_index, data) where data is a numpy array.
          distribution_mode: 'histogram', 'scatter', or 'gaussian' (default)
        """
        super().__init__(parent)
        self.setWindowTitle("Single Plot Distribution")
        self.layer_data = layer_data
        self.distribution_mode = distribution_mode.lower()
        
        layout = QVBoxLayout(self)
        self.plotWidget = pg.PlotWidget(title="Combined Distribution")
        layout.addWidget(self.plotWidget)
        self.plotWidget.setBackground('w')
        self.plotWidget.getAxis('bottom').setPen('k')
        self.plotWidget.getAxis('left').setPen('k')
        self.legend = self.plotWidget.addLegend(offset=(10, 10))
        self.legend.anchor((1, 0), (1, 0))
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updatePlot)
        self.timer.start(1000)
        
        self.updatePlot()
    
    def setupPlot(self):
        self.plotWidget.clear()
        self.legend = self.plotWidget.addLegend(offset=(10, 10))
        self.legend.anchor((1, 0), (1, 0))
    
    def updatePlot(self):
        self.setupPlot()
        if self.distribution_mode == 'histogram':
            self.createHistogramDistribution()
        elif self.distribution_mode == 'scatter':
            self.createScatterDistribution()
        else:
            self.createGaussianCurveDistribution()
    
    def createHistogramDistribution(self, bin_count=50):
        offset_step = 10  # vertical offset to separate histograms
        for i, (layer_idx, data) in enumerate(self.layer_data):
            hist, bin_edges = np.histogram(data.ravel(), bins=bin_count)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            bar_width = bin_edges[1] - bin_edges[0]
            offset = i * offset_step
            color = pg.intColor(i, hues=len(self.layer_data), alpha=200)
            bg = pg.BarGraphItem(
                x=bin_centers, height=hist, width=bar_width, brush=color, pen=pg.mkPen(None), y0=offset
            )
            self.plotWidget.addItem(bg)
            
            dummy = pg.PlotDataItem(name=f"Layer {layer_idx}")
            self.plotWidget.addItem(dummy)
            dummy.hide()
        self.plotWidget.setLabel('bottom', 'Weight Value', color='k')
        self.plotWidget.setLabel('left', 'Frequency + Offset', color='k')
        
    
    def createScatterDistribution(self, bin_count=50):
        for i, (layer_idx, data) in enumerate(self.layer_data):
            hist, bin_edges = np.histogram(data.ravel(), bins=bin_count)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            color = pg.intColor(i, hues=len(self.layer_data), alpha=200)
            scatter = pg.ScatterPlotItem(
                x=bin_centers, y=hist,
                pen=None, brush=color, size=8, name=f"Layer {layer_idx}"
            )
            self.plotWidget.addItem(scatter)
        self.plotWidget.setLabel('bottom', 'Weight Value', color='k')
        self.plotWidget.setLabel('left', 'Frequency', color='k')
        # self.plotWidget.enableAutoRange()
    
    def createGaussianCurveDistribution(self):
        # Determine global x-range from all layers.
        
        all_data = np.concatenate([data.ravel() for (_,data) in self.layer_data])
        min_val, max_val = np.min(all_data), np.max(all_data)
        margin = 0.2 * (max_val - min_val) if max_val > min_val else 1
        x = np.linspace(min_val - margin, max_val + margin, 300)
        for i, (layer_idx, data) in enumerate(self.layer_data):
            mean = np.mean(data)
            std = np.std(data)
            if std < 1e-12:
                continue
            pdf = (1.0 / (std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean)/std)**2)
            color = pg.intColor(i, hues=len(self.layer_data), alpha=150)
            curve = self.plotWidget.plot(
                x, pdf,
                pen=pg.mkPen(color, width=2), name=f"Layer {layer_idx}"
            )
            baseline = self.plotWidget.plot(x, np.zeros_like(x), pen=None)
            fill = pg.FillBetweenItem(curve, baseline, brush=color)
            self.plotWidget.addItem(fill)
        self.plotWidget.setLabel('bottom', 'Weight Value', color='k')
        self.plotWidget.setLabel('left', 'Probability Density', color='k')
        # self.plotWidget.enableAutoRange()

    @classmethod
    def WeightDistribution(cls, graph_model : GraphModel, layer_index_list : list[int],distribution_mode : str = 'gaussian', parent = None):
        
        layers_data = []
        for idx in layer_index_list:
            if 0 <= idx < len(graph_model.weights):
                data = graph_model.weights[idx]
                layers_data.append((idx,data))

        layer_dist = cls(layers_data, distribution_mode, parent)
        return layer_dist
    
    @classmethod
    def GradientWeightDistribution(cls, graph_model : GraphModel, layer_index_list : list[int],distribution_mode : str = 'gaussian', parent = None):
        
        layers_data = []
        for idx in layer_index_list:
            if 0 <= idx < len(graph_model.weights):
                data = graph_model.gradien_weight[idx]
                layers_data.append((idx,data))

        layer_dist = cls(layers_data, distribution_mode, parent)
        return layer_dist


class MultiPlotDistribution(QWidget):
    def __init__(self, layer_data: list[tuple[int, np.ndarray]], distribution_mode: str = 'gaussian', parent=None):
        """
        This class is just combining Multiple SinglePlotDistribution that each layer get it's own Graph
        Parameters:
          layer_data: List of tuples (layer_index, data)
          distribution_mode: Distribution mode for each plot ('gaussian', 'histogram', or 'scatter')
        """
        super().__init__(parent)
        self.setWindowTitle("Multi-Layer Distribution")
        self.distribution_mode = distribution_mode.lower()
        layout = QVBoxLayout(self)
        
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        layout.addWidget(self.scrollArea)
        container = QWidget()
        self.containerLayout = QVBoxLayout(container)
        self.scrollArea.setWidget(container)
        
        
        for layer_idx, data in layer_data:
            
            widget = SinglePlotDistribution(layer_data=[(layer_idx, data)], distribution_mode=self.distribution_mode)
            self.containerLayout.addWidget(widget)

    @classmethod
    def WeightDistribution(cls, graph_model : GraphModel, layer_index_list : list[int],distribution_mode : str = 'gaussian', parent = None):
        
        layers_data = []
        for idx in layer_index_list:
            if 0 <= idx < len(graph_model.weights):
                data = graph_model.weights[idx]
                layers_data.append((idx,data))

        layer_dist = cls(layers_data, distribution_mode, parent)
        return layer_dist
    @classmethod
    def GradientWeightDistribution(cls, graph_model : GraphModel, layer_index_list : list[int],distribution_mode : str = 'gaussian', parent = None):
        
        layers_data = []
        for idx in layer_index_list:
            if 0 <= idx < len(graph_model.weights):
                data = graph_model.gradien_weight[idx]
                layers_data.append((idx,data))

        layer_dist = cls(layers_data, distribution_mode, parent)
        return layer_dist