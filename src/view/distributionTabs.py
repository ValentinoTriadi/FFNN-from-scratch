from PyQt6.QtWidgets import QTabWidget, QSizePolicy
from view.layersDistribution import SinglePlotDistribution, MultiPlotDistribution
from model.graph.model import GraphModel

class DistributionTabs(QTabWidget):
    """
    DistributionTabs are Tab widget that store either Weight Distribution Tabs or Gradient Weight Distribution Tabs
    """
    def __init__(self,  parent = None):
        super().__init__(parent)
        self.setTabPosition(QTabWidget.TabPosition.West)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
    @classmethod
    def CreateWeightDistribution(cls, graph_model : GraphModel, layer_selected :list[int], parent = None):
        dist = cls(parent)
        dist.addTab(SinglePlotDistribution.WeightDistribution(graph_model, layer_selected, 'histogram'), "Histogram (Combined)")
        dist.addTab(SinglePlotDistribution.WeightDistribution(graph_model, layer_selected, 'gaussian'), "Gaussian (Combined)")
        dist.addTab(SinglePlotDistribution.WeightDistribution(graph_model, layer_selected, 'scatter'), "Scatter (Combined)")
        dist.addTab(MultiPlotDistribution.WeightDistribution(graph_model,layer_selected, "histogram"), "Histogram (Single)")
        dist.addTab(MultiPlotDistribution.WeightDistribution(graph_model, layer_selected,"gaussian"), "Gaussian (Single)")
        dist.addTab(MultiPlotDistribution.WeightDistribution(graph_model, layer_selected,"scatter"), "Scatter (Single)")

        return dist
    
    @classmethod
    def CreateGradientWeightDistribution(cls, graph_model : GraphModel, layer_selected :list[int], parent = None):
        dist = cls(parent)
        dist.addTab(SinglePlotDistribution.GradientWeightDistribution(graph_model, layer_selected, 'histogram'), "Histogram (Combined)")
        dist.addTab(SinglePlotDistribution.GradientWeightDistribution(graph_model, layer_selected, 'gaussian'), "Gaussian (Combined)")
        dist.addTab(SinglePlotDistribution.GradientWeightDistribution(graph_model, layer_selected, 'scatter'), "Scatter (Combined)")
        dist.addTab(MultiPlotDistribution.GradientWeightDistribution(graph_model,layer_selected, 'histogram'),"Histogram (Single)")
        dist.addTab(MultiPlotDistribution.GradientWeightDistribution(graph_model, layer_selected,"gaussian"), "Gaussian (Single)")
        dist.addTab(MultiPlotDistribution.GradientWeightDistribution(graph_model, layer_selected,"scatter"), "Scatter (Single)")

        return dist
