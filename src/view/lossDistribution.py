from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg

class LossDistribution(QWidget):
    """
    Widget to plot loss over epochs as a line chart.
    """

    def __init__(self, loss_values: list[float], title : str, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        plot_widget = pg.PlotWidget()
        plot_widget.setTitle(title, size="20pt")
        plot_widget.setLabel("left", "Loss")
        plot_widget.setLabel("bottom", "Epoch")

        print(f"[LossDistribution] Title: {title}")
        print(f"[LossDistribution] Loss values ({len(loss_values)}): {loss_values}")

        plot_widget.plot(list(range(1, len(loss_values) + 1)), loss_values, pen='r')

        layout.addWidget(plot_widget)
        self.setLayout(layout)
