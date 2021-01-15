import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MplPlotLayout:
    def __init__(self, parent=None, width=5, height=4, dpi=100, haveToolbar=True):
        self.canvas = MplCanvas(parent, width, height, dpi)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, parent)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        # Create a placeholder widget to hold our toolbar and canvas.
        self.widget = QtWidgets.QWidget()
        self.widget.setLayout(layout)
