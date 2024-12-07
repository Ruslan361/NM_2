# custom_layouts.py

from PySide6.QtWidgets import QVBoxLayout, QCheckBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class GraphLayout(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.checkBoxLogScale = QCheckBox("Логарифмическая шкала")
        self.addWidget(self.checkBoxLogScale)
        self.canvas = MatplotlibGraph()
        self.addWidget(self.canvas)
        self.checkBoxLogScale.stateChanged.connect(self.toggle_log_scale)

    def toggle_log_scale(self):
        if self.checkBoxLogScale.isChecked():
            self.canvas.ax.set_yscale('log')
        else:
            self.canvas.ax.set_yscale('linear')
        self.canvas.draw()

    def clear(self):
        self.canvas.clear()

    def set_ylabel(self, label):
        self.canvas.ax.set_ylabel(label)

    def set_xlabel(self, label):
        self.canvas.ax.set_xlabel(label)

    def set_title(self, title):
        self.canvas.ax.set_title(title)

    def plot(self, X, Y, label=''):
        self.canvas.ax.plot(X, Y, label=label)

    def draw(self):
        self.canvas.draw()

    def legend(self):
        self.canvas.ax.legend()

class MatplotlibGraph(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def clear(self):
        self.ax.clear()
        self.draw()
