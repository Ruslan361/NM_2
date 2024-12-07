# plot_manager.py

class PlotManager:
    def __init__(self, graph_layout):
        self.graph_layout = graph_layout

    def plot_test_task(self, x, v, list_u):
        self.graph_layout.clear()
        self.graph_layout.plot(x, v, label='Численное решение v(x)')
        self.graph_layout.plot(x, list_u, label='Точное решение u(x)')
        self.graph_layout.set_title("Тестовая задача")
        self.graph_layout.set_xlabel("x")
        self.graph_layout.set_ylabel("u(x), v(x)")
        self.graph_layout.legend()
        self.graph_layout.draw()

    def plot_main_task(self, x, v, v2_interp):
        self.graph_layout.clear()
        self.graph_layout.plot(x, v, label='$v_{n}(x)$')
        self.graph_layout.plot(x, v2_interp, label='$v_{2n}(x)$')
        self.graph_layout.set_title("Основная задача")
        self.graph_layout.set_xlabel("x")
        self.graph_layout.set_ylabel("v(x)")
        self.graph_layout.legend()
        self.graph_layout.draw()
