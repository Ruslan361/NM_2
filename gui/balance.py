# balance.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QSpinBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QGridLayout, QSizePolicy
)
from calculation_handler import CalculationHandler
from plot_manager import PlotManager
from custom_layouts import GraphLayout

class BalanceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NM_2")

        self.calculation_handler = CalculationHandler()
        self.graph_layout = GraphLayout()
        self.plot_manager = PlotManager(self.graph_layout)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Parameters Input Section
        params_layout = QGridLayout()

        # Task selection
        task_label = QLabel("Выберите задачу:")
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Тестовая задача", "Основная задача"])
        params_layout.addWidget(task_label, 0, 0)
        params_layout.addWidget(self.task_combo, 0, 1)

        # Number of segments
        n_label = QLabel("Количество участков разбиения:")
        self.n_spinbox = QSpinBox()
        self.n_spinbox.setMinimum(2)
        self.n_spinbox.setValue(2)
        params_layout.addWidget(n_label, 1, 0)
        params_layout.addWidget(self.n_spinbox, 1, 1)

        # Buttons
        self.run_button = QPushButton("Запуск")
        self.clear_button = QPushButton("Очистить")
        params_layout.addWidget(self.run_button, 2, 0)
        params_layout.addWidget(self.clear_button, 2, 1)

        # Table to display results
        self.result_table = QTableWidget()
        self.result_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Arrange layouts
        upper_layout = QHBoxLayout()
        params_widget = QWidget()
        params_widget.setLayout(params_layout)
        params_widget.setFixedWidth(300)  # Фиксируем ширину панели с параметрами
        upper_layout.addWidget(params_widget)
        upper_layout.addWidget(self.result_table)

        main_layout.addLayout(upper_layout)
        main_layout.addLayout(self.graph_layout)

        self.setLayout(main_layout)

        # Signals and slots
        self.run_button.clicked.connect(self.run_task)
        self.clear_button.clicked.connect(self.clear_plot)

    def run_task(self):
        try:
            task_name = self.task_combo.currentText()
            n = self.n_spinbox.value()
            step_size = 1 / n
            num_nodes = n + 1

            if task_name == "Тестовая задача":
                x, v, list_u, eps, id_eps, df = self.calculation_handler.perform_test_task(n)
                self.plot_manager.plot_test_task(x, v, list_u)
                self.display_results(df)
                info_message = (
                    "Вариант 1, Точка разрыва = 0.4, Диапазон = [0, 1],\n"
                    f"Количество узлов = {num_nodes},\n"
                    f"Шаг сетки = {step_size}\n"
                    f"Максимальная погрешность = {eps}, на шаге {id_eps}"
                )
            else:
                x, v, v2_interp, eps, id_eps, df = self.calculation_handler.perform_main_task(n)
                self.plot_manager.plot_main_task(x, v, v2_interp)
                self.display_results(df)
                info_message = (
                    "Основная задача,\n"
                    f"Количество узлов = {num_nodes},\n"
                    f"Шаг сетки = {step_size}\n"
                    f"Максимальная погрешность = {eps}, на шаге {id_eps}"
                )

            QMessageBox.information(self, "Результаты", info_message)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def clear_plot(self):
        self.graph_layout.clear()
        self.graph_layout.draw()
        self.result_table.clear()
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(0)

    def display_results(self, df):
        self.result_table.clear()
        self.result_table.setRowCount(len(df))
        self.result_table.setColumnCount(len(df.columns))
        self.result_table.setHorizontalHeaderLabels(df.columns)
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                self.result_table.setItem(i, j, item)
        self.result_table.resizeColumnsToContents()
