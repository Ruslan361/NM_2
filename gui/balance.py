# balance.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QSpinBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QGridLayout, QSizePolicy, QFileDialog, QCheckBox
)
from calculation_handler import CalculationHandler
from plot_manager import PlotManager
from custom_layouts import GraphLayout
import os

class BalanceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NM_2")

        self.calculation_handler = CalculationHandler()
        self.graph_layout = GraphLayout()
        self.plot_manager = PlotManager(self.graph_layout)

        self.info_message = ""

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Parameters Input Section
        params_layout = QGridLayout()
        params_layout.setSpacing(10)

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
        self.n_spinbox.setMaximum(1000000)
        self.n_spinbox.setValue(1000)
        params_layout.addWidget(n_label, 1, 0)
        params_layout.addWidget(self.n_spinbox, 1, 1)

        # Checkbox to use NumPy solver
        self.use_numpy_checkbox = QCheckBox("Использовать NumPy Solver")
        params_layout.addWidget(self.use_numpy_checkbox, 2, 0, 1, 2)

        # Buttons
        self.run_button = QPushButton("Запуск")
        self.clear_button = QPushButton("Очистить")
        self.save_button = QPushButton("Сохранить результаты")
        self.load_button = QPushButton("Загрузить результаты")
        self.display_button = QPushButton("Отобразить результаты")
        self.help_button = QPushButton("Справка")

        # Reduce size of buttons and area
        buttons = [
            self.run_button, self.clear_button,
            self.save_button, self.load_button,
            self.display_button, self.help_button
        ]
        for button in buttons:
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setMinimumHeight(30)
            button.setMaximumHeight(30)

        # Organize buttons compactly
        buttons_layout = QGridLayout()
        buttons_layout.setSpacing(5)
        buttons_layout.addWidget(self.run_button, 0, 0)
        buttons_layout.addWidget(self.clear_button, 0, 1)
        buttons_layout.addWidget(self.save_button, 1, 0)
        buttons_layout.addWidget(self.load_button, 1, 1)
        buttons_layout.addWidget(self.display_button, 2, 0)
        buttons_layout.addWidget(self.help_button, 2, 1)

        # Add buttons layout to params_layout
        params_layout.addLayout(buttons_layout, 3, 0, 1, 2)

        # Table to display results
        self.result_table = QTableWidget()
        self.result_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.result_table.setMinimumHeight(200)

        # Arrange layouts
        upper_layout = QHBoxLayout()
        params_widget = QWidget()
        params_widget.setLayout(params_layout)
        params_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        params_widget.setFixedWidth(300)
        upper_layout.addWidget(params_widget)
        upper_layout.addLayout(self.graph_layout)

        main_layout.addLayout(upper_layout)
        main_layout.addWidget(self.result_table)

        self.setLayout(main_layout)

        # Signals and slots
        self.run_button.clicked.connect(self.run_task)
        self.clear_button.clicked.connect(self.clear_plot)
        self.save_button.clicked.connect(self.save_results)
        self.load_button.clicked.connect(self.load_results)
        self.display_button.clicked.connect(self.display_results_button)
        self.help_button.clicked.connect(self.show_help)

    def run_task(self):
        try:
            task_name = self.task_combo.currentText()
            n = self.n_spinbox.value()
            step_size = 1 / n
            num_nodes = n + 1

            use_numpy = self.use_numpy_checkbox.isChecked()

            if task_name == "Тестовая задача":
                if use_numpy:
                    x, v, list_u, eps, id_eps, df = self.calculation_handler.perform_test_task_numpy(n)
                else:
                    x, v, list_u, eps, id_eps, df = self.calculation_handler.perform_test_task(n)
                self.plot_manager.plot_test_task(x, v, list_u)
                self.display_results(df)
                self.info_message = (
                    "Тестовая задача:\n"
                    f"Количество узлов = {num_nodes},\n"
                    f"Шаг сетки = {step_size}\n"
                    f"Максимальная погрешность = {eps}, на шаге {id_eps+1}\n"
                    f"При x = {x[id_eps]}"
                )
            else:
                if use_numpy:
                    x, v, v2_interp, eps, id_eps, df = self.calculation_handler.perform_main_task_numpy(n)
                else:
                    x, v, v2_interp, eps, id_eps, df = self.calculation_handler.perform_main_task(n)
                self.plot_manager.plot_main_task(x, v, v2_interp)
                self.display_results(df)
                self.info_message = (
                    "Основная задача:\n"
                    f"Количество узлов = {num_nodes},\n"
                    f"Шаг сетки = {step_size}\n"
                    f"Максимальная разность = {eps}, на шаге {id_eps+1}\n"
                    f"При x = {x[id_eps]}"
                )

            QMessageBox.information(self, "Результаты", self.info_message)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def clear_plot(self):
        self.graph_layout.clear()
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

    def display_results_button(self):
        """Отображает текущее сообщение о результатах."""
        if self.info_message:
            QMessageBox.information(self, "Результаты", self.info_message)
        else:
            QMessageBox.information(self, "Информация", "Нет результатов для отображения.")

    def save_results(self):
        try:
            if self.result_table.rowCount() == 0 or self.result_table.columnCount() == 0:
                QMessageBox.warning(self, "Предупреждение", "Нет данных для сохранения.")
                return

            task_name = self.task_combo.currentText().replace(" ", "_")
            n = self.n_spinbox.value()
            filename = f"results_n{n}_{task_name}.csv"

            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить результаты",
                filename,
                "CSV Files (*.csv)",
                options=options
            )
            if file_path:
                # Получение данных из таблицы
                rows = self.result_table.rowCount()
                cols = self.result_table.columnCount()
                headers = [self.result_table.horizontalHeaderItem(i).text() for i in range(cols)]
                data = []
                for row in range(rows):
                    row_data = []
                    for col in range(cols):
                        item = self.result_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    data.append(row_data)

                # Сохранение в CSV с помощью pandas
                import pandas as pd
                df = pd.DataFrame(data, columns=headers)
                df.to_csv(file_path, index=False)

                QMessageBox.information(self, "Успех", f"Результаты успешно сохранены в {file_path}.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка при сохранении", str(e))

    def load_results(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Загрузить результаты",
                "",
                "CSV Files (*.csv)",
                options=options
            )
            if file_path:
                import pandas as pd
                df = pd.read_csv(file_path)

                # Отображение данных в таблице
                self.display_results(df)

                # Определение задачи и n из имени файла
                base_name = os.path.basename(file_path)
                parts = base_name.split('_')
                n_part = [part for part in parts if part.startswith('n')]
                task_part = [part for part in parts if "Тестовая" in part or "Основная" in part]

                if n_part:
                    try:
                        n = int(n_part[0][1:])
                        self.n_spinbox.setValue(n)
                    except ValueError:
                        pass
                if task_part:
                    task_name = task_part[0].replace('_', ' ')
                    index = self.task_combo.findText(task_name)
                    if index != -1:
                        self.task_combo.setCurrentIndex(index)

                # Обновление графика в зависимости от задачи
                task_name = self.task_combo.currentText()
                if task_name == "Тестовая задача":
                    x, v, list_u, eps, id_eps, _ = self.calculation_handler.perform_test_task(n)
                    self.plot_manager.plot_test_task(x, v, list_u)
                    self.info_message = (
                        "Тестовая задача:\n"
                        f"Количество узлов = {n+1},\n"
                        f"Шаг сетки = {1/n}\n"
                        f"Максимальная погрешность = {eps}, на шаге {id_eps+1}\n"
                        f"При x = {x[id_eps]}"
                    )
                else:
                    x, v, v2_interp, eps, id_eps, _ = self.calculation_handler.perform_main_task(n)
                    self.plot_manager.plot_main_task(x, v, v2_interp)
                    self.info_message = (
                        "Основная задача:\n"
                        f"Количество узлов = {n+1},\n"
                        f"Шаг сетки = {1/n}\n"
                        f"Максимальная разность = {eps}, на шаге {id_eps+1}\n"
                        f"При x = {x[id_eps]}"
                    )

                QMessageBox.information(self, "Успех", f"Результаты успешно загружены из {file_path}.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка при загрузке", str(e))

    def show_help(self):
        help_text = (
            "Инструкция по использованию приложения:\n\n"
            "1. Выберите тип задачи (Тестовая или Основная).\n"
            "2. Укажите количество участков разбиения (n). Максимальное значение n: 1 000 000.\n"
            "3. Нажмите кнопку 'Запуск' для выполнения вычислений.\n"
            "4. После выполнения задачи вы можете сохранить результаты в CSV-файл или загрузить ранее сохраненные результаты.\n"
            "5. Чтобы очистить графики и таблицу результатов, нажмите кнопку 'Очистить'.\n"
            "6. Для повторного отображения результатов нажмите кнопку 'Отобразить результаты'.\n\n"
            "Форма сохранения результатов включает значение n и тип задачи в названии файла."
        )
        QMessageBox.information(self, "Справка", help_text)
