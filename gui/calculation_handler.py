# calculation_handler.py
import numpy as np
import pandas as pd
from back import MainTask, TestTask, method_balance
import back.numpy_solver as solver

class CalculationHandler:
    def __init__(self):
        pass

    def perform_test_task(self, n):
        task = TestTask()
        x, v = method_balance(n, task)
        list_u = [task.u(xi) for xi in x]
        eps_list = [abs(ui - vi) for ui, vi in zip(list_u, v)]
        eps = max(eps_list)
        id_eps = eps_list.index(eps)
        df = pd.DataFrame({
            'x': x,
            'v': v,
            'u': list_u,
            '|u - v|': eps_list
        })
        return x, v, list_u, eps, id_eps, df

    def perform_main_task(self, n):
        task = MainTask()
        x, v = method_balance(n, task)
        x2, v2 = method_balance(2 * n, task)
        v2_interp = [v2[i] for i in range(0, len(v2), 2)]
        eps_list = [abs(v2i - vi) for v2i, vi in zip(v2_interp, v)]
        eps = max(eps_list)
        id_eps = eps_list.index(eps)
        df = pd.DataFrame({
            'x': x,
            'v_n': v,
            'v_2n': v2_interp,
            '|v_2n - v_n|': eps_list
        })
        return x, v, v2_interp, eps, id_eps, df

    def perform_test_task_numpy(self, n):
        task = solver.TestTask()
        x, v = solver.solve_task(n, task)
        # x = np.array(x)
        # v = np.array(v)
        u = np.vectorize(task.u)(x)
        eps_list = np.abs(u - v)
        eps = np.max(eps_list)
        id_eps = np.argmax(eps_list)
        df = pd.DataFrame({
            'x': x,
            'v': v,
            'u': u,
            '|u - v|': eps_list
        })
        return x, v, u, float(eps), int(id_eps), df

    def perform_main_task_numpy(self, n):
        task = solver.MainTask()
        x, v = solver.solve_task(n, task)
        x2, v2 = solver.solve_task(2 * n, task)
        # x = np.array(x)
        # v = np.array(v)
        # v2 = np.array(v2)
        v2_interp = v2[::2]
        eps_list = np.abs(v2_interp - v)
        eps = np.max(eps_list)
        id_eps = np.argmax(eps_list)
        df = pd.DataFrame({
            'x': x,
            'v_n': v,
            'v_2n': v2_interp,
            '|v_2n - v_n|': eps_list
        })
        return x.tolist(), v.tolist(), v2_interp.tolist(), float(eps), int(id_eps), df