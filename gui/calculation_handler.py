# calculation_handler.py

import pandas as pd
from back import MainTask, TestTask, method_balance

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
