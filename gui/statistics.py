from back.numpy_solver import TestTask as TestTaskNumba, solve_task, MainTask as MainTaskNumba
import numpy as np
import pandas as pd
import time
def generate_n(max_n):
    multiplicators = [1, 2, 5, 8]
    i = 0
    j = 1
    while multiplicators[i] * j<=max_n:
        while i < 4:
            yield multiplicators[i] * j
            i += 1
        j *=10
        i = 0

generator = generate_n(10000000)
task = TestTaskNumba()
main_task = MainTaskNumba()
def calculate_statistics(generator, task, main_task):
    df = pd.DataFrame(columns=['n', 'Тестовая задача |u-v|', 'Основная задача |vn-v2n|', 'lgn', '-lg |u-v|', '-lg |vn-v2n|', 'solve_task_time', 'v_time', 'v2_time'])
    rows = []
    for i in generator:
        start_time = time.time()
        x, v = solve_task(i, task)
        solve_task_time = time.time() - start_time
    
        u = np.array([task.u(xi) for xi in x])
        module = np.max(np.abs(v - u))
    
        start_time = time.time()
        v = solve_task(i, main_task)[1]
        v_time = time.time() - start_time
    
        start_time = time.time()
        v2 = solve_task(2 * i, main_task)[1][::2]
        v2_time = time.time() - start_time
    
        difference = np.max(np.abs(v - v2))
        rows.append({'n': i, 'Тестовая задача |u-v|': module, 'Основная задача |vn-v2n|': difference, 'lgn': np.log10(i), '-lg |u-v|': -np.log10(module), '-lg |vn-v2n|': -np.log10(difference), 'solve_task_time': solve_task_time, 'v_time': v_time, 'v2_time': v2_time})
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    print(df)

calculate_statistics(generator, task, main_task)
