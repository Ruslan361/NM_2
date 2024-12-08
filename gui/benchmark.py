import time
import numpy as np
from back.resolve import method_balance, TestTask as TestTaskResolve
from back.numpy_solver import solve_task, TestTask as TestTaskNumba

def performance_test():
    n_values = [1, 100, 1000, 5000, 10000, 100000, 500000]
    task_resolve = TestTaskResolve()
    task_numba = TestTaskNumba()

    print("n\tmethod_balance (s)\tsolve_task (s)")
    for n in n_values:
        # Time method_balance from resolve.py
        start_time = time.time()
        x_mb, y_mb = method_balance(n, task_resolve)
        time_mb = time.time() - start_time

        # Time solve_task from numpy_solver.py
        start_time = time.time()
        x_st, y_st = solve_task(n, task_numba)
        time_st = time.time() - start_time

        print(f"{n}\t{time_mb:.6f}\t\t{time_st:.6f}")
        if not (np.max(np.abs(x_mb - x_st)) < 1e-6 and np.max(np.abs(y_mb - y_st)) < 1e-6):
            print(f"Mismatch for n={n}")
        else:
            print(f"Results match for n={n}")

if __name__ == '__main__':
    performance_test()