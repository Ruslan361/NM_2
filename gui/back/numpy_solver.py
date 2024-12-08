import numpy as np
from numba import njit, types
from numba.experimental import jitclass
from numba import njit, float64, types

# Спецификация для класса
spec = [
    ('KSI', float64),
    ('MU1', float64),
    ('MU2', float64),
    ('CONST', types.Array(float64, 2, 'C')),  # Массив: 2D с C-order
]
@jitclass(spec)
class TestTask:
    def __init__(self):
        self.KSI = 0.4
        self.MU1 = 0
        self.MU2 = 1
        self.CONST = np.array([
            [0.060557222866650, -1.060557222866650],
            [-0.47202455073443716628, -4.33108482358005765177]
        ], dtype=np.float64)

    def k1(self, x):
        return 1.4

    def k2(self, x):
        return 0.4

    def q1(self, x):
        return 0.4

    def q2(self, x):
        return 0.16

    def f1(self, x):
        return 0.4

    def f2(self, x):
        return np.exp(-0.4)

    def u(self, x):
        if x < self.KSI:
            return self.CONST[0, 0] * np.exp(x * np.sqrt(2. / 7.)) + self.CONST[0, 1] * np.exp(
                -x * np.sqrt(2. / 7.)) + 1.
        else:
            return self.CONST[1, 0] * np.exp(x * np.sqrt(0.4)) + self.CONST[1, 1] * (
                np.exp(-x * np.sqrt(0.4))) + np.exp(-0.4) / 0.16

spec = [
    ('KSI', float64),
    ('MU1', float64),
    ('MU2', float64),
]
@jitclass(spec)
class MainTask:
    def __init__(self):
        self.KSI = 0.4
        self.MU1 = 0.0
        self.MU2 = 1.0

    def k1(self, x):
        return (x + 1)

    def k2(self, x):
        return x

    def q1(self, x):
        return x

    def q2(self, x):
        return x * x

    def f1(self, x):
        return x

    def f2(self, x):
        return np.exp(-x)

# расчет коэфициентов в уравнении
@njit
def calc_ai(x, step, task):
    xi = x
    xi_1 = x - step

    if task.KSI >= xi:
        return task.k1(xi - step/2.0)
    elif task.KSI <= xi_1:
        return task.k2(xi - step/2.0)
    else:
        return ((1/step)*( ((task.KSI - xi_1) / (task.k1((xi_1 + task.KSI)/2.0))) + ((xi - task.KSI) / (task.k2((task.KSI + xi)/2.0))) ))**(-1)
@njit
def calc_di(x, step, task):
    xi_up = x + step/2.0
    xi_down = x - step/2.0

    if task.KSI >= xi_up:
        return task.q1(x)
    elif task.KSI <= xi_down:
        return task.q2(x)
    else:
        return (1/step)*((task.q1((xi_down + task.KSI)/2.0) * (task.KSI - xi_down)) + (task.q2((task.KSI + xi_up)/2.0) * (xi_up - task.KSI)))
@njit
def calc_phi_i(x, step, task):
    xi_up = x + step/2.0
    xi_down = x - step/2.0

    if task.KSI >= xi_up:
        return task.f1(x)
    elif task.KSI <= xi_down:
        return task.f2(x)
    else:
        return (1/step)*(task.f1((xi_down + task.KSI)/2.0) * (task.KSI - xi_down) + task.f2((task.KSI + xi_up)/2.0) * (xi_up - task.KSI) )
@njit
def calc_coefficients(n, task):
    """Рассчитывает коэффициенты для метода прогонки."""
    h = 1.0 / n
    x = np.linspace(0, 1, n + 1)

    # Выделяем массивы для коэффициентов
    A = np.zeros(n + 1)
    B = np.zeros(n + 1)
    C = np.zeros(n + 1)
    phi = np.zeros(n + 1)

    # Граничные условия
    C[0], phi[0] = 1, task.MU1
    C[-1], phi[-1] = 1, task.MU2

    # Заполняем внутренние точки
    for i in range(1, n):
        xi = x[i]
        ai = calc_ai(xi, h, task)
        ai_next = calc_ai(xi + h, h, task)
        di = calc_di(xi, h, task)
        phi[i] = -calc_phi_i(xi, h, task)

        A[i] = ai / (h * h)
        B[i] = ai_next / (h * h)
        C[i] = -(ai + ai_next) / (h * h) - di

    return A, B, C, phi, x

@njit
def thomas_algorithm(A, B, C, phi):
    """Решает СЛАУ с трехдиагональной матрицей методом прогонки."""
    n = len(phi)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    y = np.zeros(n)

    # Прямой ход
    alpha[1] = -B[0] / C[0]
    beta[1] = phi[0] / C[0]

    for i in range(1, n - 1):
        denominator = C[i] + A[i] * alpha[i]
        alpha[i + 1] = -B[i] / denominator
        beta[i + 1] = (phi[i] - A[i] * beta[i]) / denominator

    # Обратный ход
    y[-1] = (phi[-1] - A[-1] * beta[-1]) / (C[-1] + A[-1] * alpha[-1])
    for i in range(n - 2, -1, -1):
        y[i] = alpha[i + 1] * y[i + 1] + beta[i + 1]

    return y

@njit
def solve_task(n, task):
    """Основная функция для решения задачи."""
    A, B, C, phi, x = calc_coefficients(n, task)
    y = thomas_algorithm(A, B, C, phi)
    return x, y