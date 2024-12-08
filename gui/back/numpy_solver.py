import numpy as np
from numba import njit, types
from numba.experimental import jitclass
from numba import njit, float64, types
from typing import List, Tuple

# Спецификация для класса
spec = [
    ('KSI', float64),
    ('MU1', float64),
    ('MU2', float64),
    ('CONST', types.Array(float64, 2, 'C')),  # Массив: 2D с C-order
]
@jitclass(spec)
class TestTask:
    """
    Класс MainTask представляет основную задачу для решения дифференциального уравнения.
    
    Args:
        KSI (float): Параметр KSI.
        MU1 (float): Граничное условие слева.
        MU2 (float): Граничное условие справа.
    
    Methods:
        k1(x: float) -> float: Возвращает значение коэффициента k1.
        k2(x: float) -> float: Возвращает значение коэффициента k2.
        q1(x: float) -> float: Возвращает значение коэффициента q1.
        q2(x: float) -> float: Возвращает значение коэффициента q2.
        f1(x: float) -> float: Возвращает значение коэффициента f1.
        f2(x: float) -> float: Возвращает значение коэффициента f2.
    """
    def __init__(self):
        """
        Инициализирует экземпляр класса MainTask.
        """
        self.KSI = 0.4
        self.MU1 = 0
        self.MU2 = 1
        self.CONST = np.array([
            [0.060557222866650, -1.060557222866650],
            [-0.47202455073443716628, -4.33108482358005765177]
        ], dtype=np.float64)

    def k1(self, x: float) -> float:
        """
        Возвращает значение коэффициента k1.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента k1.
        """
        return 1.4

    def k2(self, x: float) -> float:
        """
        Возвращает значение коэффициента k2.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента k2.
        """
        return 0.4

    def q1(self, x: float) -> float:
        """
        Возвращает значение коэффициента q1.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента q1.
        """
        return 0.4

    def q2(self, x: float) -> float:
        """
        Возвращает значение коэффициента q2.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента q2.
        """
        return 0.16

    def f1(self, x: float) -> float:
        """
        Возвращает значение коэффициента f1.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента f1.
        """
        return 0.4

    def f2(self, x: float) -> float:
        """
        Возвращает значение коэффициента f2.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента f2.
        """
        return np.exp(-0.4)

    def u(self, x: float) -> float:
        """
        Возвращает точное решение u в точке x.
        
        Args:
            x (float): Точка, в которой вычисляется решение.
        
        Returns:
            float: Значение точного решения u в точке x.
        """
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
    """
    Класс MainTask представляет основную задачу для решения дифференциального уравнения.
    
    Args:
        KSI (float): Параметр KSI.
        MU1 (float): Граничное условие слева.
        MU2 (float): Граничное условие справа.
    
    Methods:
        k1(x: float) -> float: Возвращает значение коэффициента k1.
        k2(x: float) -> float: Возвращает значение коэффициента k2.
        q1(x: float) -> float: Возвращает значение коэффициента q1.
        q2(x: float) -> float: Возвращает значение коэффициента q2.
        f1(x: float) -> float: Возвращает значение коэффициента f1.
        f2(x: float) -> float: Возвращает значение коэффициента f2.
    """
    def __init__(self):
        """
        Инициализирует экземпляр класса MainTask.
        """
        self.KSI = 0.4
        self.MU1 = 0.0
        self.MU2 = 1.0

    def k1(self, x: float) -> float:
        """
        Возвращает значение коэффициента k1.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента k1.
        """
        return (x + 1)

    def k2(self, x: float) -> float:
        """
        Возвращает значение коэффициента k2.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента k2.
        """
        return x

    def q1(self, x: float) -> float:
        """
        Возвращает значение коэффициента q1.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента q1.
        """
        return x

    def q2(self, x: float) -> float:
        """
        Возвращает значение коэффициента q2.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента q2.
        """
        return x * x

    def f1(self, x: float) -> float:
        """
        Возвращает значение коэффициента f1.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента f1.
        """
        return x

    def f2(self, x):
        """
        Возвращает значение коэффициента f2.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента f2.
        """
        return np.exp(-x)

# расчет коэфициентов в уравнении
@njit(nopython=True, cache=True, inline='always', nogil=True)
def calc_ai(x: float, step: float, task: TestTask) -> float:
    """
    Вычисляет коэффициент ai.
    
    Args:
        x (float): Точка, в которой вычисляется коэффициент.
        step (float): Шаг разбиения.
        task (TestTask): Экземпляр задачи.
    
    Returns:
        float: Значение коэффициента ai.
    """
    xi = x
    xi_1 = x - step

    if task.KSI >= xi:
        return task.k1(xi - step/2.0)
    elif task.KSI <= xi_1:
        return task.k2(xi - step/2.0)
    else:
        return ((1/step)*( ((task.KSI - xi_1) / (task.k1((xi_1 + task.KSI)/2.0))) + ((xi - task.KSI) / (task.k2((task.KSI + xi)/2.0))) ))**(-1)

@njit(nopython=True, cache=True, inline='always', nogil=True)
def calc_di(x: float, step: float, task: TestTask) -> float:
    """
    Вычисляет коэффициент di.
    
    Args:
        x (float): Точка, в которой вычисляется коэффициент.
        step (float): Шаг разбиения.
        task (TestTask): Экземпляр задачи.
    
    Returns:
        float: Значение коэффициента di.
    """
    xi_up = x + step/2.0
    xi_down = x - step/2.0

    if task.KSI >= xi_up:
        return task.q1(x)
    elif task.KSI <= xi_down:
        return task.q2(x)
    else:
        return (1/step)*((task.q1((xi_down + task.KSI)/2.0) * (task.KSI - xi_down)) + (task.q2((task.KSI + xi_up)/2.0) * (xi_up - task.KSI)))

@njit(nopython=True, cache=True, inline='always', nogil=True)
def calc_phi_i(x: float, step: float, task: TestTask) -> float:
    """
    Вычисляет коэффициент phi_i.
    
    Args:
        x (float): Точка, в которой вычисляется коэффициент.
        step (float): Шаг разбиения.
        task (TestTask): Экземпляр задачи.
    
    Returns:
        float: Значение коэффициента phi_i.
    """
    xi_up = x + step/2.0
    xi_down = x - step/2.0

    if task.KSI >= xi_up:
        return task.f1(x)
    elif task.KSI <= xi_down:
        return task.f2(x)
    else:
        return (1/step)*(task.f1((xi_down + task.KSI)/2.0) * (task.KSI - xi_down) + task.f2((task.KSI + xi_up)/2.0) * (xi_up - task.KSI) )

@njit(nopython=True, cache=True, inline='always', nogil=True)
def calc_coefficients(n: int, task: TestTask) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Рассчитывает коэффициенты для метода прогонки.
    
    Args:
        n (int): Количество разбиений.
        task (TestTask): Экземпляр задачи.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Кортеж массивов коэффициентов A, B, C, phi и координат x.
    """
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

@njit(nopython=True, cache=True, inline='always', nogil=True)
def thomas_algorithm(A: np.ndarray, B: np.ndarray, C: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Решает СЛАУ с трехдиагональной матрицей методом прогонки.
    
    Args:
        A (np.ndarray): Поддиагональ.
        B (np.ndarray): Наддиагональ.
        C (np.ndarray): Главная диагональ.
        phi (np.ndarray): Вектор правых частей.
    
    Returns:
        np.ndarray: Вектор решений y.
    """
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

@njit(nopython=True, cache=True, inline='always', nogil=True)
def solve_task(n: int, task: TestTask) -> Tuple[np.ndarray, np.ndarray]:
    """
    Основная функция для решения задачи.
    
    Args:
        n (int): Количество разбиений.
        task (TestTask): Экземпляр задачи.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Кортеж массивов координат x и решений y.
    """
    A, B, C, phi, x = calc_coefficients(n, task)
    y = thomas_algorithm(A, B, C, phi)
    return x, y