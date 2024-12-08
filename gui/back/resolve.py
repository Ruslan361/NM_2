import math
from back.new_progon import Thomas_Shelby_algorythm
from typing import List, Tuple


class TestTask:
    """
    Класс TestTask представляет тестовую задачу для решения дифференциального уравнения.
    
    Args:
        KSI (float): Параметр KSI.
        MU1 (float): Граничное условие слева.
        MU2 (float): Граничное условие справа.
        CONST (List[List[float]]): Константы для вычисления точного решения.
    
    Methods:
        k1(x: float) -> float: Возвращает значение коэффициента k1.
        k2(x: float) -> float: Возвращает значение коэффициента k2.
        q1(x: float) -> float: Возвращает значение коэффициента q1.
        q2(x: float) -> float: Возвращает значение коэффициента q2.
        f1(x: float) -> float: Возвращает значение коэффициента f1.
        f2(x: float) -> float: Возвращает значение коэффициента f2.
        u(x: float) -> float: Возвращает точное решение u в точке x.
    """
    KSI = 0.4
    MU1 = 0
    MU2 = 1
    CONST = [[0.060557222866650, -1.060557222866650], [-0.47202455073443716628, -4.33108482358005765177]]

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
        return math.exp(-0.4)

    def u(self, x: float) -> float:
        """
        Возвращает точное решение u в точке x.
        
        Args:
            x (float): Точка, в которой вычисляется решение.
        
        Returns:
            float: Значение точного решения u в точке x.
        """
        if x < self.KSI:
            return self.CONST[0][0] * math.exp(x * math.sqrt(2. / 7.)) + self.CONST[0][1] * math.exp(
                -x * math.sqrt(2. / 7.)) + 1.
        else:
            return self.CONST[1][0] * math.exp(x * math.sqrt(0.4)) + self.CONST[1][1] * (
                math.exp(-x * math.sqrt(0.4))) + math.exp(-0.4) / 0.16


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
    KSI = 0.4
    MU1 = 0
    MU2 = 1

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

    def f2(self, x: float) -> float:
        """
        Возвращает значение коэффициента f2.
        
        Args:
            x (float): Точка, в которой вычисляется коэффициент.
        
        Returns:
            float: Значение коэффициента f2.
        """
        return math.exp(-x)

# расчет коэфициентов в уравнении
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

# построение 3 диагональной матрицы
def method_balance(n: int, task) -> Tuple[List[float], List[float]]:
    """
    Метод для решения дифференциального уравнения методом баланса.
    Описание функции. Вычисляет координаты и значения функции решения дифференциального уравнения с заданными параметрами.
    Args:
        n (int): количество разбиений (отвечает за число узлов сетки).
        task: объект задачи, содержащий функции коэффициентов и граничные условия.
    Returns:
        Tuple[List[float], List[float]]: возвращает кортеж из списка координат x и соответствующих значений v.
    """
    matrix_A = []
    vector_b = []

    step = 1/n
    n += 1
    xi = 0

    matrix_A.append([1, 0, 0])
    vector_b.append(task.MU1)

    for i in range(1, n-1):
        xi = i*step
        # print(calc_di(xi, step, sel))
        matrix_A.append([calc_ai(xi, step, task)/(step*step),
                          -(calc_ai(xi, step, task)+calc_ai(xi+step, step, task))/(step*step)-calc_di(xi, step, task),
                           calc_ai(xi+step, step, task)/(step*step)])
        vector_b.append(-calc_phi_i(xi, step, task))

    matrix_A.append([0, 0, 1])
    vector_b.append(task.MU2)

    # for i in range(len(matrix_A)):
    #     print(str(matrix_A[i]))
    # print('\n')
    # print(vector_b)

    v = Thomas_Shelby_algorythm(matrix_A, vector_b)
    x = [i*step for i in range(n)]
    # vec_u = [u(i) for i in x]

    # print('\n')
    # print(v)
    # print(x)
    # print(vec_u)

    return x, v

if __name__ == '__main__':
    method_balance(2, 1)
