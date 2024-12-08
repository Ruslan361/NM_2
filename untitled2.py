
import numpy as np
import math
def generate_abc(n, a, b, task):
    """Generates A, B, C coefficients with minimal ai storage.
    Args:
        n: Number of grid points.
        a: Left boundary.
        b: Right boundary.
        task: Instance of MainTask.
    Returns:
        Tuple containing A, B, C (NumPy arrays).
    """
    h = (b - a) / n
    A = np.zeros(n - 1)
    B = np.zeros(n - 1)
    C = np.zeros(n - 1)
    ai = calc_ai(a + h, h, task)  
    for i in range(n - 1):
        ai_plus_1 = calc_ai(a + (i + 2) * h, h, task)  
        di = calc_di(a + (i + 1) * h, h, task)
        A[i] = ai / (h*h)
        B[i] = ai_plus_1 / (h*h)
        C[i] = A[i] + B[i] + di
        ai = ai_plus_1  
    return A, B, C


def calculate_alpha_beta(A, B, C, phi):
    """Calculates alpha and beta coefficients for the tridiagonal matrix algorithm (TDMA).
    Args:
        A: Sub-diagonal (below main diagonal) of the tridiagonal matrix (NumPy array).
        B: Super-diagonal (above main diagonal) (NumPy array).
        C: Main diagonal (NumPy array).
        phi: Right-hand side vector (NumPy array).
    Returns:
        A tuple containing two NumPy arrays: alpha and beta.
    Raises:
        ValueError: If input arrays have incompatible shapes or zero divisions occur.
    """
    n = len(C)  
    # if len(A) != n - 1 or len(B) != n - 1 or len(phi) != n:
    #     raise ValueError("Input arrays have incompatible shapes.")
    alpha = np.zeros(n)
    beta = np.zeros(n)
    
    alpha[0] = 0 
    beta[0] = 0 
    
    
    for i in range(1, n):  
        denominator = C[i-1] - A[i-1] * alpha[i-1] # C[i] - A[i-1] * alpha[i-1]
        if denominator == 0:
            raise ValueError("Zero division encountered during TDMA.")
        alpha[i] = B[i-1] / denominator
        beta[i] = (phi[i] + A[i-1] * beta[i-1]) / denominator
        
        
    #beta[n - 1] = 1  
    
    
    return alpha, beta

class TestTask:
    KSI = 0.4
    MU1 = 0
    MU2 = 1
    CONST = [[0.060557222866650, -1.060557222866650], [-0.47202455073443716628, -4.33108482358005765177]]
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
        return math.exp(-0.4)
    def u(self, x):
        if x < self.KSI:
            return self.CONST[0][0] * math.exp(x * math.sqrt(2. / 7.)) + self.CONST[0][1] * math.exp(
                -x * math.sqrt(2. / 7.)) + 1.
        else:
            return self.CONST[1][0] * math.exp(x * math.sqrt(0.4)) + self.CONST[1][1] * (
                math.exp(-x * math.sqrt(0.4))) + math.exp(-0.4) / 0.16
class MainTask:
    KSI = 0.4
    MU1 = 0
    MU2 = 1
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
        return math.exp(-x)
def calc_ai(x, step, task):
    xi = x
    xi_1 = x - step
    if task.KSI >= xi:
        return task.k1(xi - step/2.0)
    elif task.KSI <= xi_1:
        return task.k2(xi - step/2.0)
    else:
        return ((1/step)*( ((task.KSI - xi_1) / (task.k1((xi_1 + task.KSI)/2.0))) + ((xi - task.KSI) / (task.k2((task.KSI + xi)/2.0))) ))**(-1)
def calc_di(x, step, task):
    xi_up = x + step/2.0
    xi_down = x - step/2.0
    if task.KSI >= xi_up:
        return task.q1(x)
    elif task.KSI <= xi_down:
        return task.q2(x)
    else:
        return (1/step)*((task.q1((xi_down + task.KSI)/2.0) * (task.KSI - xi_down)) + (task.q2((task.KSI + xi_up)/2.0) * (xi_up - task.KSI)))
def calc_phi_i(x, step, task):
    xi_up = x + step/2.0
    xi_down = x - step/2.0
    if task.KSI >= xi_up:
        return task.f1(x)
    elif task.KSI <= xi_down:
        return task.f2(x)
    else:
        return (1/step)*(task.f1((xi_down + task.KSI)/2.0) * (task.KSI - xi_down) + task.f2((task.KSI + xi_up)/2.0) * (xi_up - task.KSI) )
def calculate_y(alpha, beta):
    """Calculates the y vector using the provided formulas.
    Args:
        alpha: Alpha coefficients (NumPy array).
        beta: Beta coefficients (NumPy array).
    Returns:
       NumPy array containing the calculated y values.
    Raises:
        ValueError: If alpha and beta have different shapes.
    """
    n = len(alpha)+1
    if n-1 != len(beta) :
        raise ValueError("Alpha and beta must have compatible shapes")
    y = np.zeros(n)
    y[n-1] = 1  
    for i in range(n - 2, -1, -1): 
        y[i] = alpha[i] * y[i + 1] + beta[i]
    return y

import numpy as np
import math
import matplotlib.pyplot as plt
def solve_test_task(n, a, b, task):
    #x = np.linspace(a, b, n)
    A, B, C = generate_abc(n, a, b, task)
    phi = np.array([calc_phi_i(a + (i + 1) * (b - a) / n, (b - a) / n, task) for i in range(0, n-1)])
    alpha, beta = calculate_alpha_beta(A, B, C, phi)
    y = calculate_y(alpha, beta)
    return y
n = 3
y = solve_test_task(n, 0, 1, TestTask())
print(y)
u = np.array([TestTask().u(0 + (i + 1) * 1 / n) for i in range(n)])
print(np.max(np.abs(y - u)))
def plot_test_task(x, v, list_u):
    plt.figure()
    plt.plot(x, v, label='Численное решение v(x)')
    plt.plot(x, list_u, label='Точное решение u(x)')
    plt.title("Тестовая задача")
    plt.xlabel("x")
    plt.ylabel("u(x), v(x)")
    plt.legend()
    plt.show()
plot_test_task([0 + (i + 1) * 1 / 10 for i in range(n)], y, u)