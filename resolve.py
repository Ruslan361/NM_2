import math
from new_progon import Thomas_Shelby_algorythm


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

# расчет коэфициентов в уравнении
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
    
# построение 3 диагональной матрицы
def method_balance(n, task):
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
   