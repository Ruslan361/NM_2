#define CONST1_1 -4.55134
#define CONST1_2 3.44568
#define CONST2_1 0.943942
#define CONST2_2 5.0601

#define KSI 0.4
#define MU1 0
#define MU2 1

#define TEST_K1 1.4
#define TEST_K2 0.4
#define TEST_Q1 0.4
#define TEST_Q2 0.16
#define TEST_F1 0.4
#define TEST_F2 std::exp(-0.4)

#include <iostream>
#include <cmath>
#include <vector>
#include <functional>

double k1(double x)
{
  return 1 / (x + 1);
}

double k2(double x)
{
  return 1 / x;
}

double q1(double x)
{
  return x;
}

double q2(double x)
{
  return x * x;
}

double f1(double x)
{
  return x;
}

double f2(double x)
{
  return std::exp(-x);
}

double u1(double x)
{
  return CONST1_1 * std::exp(x * std::sqrt(2 / 7)) + CONST1_2 * std::exp(-x * std::sqrt(2 / 7)) + 1;
}

double u2(double x)
{
  return CONST2_1 * std::exp(x * std::sqrt(2 / 5)) + CONST2_2 * std::exp(-x * std::sqrt(2 / 5)) + std::exp(-0.4) / 0.16;
}

double Integral(double a, double b, std::function<double(double)> f, int numRectangles = 100)
{
  double result = 0.0;
  double dx = (b - a) / numRectangles;

  for (int i = 0; i < numRectangles; i++)
  {
    double x = a + i * dx;
    result += f(x) * dx;
  }
  return result;
}

/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/

double test_ai(double x, double step)
{
  double ai;
  double xi = x;
  double xi_1 = x - step;

  if (KSI >= xi)
  {
    ai = (1 / TEST_K1) * (xi - xi_1);
  }
  else if (KSI < xi && KSI > xi_1)
  {
    ai = (1 / TEST_K1) * (KSI - xi_1) + (1 / TEST_K2) * (xi - KSI);
  }
  else
  {
    ai = (1 / TEST_K2) * (xi - xi_1);
  }

  ai *= 1 / step;
  ai = 1 / ai;

  return ai;
}

double test_di(double x, double step)
{
  double di;

  double xi05 = x + step / 2;
  double xi_05 = x - step / 2;

  if (KSI >= xi05)
  {
    di = TEST_Q1 * (xi05 - xi_05);
  }
  else if (KSI < xi05 && KSI > xi_05)
  {
    di = TEST_Q1 * (KSI - xi_05) + TEST_Q2 * (xi05 - KSI);
  }
  else
  {
    di = TEST_Q2 * (xi05 - xi_05);
  }

  di *= 1 / step;

  return di;
}

double test_phi_i(double x, double step)
{
  double phi;

  double xi05 = x + step / 2;
  double xi_05 = x - step / 2;

  if (KSI >= xi05)
  {
    phi = TEST_F1 * (xi05 - xi_05);
  }
  else if (KSI < xi05 && KSI > xi_05)
  {
    phi = TEST_F1 * (KSI - xi_05) + TEST_F2 * (xi05 - KSI);
  }
  else
  {
    phi = TEST_F2 * (xi05 - xi_05);
  }

  phi *= 1 / step;

  return phi;
}

/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/

double ai(double x, double step)
{
  double ai;
  double xi = x;
  double xi_1 = x - step;

  if (KSI >= xi)
  {
    ai = Integral(xi_1, xi, k1);
  }
  else if (KSI < xi && KSI > xi_1)
  {
    ai = Integral(xi_1, KSI, k1) + Integral(KSI, xi, k2);
  }
  else
  {
    ai = Integral(xi_1, xi, k2);
  }

  ai *= 1 / step;
  ai = 1 / ai;

  return ai;
}

double di(double x, double step)
{
  double di;

  double xi05 = x + step / 2;
  double xi_05 = x - step / 2;

  if (KSI >= xi05)
  {
    di = Integral(xi_05, xi05, q1);
  }
  else if (KSI < xi05 && KSI > xi_05)
  {
    di = Integral(xi_05, KSI, q1) + Integral(KSI, xi05, q2);
  }
  else
  {
    di = Integral(xi_05, xi05, q2);
  }

  di *= 1 / step;

  return di;
}

double phi_i(double x, double step)
{
  double phi;

  double xi05 = x + step / 2;
  double xi_05 = x - step / 2;

  if (KSI >= xi05)
  {
    phi = Integral(xi_05, xi05, f1);
  }
  else if (KSI < xi05 && KSI > xi_05)
  {
    phi = Integral(xi_05, KSI, f1) + Integral(KSI, xi05, f2);
  }
  else
  {
    phi = Integral(xi_05, xi05, f2);
  }

  phi *= 1 / step;

  return phi;
}

/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/

void method_balance(int n, int choose = 1) // choose = 1 - test, choose = 0 - main problem
{
  double step = 1 / double(n);

  std::vector<std::vector<double>> matrix_with_3_diagonal(n, std::vector<double>(3, 0));

  std::vector<double> b(n);

  matrix_with_3_diagonal[0][0] = MU1;
  matrix_with_3_diagonal[0][1] = 0;
  matrix_with_3_diagonal[0][2] = 0;

  if (choose == 1)
  {
    for (int i = 1; i < n - 1; i++)
    {
      double xi = i * step;
      matrix_with_3_diagonal[i][0] = test_ai(xi, step) / (step * step);                                                                     // коэф при v_i-1
      matrix_with_3_diagonal[i][1] = -(test_ai(xi, step) / (step * step)) - (test_ai(xi + step, step) / (step * step)) - test_di(xi, step); // коэф при v_i
      matrix_with_3_diagonal[i][2] = test_ai(xi + step, step) / (step * step);                                                              // коэф при v_i+1
      b[i] = test_phi_i(xi, step);
    }
  }
  else
  {
    // инициализация 3 диагональной матрицы
    for (int i = 1; i < n - 1; i++)
    {
      double xi = i * step;
      matrix_with_3_diagonal[i][0] = ai(xi, step) / (step * step);                                                           // коэф при v_i-1
      matrix_with_3_diagonal[i][1] = -(ai(xi, step) / (step * step)) - (ai(xi + step, step) / (step * step)) - di(xi, step); // коэф при v_i
      matrix_with_3_diagonal[i][2] = ai(xi + step, step) / (step * step);                                                    // коэф при v_i+1
      b[i] = phi_i(xi, step);
    }
  }

  matrix_with_3_diagonal[n - 1][0] = 0;
  matrix_with_3_diagonal[n - 1][1] = 0;
  matrix_with_3_diagonal[n - 1][2] = MU2;
  // прогонка

  // вывод матрицы
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < 3; j++)
      std::cout << matrix_with_3_diagonal[i][j] << "\t";
    std::cout << b[i];
    std::cout << "\n";
  }
}

int main()
{
  method_balance(5, 1);

  return 0;
}
