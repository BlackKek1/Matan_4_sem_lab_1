import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

LEFT = 0.0
RIGHT = 4.0
DIVIDER = "=" * 72


# f(x) = 1 / (x + 1)
def f(x):
  x = np.asarray(x, dtype=float)
  return 1.0 / (x + 1.0)


# F(x) = 2x ^ 2 + 3
def F(x):
  x = np.asarray(x, dtype=float)
  return 2.0 * (x ** 2) + 3.0


# Значение f_n для листа x-ов
def f_n(x, n, a=LEFT, b=RIGHT):
  x = np.asarray(x, dtype=float)
  h = (b - a) / n
  ind = np.floor((x - a) / h).astype(int)
  ind = np.clip(ind, 0, n - 1)
  x_right = a + (ind + 1) * h
  y = f(x_right)
  y = np.where(x >= b, f(b), y)
  return y


# Интеграл Лебега для конкретного f_n
def lebeg_integral_fn(n, a=LEFT, b=RIGHT):
  step = (b - a) / n
  x_right = a + np.arange(1, n + 1) * step
  return step * np.sum(f(x_right))


# Интеграл Лебега–Стилтьеса для конкретного f_n
def stieltjes_integral_fn(n, a=LEFT, b=RIGHT):
  step = (b - a) / n
  x_left = a + np.arange(0, n) * step
  x_right = a + np.arange(1, n + 1) * step
  return np.sum(f(x_right) * (F(x_right) - F(x_left)))


# Аналитически полученные значения
lebeg_analytical = np.log(5.0)
lebeg_stieltjes_analytical = 16.0 - 4.0 * np.log(5.0)

print(DIVIDER)
print(f"Аналитическое значение интеграла Лебега           = {lebeg_analytical:.8f}")
print(f"Аналитическое значение интеграла Лебега-Стилтьеса = {lebeg_stieltjes_analytical:.8f}")
print(DIVIDER)

# График аппроксимации простыми функциями f_n
x_plot = np.linspace(LEFT, RIGHT, 2000)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, f(x_plot), label="f(x) = 1 / (x + 1)")

n_plot_values = [5, 10, 100, 500]
for n in n_plot_values:
  plt.step(x_plot, f_n(x_plot, n), where="post", label=f"f_{n}(x)")

plt.title("Аппроксимация f простыми функциями f_n")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

n_values_for_calc_integral = [10, 100, 1000]
n_values_for_calc_integral_stieltjes = [10, 100, 500, 1000, 5000]

# Подсчет интегралов Лебега для различных f_n (при разных n)
table_data = []
for n in n_values_for_calc_integral:
  t_start = time.perf_counter()
  integral = lebeg_integral_fn(n)
  dt = time.perf_counter() - t_start
  diff = abs(integral - lebeg_analytical)
  table_data.append([n, f"{integral:.8f}", f"{diff:.8f}", f"{dt:.6f}"])

print("Результаты для интеграла Лебега")
print(
  tabulate(
    table_data,
    headers=["n", "∫ f_n dμ", "отклонение", "время, c"],
    tablefmt="grid",
  )
)
print(DIVIDER)

# Подсчет интегралов Лебега–Стилтьеса для различных f_n (при разных n)
table_data = []
for n in n_values_for_calc_integral_stieltjes:
  t_start = time.perf_counter()
  integral = stieltjes_integral_fn(n)
  dt = time.perf_counter() - t_start
  diff = abs(integral - lebeg_stieltjes_analytical)
  table_data.append([n, f"{integral:.8f}", f"{diff:.8f}", f"{dt:.6f}"])

print("Результаты для интеграла Лебега–Стилтьеса")
print(
  tabulate(
    table_data,
    headers=["n", "∫ f_n dμ_F", "отклонение", "время, c"],
    tablefmt="grid"
  )
)
print(DIVIDER)
