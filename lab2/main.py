import csv
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. Підготовка та зчитування даних
# ==========================================
def create_sample_csv(filename="data.csv"):
    """Створює CSV файл з даними для Варіанту 3."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n', 't'])
        writer.writerows([
            [10000, 8],
            [20000, 20],
            [40000, 55],
            [80000, 150],
            [160000, 420]
        ])


def read_data(filename):
    """Зчитує дані з CSV файлу."""
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return x, y


# ==========================================
# 2. Метод прогонки та Кубічні сплайни
# ==========================================
def sweep_method(a, b, c, d):
    """Розв'язання СЛАР з тридіагональною матрицею методом прогонки."""
    n = len(d)
    p = [0] * n
    q = [0] * n

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * p[i - 1]
        if i < n - 1:
            p[i] = -c[i] / denom
        q[i] = (d[i] - a[i] * q[i - 1]) / denom

    x = [0] * n
    x[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    return x


def cubic_splines_coefficients(x, y):
    """Обчислення коефіцієнтів кубічних сплайнів та їх виведення у консолі."""
    n = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(n)]
    a_coef = y[:-1]

    # Виправлено: додано [0] в кінець списку A, щоб зрівняти його довжину з B, C та D
    A = [0] + [h[i - 1] for i in range(1, n)] + [0]
    B = [1] + [2 * (h[i - 1] + h[i]) for i in range(1, n)] + [1]
    C = [h[i] for i in range(n)] + [0]
    D = [0] + [3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) for i in range(1, n)] + [0]

    c_coef = sweep_method(A, B, C, D)

    b_coef = [0] * n
    d_coef = [0] * n
    for i in range(n):
        b_coef[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c_coef[i + 1] + 2 * c_coef[i]) / 3
        d_coef[i] = (c_coef[i + 1] - c_coef[i]) / (3 * h[i])

    print("\n--- Коефіцієнти кубічних сплайнів ---")
    for i in range(n):
        print(
            f"Відрізок {i} [{x[i]}-{x[i + 1]}]: a={a_coef[i]:.4f}, b={b_coef[i]:.4f}, c={c_coef[i]:.4f}, d={d_coef[i]:.4f}")
    return a_coef, b_coef, c_coef, d_coef


# ==========================================
# 3. Методи Інтерполяції
# ==========================================
def divided_differences(x, y):
    """Таблиця розділених різниць для многочлена Ньютона."""
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef[0, :]


def newton_polynomial(coef, x_data, x):
    """Обчислення значення інтерполяційного многочлена Ньютона."""
    n = len(x_data) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p


def lagrange_polynomial(x_data, y_data, x):
    """Обчислення значення інтерполяційного многочлена Лагранжа."""
    total = 0
    n = len(x_data)
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term = term * (x - x_data[j]) / (x_data[i] - x_data[j])
        total += term
    return total


# ==========================================
# 4. Основний блок (Завдання 3 варіанту)
# ==========================================
#create_sample_csv()
x_data, y_data = read_data("data.csv")

print("Вхідні дані (розмір вибірки):", x_data)
print("Вхідні дані (час, с):", y_data)

# Обчислення сплайнів
cubic_splines_coefficients(x_data, y_data)

# Інтерполяція Ньютона
newton_coefs = divided_differences(x_data, y_data)
target_x = 120000
newton_pred = newton_polynomial(newton_coefs, x_data, target_x)
lagrange_pred = lagrange_polynomial(x_data, y_data, target_x)

print(f"\n--- Прогноз для розміру {target_x} ---")
print(f"Метод Ньютона: {newton_pred:.2f} сек")
print(f"Метод Лагранжа: {lagrange_pred:.2f} сек")

# Графік для Варіанту 3
x_vals = np.linspace(min(x_data), max(x_data), 500)
y_newton = [newton_polynomial(newton_coefs, x_data, xi) for xi in x_vals]

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_newton, label="Інтерполяція Ньютона", color='blue')
plt.scatter(x_data, y_data, color='red', zorder=5, label="Експериментальні дані")
plt.scatter([target_x], [newton_pred], color='green', marker='x', s=100, zorder=5, label=f"Прогноз ({target_x})")
plt.title("Прогноз часу тренування моделі")
plt.xlabel("Розмір датасету")
plt.ylabel("Час (с)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 5. Дослідницька частина (Ефект Рунге)
# ==========================================
print("\n--- Дослідницька частина: Ефект Рунге ---")


# Використаємо класичну функцію Рунге: f(x) = 1 / (1 + 25x^2) на [-1, 1]
def runge_func(x):
    return 1 / (1 + 25 * x ** 2)


nodes_list = [5, 10, 20]
x_dense = np.linspace(-1, 1, 500)
y_true = runge_func(x_dense)

plt.figure(figsize=(12, 6))
plt.plot(x_dense, y_true, label="Справжня функція f(x)", color='black', linewidth=2)

for n in nodes_list:
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge_func(x_nodes)

    coefs = divided_differences(x_nodes, y_nodes)
    y_interp = [newton_polynomial(coefs, x_nodes, xi) for xi in x_dense]

    plt.plot(x_dense, y_interp, label=f"Ньютон (n={n})", linestyle='--')

    # Розрахунок похибки
    error = np.max(np.abs(y_true - y_interp))
    print(f"Максимальна похибка для {n} вузлів: {error:.4f}")

plt.title("Дослідження ефекту Рунге (Рівномірні вузли)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.ylim(-0.5, 1.5)
plt.legend()
plt.grid(True)
plt.show()