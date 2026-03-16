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


def cubic_splines_coefficients(x, y, prnt):
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

    if prnt:
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
create_sample_csv()
x_data, y_data = read_data("data.csv")

print("Вхідні дані (розмір вибірки):", x_data)
print("Вхідні дані (час, с):", y_data)

# Обчислення сплайнів
cubic_splines_coefficients(x_data, y_data, True)

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

plt.figure(figsize=(10, 5))

# Побудова ліній для 3, 4 та 5 вузлів
# Кольори адаптовано для кращої видимості на світлому фоні
nodes_configs = [
    (3, '--', '#00acc1'),  # Блакитний пунктир
    (4, '--', '#43a047'),  # Зелений пунктир
    (5, '-', '#d81b60')  # Рожево-червоний суцільний
]

for n_nodes, ls, color in nodes_configs:
    # Беремо перші n вузлів
    x_nodes = x_data[:n_nodes]
    y_nodes = y_data[:n_nodes]

    # Обчислюємо коефіцієнти та значення полінома
    coefs = divided_differences(x_nodes, y_nodes)
    y_interp = [newton_polynomial(coefs, x_nodes, xi) for xi in x_vals]

    # Малюємо лінію
    plt.plot(x_vals, y_interp, label=f"Ньютон ({n_nodes} вузли)", color=color, linestyle=ls, linewidth=2)

# Експериментальні дані (жовті точки)
plt.scatter(x_data, y_data, color='#ffb300', zorder=5, s=60, label="Експериментальні дані")
# Прогноз (зелений хрестик)
plt.scatter([target_x], [newton_pred], color='green', marker='x', s=100, zorder=5, label=f"Прогноз ({target_x})")

plt.title("Прогноз часу тренування моделі")
plt.xlabel("Розмір датасету")
plt.ylabel("Час (с)")
plt.legend()
plt.grid(True, alpha=0.5)  # Трохи прозоріша сітка для охайності
plt.show()

# ==========================================
# 5. Дослідницька частина (Ефект Рунге без scipy)
# ==========================================
print("\n--- Дослідницька частина: Ефект Рунге на базі датасету ---")

# Власна функція для обчислення значення кубічного сплайна у будь-якій точці x
def evaluate_spline(x_val, x_data, a, b, c, d):
    n = len(x_data) - 1
    # Шукаємо, до якого відрізка належить заданий x_val
    for i in range(n):
        if x_data[i] <= x_val <= x_data[i + 1]:
            dx = x_val - x_data[i]
            return a[i] + b[i] * dx + c[i] * (dx ** 2) + d[i] * (dx ** 3)

    # На випадок похибок округлення (якщо x_val трохи більше за останній вузол)
    dx = x_val - x_data[-2]
    return a[-1] + b[-1] * dx + c[-1] * (dx ** 2) + d[-1] * (dx ** 3)

# Отримуємо коефіцієнти сплайна, використовуючи вашу ж функцію з п.2
# Вони створять ідеальну плавну лінію через 5 експериментальних точок
a_coef, b_coef, c_coef, d_coef = cubic_splines_coefficients(x_data, y_data, False)

# Генеруємо 500 точок для еталонної кривої
x_dense = np.linspace(min(x_data), max(x_data), 500)
y_true = np.array([evaluate_spline(x, x_data, a_coef, b_coef, c_coef, d_coef) for x in x_dense])

# --- ПЕРШИЙ ГРАФІК: Інтерполяція (Ефект Рунге) ---
plt.figure(figsize=(12, 6))
plt.plot(x_dense, y_true, label="Еталонна крива (Сплайн)", color='black', linewidth=2)
plt.scatter(x_data, y_data, color='red', zorder=5, label="Дані (5 точок)", s=50)

# Щоб уникнути переповнення пам'яті, тимчасово зменшуємо масштаб X
scale_x = 10000
nodes_list = [5, 10, 20]

# Словник для збереження масивів похибок для другого графіка
errors_dict = {}

for n in nodes_list:
    # 1. Беремо n рівномірних точок з нашого еталонного сплайна
    x_nodes = np.linspace(min(x_data), max(x_data), n)
    y_nodes = [evaluate_spline(x, x_data, a_coef, b_coef, c_coef, d_coef) for x in x_nodes]

    # 2. Масштабуємо вузли перед обчисленням полінома Ньютона
    x_nodes_scaled = [x / scale_x for x in x_nodes]
    x_dense_scaled = [x / scale_x for x in x_dense]

    # 3. Будуємо поліном Ньютона
    coefs = divided_differences(x_nodes_scaled, y_nodes)
    y_interp = np.array([newton_polynomial(coefs, x_nodes_scaled, xi) for xi in x_dense_scaled])

    # Малюємо лінію інтерполяції
    plt.plot(x_dense, y_interp, label=f"Ньютон (n={n})", linestyle='--')

    # 4. Рахуємо масив похибок та зберігаємо його
    current_error = np.abs(y_true - y_interp)
    errors_dict[n] = current_error
    print(f"Максимальна розбіжність для {n} вузлів: {np.max(current_error):.4f}")

plt.title("Дослідження ефекту Рунге")
plt.xlabel("Розмір датасету")
plt.ylabel("Час (с)")
plt.ylim(min(y_data) - 50, max(y_data) + 150)
plt.legend()
plt.grid(True)
plt.show()

# --- Похибки інтерполяції ---
plt.figure(figsize=(12, 6))

# Стилі ліній: суцільна, пунктирна, штрихпунктирна (кольори залишаємо за замовчуванням)
line_styles = {5: '-', 10: '-', 20: '-'}

for n in nodes_list:
    # Використовуємо стандартні кольори matplotlib, але різні стилі ліній
    plt.plot(x_dense, errors_dict[n], label=f"Похибка для n={n}", linestyle=line_styles.get(n, '-'), linewidth=2)

plt.title("Абсолютна похибка інтерполяції |f(x) - P(x)| (Ефект Рунге)")
plt.xlabel("Розмір датасету")
plt.ylabel("Похибка")
# plt.yscale('log') # ВІДКЛЮЧЕНО: щоб отримати графік зі сплесками на краях як на скріншоті
plt.legend()
plt.grid(True)
plt.show()