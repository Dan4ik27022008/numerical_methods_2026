import requests
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1-3. Запит до API та отримання даних
# ==========================================
url = (
    "https://api.open-elevation.com/api/v1/lookup?locations="
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)

try:
    response = requests.get(url)
    data = response.json()
    results = data["results"]
except Exception as e:
    print(f"Помилка API: {e}. Використовуються резервні дані.")
    # Резервні координати, якщо API не відповідає
    coords_mock = [
        (48.164214, 24.536044), (48.164983, 24.534836), (48.165605, 24.534068),
        (48.166228, 24.532915), (48.166777, 24.531927), (48.167326, 24.530884),
        (48.167011, 24.530061), (48.166053, 24.528039), (48.166655, 24.526064),
        (48.166497, 24.523574), (48.166128, 24.520214), (48.165416, 24.517170),
        (48.164546, 24.514640), (48.163412, 24.512980), (48.162331, 24.511715),
        (48.162015, 24.509462), (48.162147, 24.506932), (48.161751, 24.504244),
        (48.161197, 24.501793), (48.160580, 24.500537), (48.160250, 24.500106)
    ]
    # Приблизні висоти для резерву
    elevations = [1250, 1280, 1310, 1340, 1370, 1400, 1450, 1500, 1550, 1600,
                  1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2030, 2050, 2061]
    results = [{"latitude": c[0], "longitude": c[1], "elevation": e} for c, e in zip(coords_mock, elevations)]

n = len(results)
print("Кількість вузлів:", n)
print("\nТабуляція вузлів:")
print(" i |  Latitude   |  Longitude  | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")


# ==========================================
# 4. Обчислення кумулятивної відстані
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])
distances = [0.0]

for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

distances = np.array(distances)

print("\nТабуляція (відстань, висота):")
print(" i | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")


# ==========================================
# 6-7. Метод прогонки для системи рівнянь [cite: 139-141]
# ==========================================
def tridiagonal_matrix_algorithm(alpha, beta, gamma, delta):
    n = len(delta)
    A = np.zeros(n)
    B = np.zeros(n)
    x = np.zeros(n)

    # Пряма прогонка [cite: 53-64]
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    for i in range(1, n - 1):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom

    denom = alpha[-1] * A[-2] + beta[-1]
    x[-1] = (delta[-1] - alpha[-1] * B[-2]) / denom

    # Зворотна прогонка [cite: 66-72]
    for i in range(n - 2, -1, -1):
        x[i] = A[i] * x[i + 1] + B[i]

    return x


# ==========================================
# 8-9. Обчислення коефіцієнтів a, b, c, d [cite: 142-144]
# ==========================================
def compute_cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)

    # Формування системи для c_i [cite: 40-49]
    alpha = np.zeros(n - 1)
    beta = np.zeros(n - 1)
    gamma = np.zeros(n - 1)
    delta = np.zeros(n - 1)

    for i in range(1, n):
        idx = i - 1
        alpha[idx] = h[i - 1] if i > 1 else 0
        beta[idx] = 2 * (h[i - 1] + h[i])
        gamma[idx] = h[i] if i < n - 1 else 0
        delta[idx] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    c_inner = tridiagonal_matrix_algorithm(alpha, beta, gamma, delta)
    c = np.zeros(n + 1)
    c[1:n] = c_inner  # c_0 = 0, c_n = 0 [cite: 45, 48]

    a = y[:-1]  # a_i = y_{i-1} [cite: 36]
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3  # [cite: 38]
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])  # [cite: 37]

    return a, b, c, d


# Функція для знаходження значення сплайна
def evaluate_spline(xx, x_nodes, a, b, c, d):
    yy = np.zeros_like(xx)
    for j, x_val in enumerate(xx):
        # Знайти потрібний інтервал
        idx = np.searchsorted(x_nodes, x_val) - 1
        idx = np.clip(idx, 0, len(a) - 1)
        dx = x_val - x_nodes[idx]
        yy[j] = a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3  # [cite: 11]
    return yy


# Обчислення коефіцієнтів для всього набору даних
a, b, c, d = compute_cubic_spline(distances, elevations)
print("\nКоефіцієнти сплайнів (перші 5):")
for i in range(min(5, len(a))):
    print(f"Інтервал {i}: a={a[i]:.2f}, b={b[i]:.2f}, c={c[i]:.2f}, d={d[i]:.4f}")

# ==========================================
# 10. Побудова графіків для 10, 15, 20 вузлів
# ==========================================
plt.figure(figsize=(12, 8))
x_dense = np.linspace(distances[0], distances[-1], 500)

for num_nodes in [10, 15, 20]:
    # Вибираємо рівномірно розподілені вузли
    indices = np.linspace(0, len(distances) - 1, num_nodes, dtype=int)
    x_sub = distances[indices]
    y_sub = elevations[indices]

    a_sub, b_sub, c_sub, d_sub = compute_cubic_spline(x_sub, y_sub)
    y_interp = evaluate_spline(x_dense, x_sub, a_sub, b_sub, c_sub, d_sub)

    plt.plot(x_dense, y_interp, label=f'Сплайн ({num_nodes} вузлів)')
    plt.scatter(x_sub, y_sub, s=20)

plt.plot(distances, elevations, 'k--', label='Оригінальні дані (22 вузли)', alpha=0.5)
plt.title('Профіль висоти маршруту: Інтерполяція кубічними сплайнами')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Висота (м)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# ДОДАТКОВЕ ЗАВДАННЯ
# ==========================================
print("\n--- ХАРАКТЕРИСТИКИ МАРШРУТУ ---")
# 1. Довжина та набори висоти [cite: 151-157]
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, len(elevations)))
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, len(elevations)))
print(f"Сумарний спуск (м): {total_descent:.2f}")

# 2. Аналіз градієнта [cite: 158-169]
yy_full = evaluate_spline(x_dense, distances, a, b, c, d)
grad_full = np.gradient(yy_full, x_dense) * 100
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

# 3. Механічна енергія підйому [cite: 170-180]
mass = 80
g = 9.81
energy = mass * g * total_ascent
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Механічна робота (кДж): {energy / 1000:.2f}")
print(f"Енергія (ккал): {energy / 4184:.2f}")