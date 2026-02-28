import numpy as np
import matplotlib.pyplot as plt
import requests

# ==========================================
# 1. Запит до API
# ==========================================
url = ("https://api.open-elevation.com/api/v1/lookup?locations=48.164214,%2024.536044|48.164983,%2024.534836|48.165605,%2024.534068|48.166228,%2024.532915|48.166777,%2024.531927|48.167326,%2024.530884|48.167011,%2024.530061|48.166053,%2024.528039|48.166655,%2024.526064|48.166497,%2024.523574|48.166128,%2024.520214|48.165416,%2024.517170|48.164546,%2024.514640|48.163412,%2024.512980|48.162331,%2024.511715|48.162015,%2024.509462|48.162147,%2024.506932|48.161751,%2024.504244|48.161197,%2024.501793|48.160580,%2024.500537|48.160250,%2024.500106")

response = requests.get(url)  # [cite: 103]
data = response.json()  # [cite: 103]
results = data["results"]  # [cite: 105]

n = len(results)
print(f"Кількість вузлів: {n}")
print("Табуляція вузлів:")
print(" i |  Latitude   |  Longitude  | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")


# ==========================================
# 2. Обчислення кумулятивної відстані
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
# 3. Алгоритми кубічного сплайна
# ==========================================
def tridiagonal_matrix_algorithm(alpha, beta, gamma, delta):
    n = len(delta)
    A = np.zeros(n)
    B = np.zeros(n)
    x = np.zeros(n)

    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    for i in range(1, n - 1):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom

    denom = alpha[-1] * A[-2] + beta[-1]
    x[-1] = (delta[-1] - alpha[-1] * B[-2]) / denom

    for i in range(n - 2, -1, -1):
        x[i] = A[i] * x[i + 1] + B[i]
    return x


def compute_cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)

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
    c[1:n] = c_inner

    a = y[:-1]
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c, d


def evaluate_spline(xx, x_nodes, a, b, c, d):
    yy = np.zeros_like(xx)
    for j, x_val in enumerate(xx):
        idx = np.searchsorted(x_nodes, x_val) - 1
        idx = np.clip(idx, 0, len(a) - 1)
        dx = x_val - x_nodes[idx]
        yy[j] = a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3
    return yy


# ==========================================
# 4. Обчислення похибок (10, 15, 20 вузлів)
# ==========================================
print("\n")
# Еталонний сплайн (21 вузол)
a_ref, b_ref, c_ref, d_ref = compute_cubic_spline(distances, elevations)
x_dense = np.linspace(distances[0], distances[-1], 500)
y_ref_dense = evaluate_spline(x_dense, distances, a_ref, b_ref, c_ref, d_ref)

splines_data = {}

for num_nodes in [10, 15, 20]:
    indices = np.linspace(0, len(distances) - 1, num_nodes, dtype=int)
    x_sub = distances[indices]
    y_sub = elevations[indices]

    a_sub, b_sub, c_sub, d_sub = compute_cubic_spline(x_sub, y_sub)
    y_interp = evaluate_spline(x_dense, x_sub, a_sub, b_sub, c_sub, d_sub)

    error = np.abs(y_interp - y_ref_dense)
    splines_data[num_nodes] = {
        'y_interp': y_interp,
        'error': error
    }

    print(f"{num_nodes} вузлів")
    print(f"Максимальна похибка: {np.max(error)}")
    print(f"Середня похибка: {np.mean(error)}")

# ==========================================
# 5. Побудова графіків
# ==========================================

plt.figure("Figure 1", figsize=(8, 6))
plt.title('Вплив кількості вузлів')
plt.plot(x_dense, y_ref_dense, label='21 вузол (еталон)')
plt.plot(x_dense, splines_data[10]['y_interp'], label='10 вузлів')
plt.plot(x_dense, splines_data[15]['y_interp'], label='15 вузлів')
plt.plot(x_dense, splines_data[20]['y_interp'], label='20 вузлів')
plt.legend()
plt.show()

plt.figure("Figure 2", figsize=(8, 6))
plt.title('Похибка апроксимації')
plt.plot(x_dense, splines_data[10]['error'], label='10 вузлів')
plt.plot(x_dense, splines_data[15]['error'], label='15 вузлів')
plt.plot(x_dense, splines_data[20]['error'], label='20 вузлів')
plt.legend()
plt.show()