import numpy as np
import matplotlib.pyplot as plt
import math



# --- 1. Задана функція ---
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a, b = 0, 24

# --- 2. Аналітичне точне значення інтегралу I0 ---
# Використовуємо математичне обчислення замість scipy.integrate.quad
# I0 = 1200 + 0 + 5 * sqrt(pi / 0.2) * erf(12 * sqrt(0.2))
I0 = 1200 + 5 * math.sqrt(math.pi / 0.2) * math.erf(12 * math.sqrt(0.2))


x_fine = np.linspace(a, b, 1000)
y_fine = f(x_fine)


# ГРАФІК 1: Сама функція та точний інтеграл (площа під кривою)
plt.figure(figsize=(10, 5))
plt.plot(x_fine, y_fine, 'b-', linewidth=2, label='f(x)')
plt.fill_between(x_fine, y_fine, alpha=0.3, color='skyblue', label=f'Площа (I0 ≈ {I0:.2f})')
plt.title('Пункт 1-2: Навантаження на сервер та геометричний зміст інтегралу', fontsize=12)
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()


# --- 3. Функція для методу Сімпсона ---
def simpson(f, a, b, N):
    if N % 2 != 0: N += 1  # N має бути парним
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1]
    return (h / 3) * S


# --- 4. Дослідження залежності точності від N ---
N_vals = np.arange(10, 500, 2)
errors = [abs(simpson(f, a, b, N) - I0) for N in N_vals]

target_eps = 1e-12
N_opt = 10
eps_opt = abs(simpson(f, a, b, N_opt) - I0)

while eps_opt > target_eps and N_opt < 5000:
    N_opt += 2
    eps_opt = abs(simpson(f, a, b, N_opt) - I0)


# ГРАФІК 2: Дослідження похибки від N (Лінійний та Логарифмічний масштаби)
plt.figure(figsize=(8, 5))
plt.plot(N_vals, errors, 'r.-')
plt.yscale('log')
plt.title('Пункт 4: Залежність похибки методу Сімпсона від N', fontsize=14)
plt.xlabel('Число розбиттів N')
plt.ylabel('Абсолютна похибка (логарифмічна шкала)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- 5. Обчислення похибки для N0 ---
N0_approx = N_opt // 10
N0 = N0_approx + (8 - N0_approx % 8) if N0_approx % 8 != 0 else N0_approx
N0 = max(8, N0)

I_N0 = simpson(f, a, b, N0)
I_N0_2 = simpson(f, a, b, N0 // 2)
I_N0_4 = simpson(f, a, b, N0 // 4)

eps0 = abs(I_N0 - I0)

# --- 6. Метод Рунге-Ромберга ---
I_R = I_N0 + (I_N0 - I_N0_2) / 15
epsR = abs(I_R - I0)

# --- 7. Метод Ейткена ---
denom = 2 * I_N0_2 - (I_N0 + I_N0_4)
epsE = 0
if denom != 0:
    I_E = (I_N0_2 ** 2 - I_N0 * I_N0_4) / denom
    epsE = abs(I_E - I0)
    p = (1 / np.log(2)) * np.log(abs((I_N0_4 - I_N0_2) / (I_N0_2 - I_N0)))


# ГРАФІК 3: Порівняння похибок методів
methods = [f'Базовий (N={N0})', 'Рунге-Ромберг', 'Ейткен']
method_errors = [eps0, epsR, epsE]

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, method_errors, color=['gray', 'orange', 'green'])
plt.yscale('log')
plt.title(f'Пункти 6-8: Ефективність методів підвищення точності', fontsize=12)
plt.ylabel('Абсолютна похибка (log)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1e}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.show()

# --- 9. Адаптивний алгоритм ---
eval_points = set()


def adaptive_simpson(a, b, tol):
    c = (a + b) / 2
    h = b - a
    eval_points.update([a, c, b])

    I1 = (h / 6) * (f(a) + 4 * f(c) + f(b))
    d = (a + c) / 2
    e = (c + b) / 2
    eval_points.update([d, e])

    I2 = (h / 12) * (f(a) + 4 * f(d) + 2 * f(c) + 4 * f(e) + f(b))

    if abs(I1 - I2) <= 15 * tol:
        return I2 + (I2 - I1) / 15
    else:
        return adaptive_simpson(a, c, tol / 2) + adaptive_simpson(c, b, tol / 2)


tol_adapt = 1e-4
I_adapt = adaptive_simpson(a, b, tol_adapt)
eps_adapt = abs(I_adapt - I0)


# ГРАФІК 4: Розподіл вузлів в адаптивному алгоритмі
points = sorted(list(eval_points))
y_points = f(np.array(points))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(x_fine, y_fine, 'b-', alpha=0.5, label='f(x)')
ax1.scatter(points, y_points, color='red', s=20, zorder=5, label=f'Вузли сітки (Всього: {len(points)})')
ax1.set_title(f'Пункт 9: Робота адаптивного алгоритму (Точність $\\epsilon$ = {tol_adapt})', fontsize=12)
ax1.set_ylabel('Навантаження, f(x)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

ax2.hist(points, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Час, x (год)')
ax2.set_ylabel('Щільність вузлів')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Вивід результатів
print(f"Точне значення площі: {I0:.15f}")
print(f"Оптимальне N: {N_opt}, Похибка: {abs(simpson(f, a, b, N_opt) - I0):.2e}")
print("-" * 30)
print(f"Результати для N0 = {N0}:")
print(f"Метод Сімпсона: {I_N0:.15f}, Похибка: {abs(I_N0 - I0):.2e}")
print(f"Метод Рунге:    {I_R:.15f}, Похибка: {abs(I_R - I0):.2e}")

if denom != 0:
    print(f"Метод Ейткена:  {I_E:.15f}, Похибка: {abs(I_E - I0):.2e}")
    print(f"Порядок p (Ейткен): {p:.2f}")
else:
    print("Метод Ейткена: знаменник рівний нулю")

print("-" * 30)
print(f"Адаптивний алгоритм (tol={tol_adapt}):")
print(f"Інтеграл:       {I_adapt:.15f}, Похибка: {abs(I_adapt - I0):.2e}")
print(f"Кількість вузлів: {len(eval_points)}")