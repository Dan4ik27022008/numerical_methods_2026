import math
import numpy as np
import matplotlib.pyplot as plt


# --- 1. Оголошення функцій ---
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)


def dM(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)


def diff_central(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)


t0 = 1.0
exact_val = dM(t0)

# Графік 1: M(t) та M'(t)
t_vals = np.linspace(0, 20, 500)
plt.figure(figsize=(10, 6))
plt.plot(t_vals, M(t_vals), label='M(t) - Вологість', color='blue')
plt.plot(t_vals, dM(t_vals), label="M'(t) - Швидкість висихання", color='red')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(t0, color='green', linestyle=':', label=f'Досліджувана точка t0={t0}')
plt.title("Вологість ґрунту та швидкість її зміни (П. 1)")
plt.xlabel("Час t")
plt.ylabel("Значення")
plt.legend()
plt.grid(True)
plt.show()

# --- 2. Дослідження залежності похибки від кроку h ---
h_vals = np.logspace(-16, 1, 100)
errors = [abs(diff_central(t0, h) - exact_val) for h in h_vals]

# Знаходимо оптимальний крок
best_h_idx = np.argmin(errors)
best_h = h_vals[best_h_idx]
min_error = errors[best_h_idx]

print(f"2. Оптимальний крок h0: {best_h:.2e}, Досягнута точність: {min_error:.2e}")

# Графік 2: Похибка vs Крок
plt.figure(figsize=(10, 6))
plt.loglog(h_vals, errors, marker='.', linestyle='-', color='purple')
plt.axvline(best_h, color='orange', linestyle='--', label=f'Оптимальний крок ~{best_h:.1e}')
plt.title("Залежність похибки R від кроку h (П. 2)")
plt.xlabel("Крок h (логарифмічна шкала)")
plt.ylabel("Похибка R (логарифмічна шкала)")
plt.gca().invert_xaxis()
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# Графік 3: Геометрія центральної різниці
plt.figure(figsize=(8, 8))
t_zoom = np.linspace(0.2, 1.8, 100)
plt.plot(t_zoom, M(t_zoom), label='M(t)', color='blue', linewidth=2)
plt.plot(t_zoom, M(t0) + exact_val * (t_zoom - t0), label='Дотична (точна похідна)', color='red', linestyle='--')

for h_vis, col in zip([0.2, 0.4, 0.6], ['green', 'orange', 'purple']):
    slope = diff_central(t0, h_vis)
    plt.plot(t_zoom, M(t0) + slope * (t_zoom - t0), label=f'Апроксимація (h={h_vis})', color=col, linestyle='-.',
             alpha=0.7)
    plt.scatter([t0 - h_vis, t0 + h_vis], [M(t0 - h_vis), M(t0 + h_vis)], color=col, s=15)

plt.scatter([t0], [M(t0)], color='black', zorder=5, label=f'Точка t0={t0}')
plt.title("Геометричний зміст апроксимації (збільшені кроки)")
plt.xlabel("t")
plt.ylabel("M(t)")
plt.legend()
plt.grid(True)
plt.show()

# --- 3-7. Обчислення методів покращення точності ---
h_base = 1e-3
print(f"\n3. Приймаємо крок h = {h_base}")

D_h = diff_central(t0, h_base)
D_2h = diff_central(t0, 2 * h_base)
D_4h = diff_central(t0, 4 * h_base)

# Рунге-Ромберг
D_RR = D_h + (D_h - D_2h) / 3

# Ейткен
D_E = (D_2h ** 2 - D_4h * D_h) / (2 * D_2h - (D_4h + D_h))
ratio = abs((D_4h - D_2h) / (D_2h - D_h))
p_order = (1 / math.log(2)) * math.log(ratio)

err_h = abs(D_h - exact_val)
err_2h = abs(D_2h - exact_val)
err_4h = abs(D_4h - exact_val)
err_RR = abs(D_RR - exact_val)
err_E = abs(D_E - exact_val)

print(f"\n4. Формула (h):  {D_h:.7f}, Похибка: {err_h:.2e}")
print(f"5. Формула (2h): {D_2h:.7f}, Похибка: {err_2h:.2e}")
print(f"6. Рунге-Ромберг: {D_RR:.7f}, Похибка: {err_RR:.2e}")
print(f"7. Ейткен:        {D_E:.7f}, Похибка: {err_E:.2e}, Порядок p: {p_order:.3f}")

# Графік 4: Порівняння похибок
labels = ['Формула (h)', 'Формула (2h)', 'Формула (4h)', 'Рунге-Ромберг', 'Ейткен']
errors_list = [err_h, err_2h, err_4h, err_RR, err_E]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, errors_list, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.yscale('log')
plt.title(f"Порівняння абсолютних похибок (базовий крок h={h_base})")
plt.ylabel("Абсолютна похибка (логарифмічна шкала)")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval * 1.2, f'{yval:.1e}', ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()