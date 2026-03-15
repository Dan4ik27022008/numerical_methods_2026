import csv
import matplotlib.pyplot as plt


# 1. Зчитування середньомісячних температур з CSV
def load_data(filename):
    x = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Пропуск заголовка
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y


# 2. Функції МНК
def form_matrix(x, m):
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum(xk ** (i + j) for xk in x)
    return A


def form_vector(x, y, m):
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(yk * (xk ** i) for xk, yk in zip(x, y))
    return b


def gauss_solve(A, b):
    n = len(b)
    # Прямий хід з вибором головного елемента
    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[max_row][k]):
                max_row = i
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

            # Зворотній хід
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (b[i] - s) / A[i][i]
    return x_sol


def polynomial(x_vals, coef):
    y_poly = [0.0] * len(x_vals)
    for idx, xv in enumerate(x_vals):
        for i in range(len(coef)):
            y_poly[idx] += coef[i] * (xv ** i)
    return y_poly


def variance(y_true, y_approx):
    # Дисперсія / середньоквадратична похибка
    return sum((yt - ya) ** 2 for yt, ya in zip(y_true, y_approx)) / len(y_true)


# Функції для кубічних сплайнів та методу прогонки
def sweep_method(a, b_diag, c, d):
    n = len(d)
    alpha = [0.0] * n
    beta = [0.0] * n

    alpha[0] = -c[0] / b_diag[0]
    beta[0] = d[0] / b_diag[0]

    for i in range(1, n - 1):
        denom = b_diag[i] + a[i] * alpha[i - 1]
        alpha[i] = -c[i] / denom
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom

    x_sol = [0.0] * n
    x_sol[-1] = (d[-1] - a[-1] * beta[-2]) / (b_diag[-1] + a[-1] * alpha[-2])

    for i in range(n - 2, -1, -1):
        x_sol[i] = alpha[i] * x_sol[i + 1] + beta[i]
    return x_sol


def calc_and_print_cubic_splines(x, y):
    n = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(n)]

    a = [0.0] * (n - 1)
    b_diag = [2 * (h[i] + h[i + 1]) for i in range(n - 1)]
    c = [0.0] * (n - 1)
    d = [0.0] * (n - 1)

    for i in range(n - 1):
        if i > 0: a[i] = h[i]
        if i < n - 2: c[i] = h[i + 1]
        d[i] = 6 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i])

    c_inner = sweep_method(a, b_diag, c, d)
    c_coef = [0.0] + c_inner + [0.0]

    print("\n--- Коефіцієнти кубічних сплайнів ---")
    for i in range(n):
        a_i = y[i]
        b_i = (y[i + 1] - y[i]) / h[i] - h[i] * (c_coef[i + 1] + 2 * c_coef[i]) / 3
        d_i = (c_coef[i + 1] - c_coef[i]) / (3 * h[i])
        print(f"Відрізок {i + 1} [x={x[i]}..{x[i + 1]}]: a={a_i:.4f}, b={b_i:.4f}, c={c_coef[i]:.4f}, d={d_i:.4f}")
    print("-------------------------------------\n")


def main():
    # 1. Вхідні дані
    x, y = load_data('data.csv')

    # Виведення коефіцієнтів кубічних сплайнів у консоль
    calc_and_print_cubic_splines(x, y)

    # 3. Вибір оптимального ступеня полінома
    max_degree = 12
    variances = []

    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b_vec = form_vector(x, y, m)
        coef = gauss_solve(A, b_vec)
        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        variances.append(var)

    optimal_m = variances.index(min(variances)) + 1

    # 4. Побудова апроксимації
    A_opt = form_matrix(x, optimal_m)
    b_opt = form_vector(x, y, optimal_m)
    coef_opt = gauss_solve(A_opt, b_opt)
    y_approx_opt = polynomial(x, coef_opt)

    # 5. Прогноз на наступні 3 місяці
    last_x = x[-1]
    x_future = [last_x + 1, last_x + 2, last_x + 3]
    y_future = polynomial(x_future, coef_opt)

    # 6. Похибка апроксимації
    error_y = [abs(yt - ya) for yt, ya in zip(y, y_approx_opt)]

    # 7. Вивід результатів у консоль
    print("Дисперсії для різних ступенів (m=1..10):")
    for m, var in enumerate(variances, 1):
        print(f"m = {m}:\t{var:.4f}")

    print(f"\nОптимальний ступінь полінома: m = {optimal_m}")
    print(f"Прогноз температур на наступні 3 місяці {x_future}:")
    for month, temp in zip(x_future, y_future):
        print(f"Місяць {month}: {temp:.2f}")

    # Побудова графіків
    plt.figure(figsize=(10, 8))

    # Графік 1: Апроксимація
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'ko', label='Фактичні дані (CSV)')
    plt.plot(x, y_approx_opt, 'b-', linewidth=2, label=f'Апроксимація (МНК, m={optimal_m})')
    plt.plot(x_future, y_future, 'r--', marker='o', label='Прогноз (Екстраполяція)')
    plt.title('Апроксимація температурних даних')
    plt.xlabel('Місяць')
    plt.ylabel('Температура')
    plt.legend()
    plt.grid(True)

    # Графік 2: Похибка
    plt.subplot(2, 1, 2)
    plt.plot(x, error_y, 'g-', marker='.', label='Похибка $\epsilon(x)$')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title('Похибка апроксимації')
    plt.xlabel('Місяць')
    plt.ylabel('Відхилення')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()