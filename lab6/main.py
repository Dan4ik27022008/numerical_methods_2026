import numpy as np


# 1. Генерація матриці A та вектора B
def generate_data(n):
    A = np.random.rand(n, n)
    x_true = np.full(n, 2.5)  # Заданий розв'язок за умовою
    b = A @ x_true  # Обчислення вектора вільних членів

    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_B.txt', b)
    return n


# 2. Функції для LU-розкладу та розв'язання
def lu_decomposition(A, n):
    L = np.zeros((n, n))
    U = np.eye(n)  # Діагональні елементи U = 1

    for k in range(n):
        # Обчислення елементів L (стовпці)
        for i in range(k, n):
            L[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(k))

        # Обчислення елементів U (рядки)
        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(k))) / L[k, k]

    return L, U


def solve_lu(L, U, b, n):
    # Розв'язок LZ = B (прямий хід)
    z = np.zeros(n)
    for k in range(n):
        z[k] = (b[k] - sum(L[k, j] * z[j] for j in range(k))) / L[k, k]

    # Розв'язок UX = Z (зворотний хід)
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = z[k] - sum(U[k, j] * x[j] for j in range(k + 1, n))

    return x


# 3. Ітераційне уточнення
def refine_solution(A, L, U, b, x0, n, eps):
    x = x0.copy()
    iterations = 0

    while True:
        iterations += 1
        # Обчислення вектора нев'язки R = B - AX
        r = b - A @ x

        # Перевірка умови виходу за нормою нев'язки
        if np.linalg.norm(r, np.inf) < eps:
            break

        # Розв'язання системи A * delta_x = r за допомогою готового LU
        delta_x = solve_lu(L, U, r, n)
        x = x + delta_x

        if np.linalg.norm(delta_x, np.inf) < eps or iterations > 100:
            break

    return x, iterations


# Основний цикл виконання
eps = 1e-14
n = generate_data(100)
A = np.loadtxt('matrix_A.txt')
b = np.loadtxt('vector_B.txt')

# LU-розклад
L, U = lu_decomposition(A, n)
np.savetxt('LU_decomposition.txt', L + U - np.eye(n))  # Збереження в один файл

# Початковий розв'язок
x_initial = solve_lu(L, U, b, n)

# Оцінка початкової точності
initial_error = np.max(np.abs(A @ x_initial - b))
print(f"Початкова похибка (eps): {initial_error:.2e}")

# Уточнення
x_refined, iters = refine_solution(A, L, U, b, x_initial, n, eps)
final_error = np.max(np.abs(A @ x_refined - b))

print(f"Кількість ітерацій для уточнення: {iters}")
print(f"Кінцева похибка після уточнення: {final_error:.2e}")
print(f"Перші 5 елементів уточненого розв'язку: {x_refined[:5]}")