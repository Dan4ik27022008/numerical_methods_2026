import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. Генерація та збереження даних
# ==========================================
def generate_data(n=100, x_true_val=2.0):
    A = np.random.rand(n, n) * 10

    # Забезпечення діагонального переважання
    for i in range(n):
        A[i, i] = np.sum(A[i, :]) - A[i, i] + np.random.rand() * 10 + 1

    X_true = np.full(n, x_true_val)
    B = np.dot(A, X_true)

    np.savetxt("matrix_A.txt", A)
    np.savetxt("vector_B.txt", B)
    return X_true


# ==========================================
# 2. Допоміжні функції
# ==========================================
def read_matrix(filename): return np.loadtxt(filename)


def read_vector(filename): return np.loadtxt(filename)


def multiply_matrix_vector(A, x): return np.dot(A, x)


def vector_norm(x): return np.max(np.abs(x))


def matrix_norm(A): return np.max(np.sum(np.abs(A), axis=1))


# ==========================================
# 3. Ітераційні методи
# ==========================================
def simple_iteration(A, B, x0, eps):
    n = len(B)
    x_k = np.copy(x0)
    tau = 1.0 / matrix_norm(A)
    C = np.eye(n) - tau * A
    d = tau * B

    errors = []
    iterations = 0
    while True:
        x_k_next = multiply_matrix_vector(C, x_k) + d

        step_err = vector_norm(x_k_next - x_k)
        errors.append(step_err)

        iterations += 1
        if step_err < eps: break
        x_k = x_k_next

    return x_k_next, iterations, errors


def jacobi(A, B, x0, eps):
    n = len(B)
    x_k = np.copy(x0)
    x_k_next = np.zeros_like(x0)
    errors = []

    iterations = 0
    while True:
        for i in range(n):
            s = np.dot(A[i, :], x_k) - A[i, i] * x_k[i]
            x_k_next[i] = (B[i] - s) / A[i, i]

        step_err = vector_norm(x_k_next - x_k)
        errors.append(step_err)

        iterations += 1
        if step_err < eps or iterations > 4000: break
        x_k = np.copy(x_k_next)

    return x_k_next, iterations, errors


def seidel(A, B, x0, eps):
    n = len(B)
    x_k = np.copy(x0)
    x_k_next = np.copy(x0)
    errors = []

    iterations = 0
    while True:
        for i in range(n):
            s1 = np.dot(A[i, :i], x_k_next[:i])
            s2 = np.dot(A[i, i + 1:], x_k[i + 1:])
            x_k_next[i] = (B[i] - s1 - s2) / A[i, i]

        step_err = vector_norm(x_k_next - x_k)
        errors.append(step_err)

        iterations += 1
        if step_err < eps: break
        x_k = np.copy(x_k_next)

    return x_k_next, iterations, errors


# ==========================================
# 4. Виконання та побудова графіків
# ==========================================
if __name__ == "__main__":
    n = 100
    eps0 = 1e-14

    generate_data(n, 2.5)

    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")
    x0 = np.full(n, 1.0)

    print(f"Розв'язуємо систему {n}x{n} з точністю {eps0}...\n")

    # Обчислення
    sol_si, it_si, err_si = simple_iteration(A, B, x0, eps0)
    print(f"Проста ітерація: {it_si} ітерацій.")

    sol_jac, it_jac, err_jac = jacobi(A, B, x0, eps0)
    print(f"Метод Якобі: {it_jac} ітерацій.")

    sol_seid, it_seid, err_seid = seidel(A, B, x0, eps0)
    print(f"Метод Зейделя: {it_seid} ітерацій.")

    # --- ВІЗУАЛІЗАЦІЯ ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # ---------------------------------------------------------
    # Графік 1: Динаміка похибки
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.semilogy(err_si, label='Проста ітерація', color='#1f77b4', linewidth=2)
    plt.semilogy(err_jac, label='Метод Якобі', color='#ff7f0e', linewidth=2)
    plt.semilogy(err_seid, label='Метод Зейделя', color='#2ca02c', linewidth=2)

    plt.title("Спадання похибки", fontsize=14, pad=15)
    plt.xlabel("Номер ітерації", fontsize=12)
    plt.ylabel("Норма похибки (логарифмічна шкала)", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    # plt.savefig("графік_похибки.png", dpi=300) # Розкоментуйте, щоб зберегти як картинку

    # ---------------------------------------------------------
    # Графік 2: Кількість ітерацій
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    methods = ['Зейделя', 'Якобі', 'Проста ітерація']
    iterations = [it_seid, it_jac, it_si]
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

    bars = plt.bar(methods, iterations, color=colors, alpha=0.8, edgecolor='black')

    plt.title("Кількість ітерацій до досягнення заданої точності", fontsize=14, pad=15)
    plt.ylabel("Кількість ітерацій", fontsize=12)

    # Додавання точних значень над кожним стовпчиком
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(iterations) * 0.01),
                 int(yval), ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    # plt.savefig("графік_ітерацій.png", dpi=300) # Розкоментуйте, щоб зберегти як картинку

    # Показати обидва вікна
    plt.show()