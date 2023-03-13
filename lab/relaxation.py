import math
import operator

from scipy import optimize

from lab import common


def run():
    print('МЕТОД РЕЛАКСАЦІЇ\n')
    m = calc_f_derivative_min()
    M = calc_f_derivative_max()

    check_for_convergence(m, M)
    print()

    t = calc_optimal_parameter(m, M)
    print()

    calc_aprior(common.x0, m, M)
    print()

    x = run_iterations(common.x0, t)
    print(f'\nx* = {x:.{common.DECIMAL_DIGITS_FINAL}f}')


def calc_f_derivative_min():
    x_min = optimize.fminbound(lambda x: abs(common.f_derivative(x)), common.a, common.b, xtol=common.EPS / 100)
    return abs(common.f_derivative(x_min))


def calc_f_derivative_max():
    x_max = optimize.fminbound(lambda x: -abs(common.f_derivative(x)), common.a, common.b, xtol=common.EPS / 100)
    return abs(common.f_derivative(x_max))


def check_for_convergence(m, M):
    print("Перевірка достатньої умови збіжності 0 < m < |f'(x)| < M, де m = min|f'(x)|, M = max|f'(x)|, x ∈ [a, b]:")
    print(f'm_1 = {m:{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
    print(f'M_1 = {M:{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
    if 0 < m < M:
        print('Достатня умова збіжності виконана')
    else:
        print('Достатня умова збіжності не виконана')


def calc_optimal_parameter(m, M):
    t = 2 / (m + M)
    print(f'Оптимальний параметр τ = 2 / (M_1 + m_1) = {t:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
    return t


def calc_aprior(x0, m, M):
    print('Обчислення апріорної оцінки:')
    q0 = (M - m) / (M + m)
    print(f'q_0 = (M_1 - m_1) / (M_1 + m_1) = {q0:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')

    def x_diff_abs(x):
        return abs(x0 - x)

    x_diff_abs_max = x_diff_abs(optimize.fminbound(lambda x: -x_diff_abs(x), common.a, common.b, xtol=common.EPS / 100))
    print(f"|x0 − x*| <= {x_diff_abs_max:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}")
    n = math.floor(math.log(x_diff_abs_max / common.EPS) / math.log(1 / q0)) + 1
    print(f"n0 >= [ln(|x0 − x*| / ε) / ln(1 / q_0)] + 1 = {n:.0f}")


def run_iterations(x0, t, max_iterations=100):
    x_prev = x0
    if common.f_derivative(x0) < 0:
        plus_minus_op = operator.add
        print("f'(x) < 0, тому x_(n+1) = x_n + τ * f(x_n)")
    else:
        plus_minus_op = operator.sub
        print("f'(x) > 0, тому x_(n+1) = x_n - τ * f(x_n)")
    print('Ітерації:')
    for i in range(1, max_iterations + 1):
        x_n = plus_minus_op(x_prev, t * common.f(x_prev))
        f_value = common.f(x_n)
        print(f'{i}) x_{i} = {x_n :.{common.DECIMAL_DIGITS_INTERMEDIATE}f}; '
              f'f(x_{i}) = {f_value:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
        if abs(x_n - x_prev) < common.EPS:
            break
        x_prev = x_n
    else:
        print('Послідовність не збіжна')
        return

    print(f'\nАпостеріорна оцінка: n = {i}')
    return x_n
