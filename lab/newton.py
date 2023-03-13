import functools
import math

from scipy import optimize

from lab import common


def run():
    print('МЕТОД НЬЮТОНА\n')

    check_for_convergence()
    print()
    calc_aprior()
    print()
    x = run_iterations()
    print(f'\nx* = {x:.{common.DECIMAL_DIGITS_FINAL}f}')


def check_for_convergence():
    print('Перевірка достатньої умови збіжності:')
    print(f'S = [{common.a}; {common.b}]')

    f_second_derivative_min_x = optimize.fminbound(common.f_second_derivative, common.a, common.b,
                                                   xtol=common.EPS / 100)
    f_second_derivative_max_x = optimize.fminbound(lambda x: -common.f_second_derivative(x), common.a, common.b,
                                                   xtol=common.EPS / 100)
    f_second_derivative_min = common.f_second_derivative(f_second_derivative_min_x)
    f_second_derivative_max = common.f_second_derivative(f_second_derivative_max_x)

    if f_second_derivative_min * f_second_derivative_max <= 0:
        print("Умова не виконується: f''(x) - не знакостала")
        return

    print("1) f(x_0)f''(x_0) > 0")
    f_at_x0 = common.f(common.x0)
    f_second_derivative_at_x0 = common.f_second_derivative(common.x0)
    print(f'f(x_0) = {f_at_x0:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
    print(f"f''(x_0) = {f_second_derivative_at_x0:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}")
    if f_at_x0 * f_second_derivative_at_x0 <= 0:
        print('Умова не виконується')
        return

    print("2) q < 1")
    q = calc_q()
    if q >= 1:
        print('Умова не виконується')
        return
    print('Умова виконується')


def calc_aprior():
    print('Апріорна оцінка:')
    n = math.floor(math.log2(math.log(calc_x_diff_abs_max() / common.EPS) / math.log(1 / calc_q()) + 1))
    print(f'n >= [log2(ln(|x_0 − x*| / ε) / ln(1 / q) + 1)] + 1 = {n:.0f}')


def run_iterations(max_iterations=100):
    print('Ітерації:')
    x_prev = common.x0
    for i in range(1, max_iterations+1):
        x_n = x_prev - common.f(x_prev) / common.f_derivative(x_prev)
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


@functools.cache
def calc_q():
    print('q = M_2 * |x_0 − x*| / (2 * m_1)')
    x_diff_abs_max = calc_x_diff_abs_max()
    M2 = calc_M2()
    m1 = calc_m1()
    q = M2 * x_diff_abs_max / (2 * m1)
    print(f'q = {q:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
    return q


@functools.cache
def calc_x_diff_abs_max():
    def x_diff_abs(x):
        return abs(common.x0 - x)

    x_diff_abs_max = x_diff_abs(optimize.fminbound(lambda x: -x_diff_abs(x), common.a, common.b, xtol=common.EPS / 100))
    print(f"|x_0 − x*| <= {x_diff_abs_max:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}")
    return x_diff_abs_max


@functools.cache
def calc_M2():
    x_max = optimize.fminbound(lambda x: -abs(common.f_second_derivative(x)), common.a, common.b, xtol=common.EPS / 100)
    M2 = abs(common.f_second_derivative(x_max))
    print(f"M_2 = max|f''(x)| = {M2:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}, x ∈ S")
    return M2


@functools.cache
def calc_m1():
    x_min = optimize.fminbound(lambda x: abs(common.f_derivative(x)), common.a, common.b, xtol=common.EPS / 100)
    m1 = abs(common.f_derivative(x_min))
    print(f"m_1 = min|f''(x)| = {m1:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}, x ∈ S")
    return m1
