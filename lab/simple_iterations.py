import math

from scipy import optimize

from lab import common
from lab.common import x0

# Delta
d = max(abs(common.a - x0), abs(common.b - x0))
q: float


def phi(x):
    return (- x ** 3 + 3 * x ** 2 + 1 - 0.5 * math.e ** x) / 3


def phi_derivative(x):
    return - x ** 2 + 2 * x - 1 / 6 * math.e ** x


def run():
    print('МЕТОД ПРОСТИХ ІТЕРАЦІЙ\n')

    print('φ(x) = (-x ^ 3 + 3 * x ^ 2 + 1 - 0.5 * e ** x) / 3')
    print(f'δ = max(|a - x0|, |b - x0|) =', d)

    check_for_convergence()
    print()
    calc_aprior()
    print()
    x = run_iterations()
    print(f'\nx* = {x:.{common.DECIMAL_DIGITS_FINAL}f}')


def check_for_convergence():
    print('Перевірка на збіжність')
    print(f'S = [{x0 - d}; {x0 + d}]')

    print(f"1) max |φ'(x)| <= q < 1, x ∈ S")

    maximizer = optimize.fminbound(lambda x: -abs(phi_derivative(x)), common.a, common.b, xtol=common.EPS / 100)
    phi_maximum = abs(phi_derivative(maximizer))  # noqa
    print(f"max |φ'(x)| = {phi_maximum:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}, x ∈ [{x0 - d}; {x0 + d}]")
    global q
    q = round(phi_maximum, common.DECIMAL_DIGITS_INTERMEDIATE)
    print(f'q = {q:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
    if q >= 1:
        print('Достатня умова не виконана: q >= 1')
        return

    print("2)|φ(x0) − x0| <= (1 − q)δ")

    left_side = round(abs(phi(x0) - x0), common.DECIMAL_DIGITS_INTERMEDIATE)
    right_side = round((1 - q) * d, common.DECIMAL_DIGITS_INTERMEDIATE)

    print(f'|φ(x0) − x0| = {left_side:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
    print(f'(1 − q)δ = {right_side:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')

    if left_side > right_side:
        print('Достатня умова не виконана: |φ(x0) − x0| > (1 − q)δ')
        return

    print('|φ(x0) − x0| <= (1 − q)δ')
    print('Достатня умова виконана')


def calc_aprior():
    n_lb = math.floor(math.log(abs(phi(x0) - x0)
                               / ((1 - q) * common.EPS))
                      / math.log(1 / q)) + 1
    print(f'Апріорна оцінка: n >= {n_lb}')


def run_iterations(max_iterations=100):
    print('Ітерації:')
    x_prev = x0
    for i in range(1, max_iterations + 1):
        x = phi(x_prev)
        f_value = common.f(x)
        print(f'{i}) x_{i} = {x :.{common.DECIMAL_DIGITS_INTERMEDIATE}f}; '
              f'f(x_{i}) = {f_value:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
        if check_iter_stop(x_prev, x):
            break
        x_prev = x
    else:
        print('Послідовність не збіжна')
        return

    print(f'\nАпостеріорна оцінка: n = {i}')
    return x


def check_iter_stop(x_prev, x_cur):
    if q < 1/2:
        return abs(x_prev - x_cur) <= (1 - q) / q * common.EPS
    return abs(x_prev - x_cur) <= common.EPS
