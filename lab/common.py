import math

DECIMAL_DIGITS_INTERMEDIATE = 6
DECIMAL_DIGITS_FINAL = 6

EPS = 1e-4

# Localized root x ∈ [a, b]
a, b = 0, 0.2

x0 = (a + b) / 2


def f(x):
    return (x - 1) ** 3 + 0.5 * math.e ** x


def f_derivative(x):
    return 3 * (x - 1) ** 2 + 0.5 * math.e ** x


def f_second_derivative(x):
    return 6 * (x - 1) + 0.5 * math.e ** x
