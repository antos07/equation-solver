from lab import simple_iterations, relaxation, newton, plot, common

if __name__ == '__main__':

    print('ОСНОВНА ІНФОРМАЦІЯ\n')
    print('Варіант 48\n')
    print('Рівняння: (x - 1) ^ 3 + 0.5 * e ^ x = 0')
    print('f(x) = (x - 1) ^ 3 + 0.5 * e ^ x\n')

    plot.draw()

    print('Рівняння має єдиний корінь x*')
    print(f'Локалізація кореня: x* ∈ [{common.a}; {common.b}]')
    f_a_value = common.f(common.a)
    f_b_value = common.f(common.b)
    print(f'f({common.a:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}) = '
          f'{f_a_value:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
    print(f'f({common.b:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}) = '
          f'{f_b_value:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')
    if f_a_value * f_b_value < 0:
        print(f'f({common.a:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}) '
              f'* f({common.b:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}) '
              f'< 0 - Проміжок містить один корінь')
    else:
        print(f'f({common.a:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}) '
              f'* f({common.b:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}) '
              f'>- 0 - Проміжок не містить коренів')
        exit(0)

    print(f'\nДля всіх методів x_0 = {common.x0:.{common.DECIMAL_DIGITS_INTERMEDIATE}f}')

    print()

    simple_iterations.run()
    print()
    print()
    relaxation.run()
    print()
    print()
    newton.run()
