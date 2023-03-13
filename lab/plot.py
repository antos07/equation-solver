import matplotlib.pyplot as plt
import numpy as np

from lab import common


def draw():
    # 200 linearly spaced numbers
    x = np.linspace(-0.3, 1, 200)

    y = common.f(x)

    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(x, y, 'g', label="y=f(x)")
    plt.legend(loc='lower right')

    # show the plot
    plt.show()
