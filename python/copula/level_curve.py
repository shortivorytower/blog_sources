import numpy as np
import matplotlib.pyplot as plt


def clayton_copula_generator(theta, t):
    return np.power(t, -theta) - 1


def clayton_copula_inverted_generator(theta, s):
    return np.power(1.0 + s, -1.0 / theta)


def level_curve(theta, u, t):
    """Get the level curve value of 'v' from given u at level t
    """
    v = clayton_copula_inverted_generator(theta, clayton_copula_generator(theta, t) - clayton_copula_generator(theta, u))
    return v

if __name__ == '__main__':
    theta = 3.0
    u_list = np.arange(0.01, 1, 0.01)
    t = 0.6
    v_list = level_curve(theta, u_list, t)
    print(u_list)
    print(v_list)

    plt.plot(u_list, v_list, 'b+')
    plt.axis([0, 1, 0, 1])
    plt.show()
