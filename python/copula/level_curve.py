import numpy as np
import matplotlib.pyplot as plt


def level_curve(generator_func, inverted_generator_func, u, t):
    """Get the level curve value of 'v' from given u at level t
    """
    v = inverted_generator_func(generator_func(t) - generator_func(u))
    return v


if __name__ == '__main__':
    theta = 2
    u_list = np.arange(0.01, 1, 0.0001)

    clayton_generator = lambda t: np.power(t, -theta) - 1
    clayton_inverted_generator = lambda s: np.power(1.0 + s, -1.0 / theta)

    frank_generator = lambda t: -np.log((np.exp(-theta * t) - 1) / (np.exp(-theta) - 1))
    frank_inverted_generator = lambda s: -1.0 / theta * np.log(np.exp(-s) * (np.exp(-theta) - 1) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # plot clayton
    for t in np.arange(0.1, 1, 0.1):
        clayton_v_list = level_curve(clayton_generator, clayton_inverted_generator, u_list, t)
        axs[0].plot(u_list, clayton_v_list, label='t={0:0.0}'.format(t))
        axs[0].set_title('Clayton Copula Level Curves')
        axs[0].axis([0, 1, 0, 1])
        axs[0].grid()

        frank_v_list = level_curve(frank_generator, frank_inverted_generator, u_list, t)
        axs[1].plot(u_list, frank_v_list, label='t={0:0.0}'.format(t))
        axs[1].set_title('Frank Copula Level Curves')
        axs[1].axis([0, 1, 0, 1])
        axs[1].grid()

    plt.show()
