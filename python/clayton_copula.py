import numpy as np
import matplotlib.pyplot as plt


def clayton_copula_pdf_inverse(u, theta, t):
    return np.power(np.power(u, -theta) * (np.power(t, -theta / (theta + 1)) - 1.0) + 1.0, -1.0 / theta)


def generate_clayton_copula_2d(theta, sim_count=50):
    u_list = np.random.uniform(size=sim_count)
    t_list = np.random.uniform(size=sim_count)
    v_list = np.vectorize(lambda u, t: clayton_copula_pdf_inverse(u, theta, t))(u_list, t_list)
    return np.stack((u_list, v_list))


if __name__ == '__main__':
    theta = 3.0

    np.random.seed(1)
    sim_count = 1000
    sim_result = generate_clayton_copula_2d(theta, sim_count)

    plt.plot(sim_result[0], sim_result[1], 'r+')
    plt.axis([0, 1, 0, 1])
    plt.show()
