import numpy as np
import matplotlib.pyplot as plt
import math



def frank_coupla(u_vector, theta):
    n = len(u_vector)
    if n < 2:
        raise ValueError('Number of random variables must be >= 2')
    if math.isclose(theta, 0.0):
        raise ValueError('Theta equals 0 is not yet implemented')

    terms = np.vectorize(lambda u: np.exp(-theta * u) - 1.0)(u_vector)
    return -1.0 / theta * np.log(1.0 + np.prod(terms) / np.power(np.exp(-theta) - 1.0, n - 1))

def frank_copula_pdf_inverse(u, theta, t):
    return -1.0 / theta * np.log(((t - 1) * np.exp(-theta * u) - t * np.exp(-theta)) / ((t - 1) * np.exp(-theta * u) - t))

def generate_frank_coupla_2d(theta, sim_count=50):
    u_list = np.random.uniform(size=sim_count)
    t_list = np.random.uniform(size=sim_count)
    v_list = np.vectorize(lambda u, t: frank_copula_pdf_inverse(u, theta, t))(u_list, t_list)
    return np.stack((u_list, v_list))


if __name__ == '__main__':

    theta = 5.0

    print(frank_coupla([0.1, 0.5], theta))

    np.random.seed(1)
    sim_count = 1000
    frank_copula_sim = generate_frank_coupla_2d(theta, sim_count)

    plt.plot(frank_copula_sim[0], frank_copula_sim[1], 'r+')
    plt.axis([0,1,0,1])
    plt.show()

