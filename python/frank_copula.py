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


if __name__ == '__main__':
    print(frank_coupla([0.1, 0.5], 5))
