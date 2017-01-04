import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def simulate_sde(count, T, mu, sigma, S0, steps):
    result = np.zeros(count)
    dt = T / steps
    S = np.zeros(steps + 1)
    for i in xrange(count):
        # simulate one path using discretized SDE
        dW = np.random.normal(0, np.sqrt(dt), steps)
        S[0] = S0
        for t in xrange(0, steps):
            # dSt = mu * St * dt + sigma * St * dWt
            dSt = mu * S[t] * dt + sigma * S[t] * dW[t]
            S[t + 1] = S[t] + dSt
        result[i] = S[steps]
    return result


def simulate_closed_form(count, T, mu, sigma, S0):
    result = np.zeros(count)
    for i in xrange(count):
        W_T = np.random.normal(0, np.sqrt(T), 1)
        S_T = S0 * np.exp((mu - sigma * sigma * 0.5) * T + sigma * W_T)
        result[i] = S_T
    return result


if __name__ == '__main__':
    np.random.seed(0)
    steps = 2500
    T = 1.0
    mu = 0.08
    sigma = 0.25
    S0 = 80.0

    sde_result = simulate_sde(50000, T, mu, sigma, S0, steps)

    print np.average(sde_result)
    print np.var(sde_result)
    print stats.skew(sde_result)
    print stats.kurtosis(sde_result)

    closed_form_result = simulate_closed_form(50000, T, mu, sigma, S0)
    print np.average(closed_form_result)
    print np.var(closed_form_result)
    print stats.skew(closed_form_result)
    print stats.kurtosis(closed_form_result)



    # plt.hist(sde_result, bins=50)
    # plt.show()

    '''
    # get the same final value using close form GBM solution.
    # S_T = S0 * exp( (mu - sigma^2 * 0.5) * T + sigma * W_T )
    W_T = np.sum(dW)
    S_T = S[0] * np.exp((mu - sigma * sigma * 0.5) * T + sigma * W_T)
    print 'Closed Form Solution: Simulated Path, final S =', S_T
    '''

    # plt.plot(S)
    # plt.grid()
    # plt.show()
