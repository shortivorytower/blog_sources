import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from scipy.stats.mstats import kruskalwallis


def simulate_sde_once(T, mu, sigma, S0, task_id):
    np.random.seed(task_id)
    steps = 250
    dt = T / steps
    S = np.zeros(steps + 1)
    # simulate one path using discretized SDE
    dW = np.random.normal(0, np.sqrt(dt), steps)
    S[0] = S0
    for t in xrange(0, steps):
        # dSt = mu * St * dt + sigma * St * dWt
        dSt = mu * S[t] * dt + sigma * S[t] * dW[t]
        S[t + 1] = S[t] + dSt
    return S[steps]


def simulate_closed_form_once(T, mu, sigma, S0, task_id):
    np.random.seed(task_id)
    # simulate the S_T directly with the closed form solution.
    W_T = np.random.normal(0, np.sqrt(T), 1)[0]
    S_T = S0 * np.exp((mu - sigma * sigma * 0.5) * T + sigma * W_T)
    return S_T


def simulate_batch(count, sim_once_func):
    pool = Pool(processes=24)
    result_array = pool.map(sim_once_func, [i for i in xrange(count)])
    pool.close()
    pool.join()
    return np.array(result_array)


if __name__ == '__main__':
    T = 1.0
    mu = 0.08
    sigma = 0.25
    S0 = 80.0

    count = 20000

    print 'Simulating GBM SDE'
    sde_result = simulate_batch(count, partial(simulate_sde_once, T, mu, sigma, S0))
    print 'Mean', np.average(sde_result)
    print 'Variance', np.var(sde_result)
    print 'Skewness', stats.skew(sde_result)
    print 'Kurtosis', stats.kurtosis(sde_result)

    print 'Simulating GBM Closed Form'
    closed_form_result = simulate_batch(count, partial(simulate_closed_form_once, T, mu, sigma, S0))
    print 'Mean', np.average(closed_form_result)
    print 'Variance', np.var(closed_form_result)
    print 'Skewness', stats.skew(closed_form_result)
    print 'Kurtosis', stats.kurtosis(closed_form_result)

    print 'Kruskal-Wallis test'
    h_stat, p_value = kruskalwallis(sde_result, closed_form_result)
    print 'H Statistics', h_stat, 'P-value', p_value

    fig = plt.figure()
    sp1 = fig.add_subplot(211)
    sp1.hist(sde_result, bins=50)
    sp1.set_xlim([0, 250])
    sp1.set_ylim([0, 0.1 * count])
    sp1.grid()
    sp1.set_title('Simulations with Discretized GBM SDE')

    sp2 = fig.add_subplot(212)
    sp2.hist(closed_form_result, bins=50)
    sp2.set_xlim([0, 250])
    sp2.set_ylim([0, 0.1 * count])
    sp2.grid()
    sp2.set_title('Simulations with GBM Closed Form')

    plt.show()
