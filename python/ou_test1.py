import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from scipy.stats.mstats import kruskalwallis


def simulate_sde_once(T, theta, mu, sigma, X0, task_id):
    np.random.seed(task_id)
    steps = 250
    dt = T / steps
    X = np.zeros(steps + 1)
    # simulate one path using discretized SDE
    dW = np.random.normal(0, np.sqrt(dt), steps)
    X[0] = X0
    for t in xrange(0, steps):
        # dXt = theta * (mu - Xt) * dt + sigma * dWt
        dXt = theta * (mu - X[t]) * dt + sigma * dW[t]
        X[t + 1] = X[t] + dXt
    return X[steps]


def simulate_solution_once(T, theta, mu, sigma, X0, task_id):
    np.random.seed(task_id)
    steps = 250
    dt = T / steps
    dW = np.random.normal(0, np.sqrt(dt), steps)

    integral_term = 0.0
    for s in xrange(steps):
        integral_term += np.exp(-theta * (T - s * dt)) * dW[s]

    X_T = X0 * np.exp(-theta * T) + mu * (1.0 - np.exp(-theta * T)) + sigma * integral_term
    return X_T


def simulate_batch(count, sim_once_func):
    pool = Pool(processes=24)
    result_array = pool.map(sim_once_func, [i for i in xrange(count)])
    pool.close()
    pool.join()
    return np.array(result_array)


if __name__ == '__main__':
    T = 1.0
    mu = 1.2
    theta = 1.5
    sigma = 3.0
    X0 = 15.0

    count = 20000

    # print 'expected', X0 * np.exp(-theta * T) + mu * (1.0 - np.exp(-theta * T))
    # print 'variance', sigma * sigma / 2.0 / theta

    print 'Simulating OU SDE'
    sde_result = simulate_batch(count, partial(simulate_sde_once, T, theta, mu, sigma, X0))
    print 'Mean', np.average(sde_result)
    print 'Variance', np.var(sde_result)
    print 'Skewness', stats.skew(sde_result)
    print 'Kurtosis', stats.kurtosis(sde_result)

    print 'Simulating OU Solutions'
    solution_form_result = simulate_batch(count, partial(simulate_solution_once, T, theta, mu, sigma, X0))
    print 'Mean', np.average(solution_form_result)
    print 'Variance', np.var(solution_form_result)
    print 'Skewness', stats.skew(solution_form_result)
    print 'Kurtosis', stats.kurtosis(solution_form_result)

    print 'Kruskal-Wallis test'
    h_stat, p_value = kruskalwallis(sde_result, solution_form_result)
    print 'H Statistics', h_stat, 'P-value', p_value

    fig = plt.figure()
    sp1 = fig.add_subplot(211)
    sp1.hist(sde_result, bins=50)
    sp1.set_xlim([-4, 12])
    sp1.grid()
    sp1.set_title('Simulations with Discretized OU SDE')

    sp2 = fig.add_subplot(212)
    sp2.hist(solution_form_result, bins=50)
    sp2.set_xlim([-4, 12])
    sp2.grid()
    sp2.set_title('Simulations with OU Solution Form')

    plt.show()
