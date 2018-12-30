import numpy as np
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt


def simulate_sde_once(T, theta, mu, sigma, X0, stop_loss, stop_gain, task_id):
    np.random.seed(task_id)
    steps = 250
    dt = T / steps
    X = np.zeros(steps + 1)
    # simulate one path using discretized SDE
    dW = np.random.normal(0, np.sqrt(dt), steps)
    X[0] = X0
    for t in range(0, steps):
        dXt = theta * (mu - X[t]) * dt + sigma * dW[t]
        new_X = X[t] + dXt
        X[t + 1] = new_X
        if new_X >= stop_gain:
            return stop_gain - X0
        elif new_X <= stop_loss:
            return stop_loss - X0
    return X[steps] - X0

def simulate_sde_square_once(T, theta, mu, sigma, X0, stop_loss, stop_gain, task_id):
    np.random.seed(task_id)
    steps = 250
    dt = T / steps
    X = np.zeros(steps + 1)
    # simulate one path using discretized SDE
    dW = np.random.normal(0, np.sqrt(dt), steps)
    X[0] = X0
    for t in range(0, steps):
        dXt = theta * (mu - X[t]) * dt + sigma * dW[t]
        new_X = X[t] + dXt
        X[t + 1] = new_X
        if new_X >= stop_gain:
            return (stop_gain - X0)**2
        elif new_X <= stop_loss:
            return (stop_loss - X0)**2
    return (X[steps] - X0)**2

def simulate_batch(paths_count, sim_once_func):
    pool = Pool(processes=24)
    result_array = pool.map(sim_once_func, [i for i in range(paths_count)])
    pool.close()
    pool.join()
    return np.array(result_array)


if __name__ == '__main__':
    T = 1.0
    mu = 3
    theta = 1.5
    sigma = 1.0
    X0 = 1.0
    stop_loss = X0-3.2
    stop_gain = X0+2.25

    count = 20000

    print('Simulating OU SDE')
    sde_result = simulate_batch(count, partial(simulate_sde_once, T, theta, mu, sigma, X0, stop_loss, stop_gain))

    ptf_mu = np.average(sde_result)
    ptf_sigma = np.std(sde_result)
    print('Mean', ptf_mu)
    print('std', ptf_sigma)
    print('sharpe', ptf_mu / ptf_sigma)

    sde_square_result = simulate_batch(count, partial(simulate_sde_square_once, T, theta, mu, sigma, X0, stop_loss, stop_gain))
    print('std using another method', np.sqrt(np.average(sde_square_result) - ptf_mu**2))


    fig = plt.figure()
    sp1 = fig.add_subplot(111)
    sp1.hist(sde_result, bins=50)
    sp1.set_xlim([-3, 5])
    sp1.grid()
    sp1.set_title('Simulations with Discretized OU SDE')

    plt.show()
