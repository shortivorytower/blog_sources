import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    steps = 250

    T = 1.0
    mu = 0.02
    sigma = 0.25

    dt = T / steps

    # the only source of randomness in the GBM SDE
    np.random.seed(0)
    dW = np.random.normal(0, dt, steps)

    # simulate one path using discretized SDE
    S = np.zeros(steps + 1)
    S[0] = 80.0
    for t in xrange(0, steps):
        # dSt = mu * St * dt + sigma * St * dWt
        dSt = mu * S[t] * dt + sigma * S[t] * dW[t]
        S[t + 1] = S[t] + dSt

    print 'Discretized SDE : Simulated Path, final S =', S[steps]

    # get the same final value using close form GBM solution.
    # S_T = S0 * exp( (mu - sigma^2 * 0.5) * T + sigma * W_T )
    W_T = np.sum(dW)
    S_T = S[0] * np.exp((mu - sigma * sigma * 0.5) * T + sigma * W_T)
    print 'Closed Form Solution: Simulated Path, final S =', S_T

    # plt.plot(S)
    # plt.grid()
    # plt.show()
