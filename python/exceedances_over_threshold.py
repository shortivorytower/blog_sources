import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    source_distribution = stats.gamma(a=2.5)
    threshold = 6.5
    # source distribution of the data has nothing to do beyond this line.
    prob_gt_threshold = 1.0 - source_distribution.cdf(threshold)
    print(r'Pr(X>={0}) = {1}'.format(threshold, prob_gt_threshold))

    plot_k_range = np.arange(0, 21)
    number_of_trials = 200
    binomial_dist_k = np.array([stats.binom.pmf(k, number_of_trials, prob_gt_threshold) for k in plot_k_range])
    poisson_dist_k = np.array([stats.poisson.pmf(k, number_of_trials * prob_gt_threshold) for k in plot_k_range])

    width = 0.35
    fig, ax = plt.subplots()
    plt.grid(True)
    binomial_bars = ax.bar(plot_k_range - width / 2, binomial_dist_k, width, color='r')
    poisson_bars = ax.bar(plot_k_range + width / 2, poisson_dist_k, width, color='y')
    ax.set_ylabel(r'$\Pr(\sum I(X_i \geq {0}) = k) $'.format(threshold))
    ax.set_xlabel(r'# of Exceedances $k$')
    ax.legend((binomial_bars[0], poisson_bars[0]), ('Binomial', 'Poisson'))
    ax.set_title(r'Probability of # of Exceedances over threshold {0} in {1} trials'.format(threshold, number_of_trials))
    plt.show()
