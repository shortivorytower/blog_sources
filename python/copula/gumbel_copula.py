import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import seaborn as sns


def generate_gumbel_copula_2d(theta, sim_count=50):
    gumbel_generator = lambda u: np.power(-np.log(u), theta)
    gumbel_inverted_generator = lambda y: np.exp(-np.power(y, 1.0 / theta))
    c_measure = lambda r: r - r / theta * np.log(r)
    inverted_c_measure = lambda t: brentq(lambda x: c_measure(x) - t, 0.0000000000000001, 1)

    s_list = np.random.uniform(size=sim_count)
    t_list = np.random.uniform(size=sim_count)
    w_list = np.vectorize(inverted_c_measure)(t_list)

    gen_w_list = gumbel_generator(w_list)

    u_list = np.vectorize(gumbel_inverted_generator)(s_list * gen_w_list)
    v_list = np.vectorize(gumbel_inverted_generator)((1 - s_list) * gen_w_list)

    return np.stack((u_list, v_list))


if __name__ == '__main__':

    np.random.seed(1)
    sim_count = 500

    u_col = np.array([], dtype=np.float32)
    v_col = np.array([], dtype=np.float32)
    theta_col = np.array([], dtype=np.float32)

    # theta has to be >= 1
    thetas = [1.0, 3.0, 5.0, 10.0]

    for theta in thetas:
        sim_result = generate_gumbel_copula_2d(theta, sim_count)
        theta_col = np.append(theta_col, np.full_like(sim_result[0], theta))
        u_col = np.append(u_col, sim_result[0])
        v_col = np.append(v_col, sim_result[1])

    df = pd.DataFrame({'Theta': theta_col, 'U': u_col, 'V': v_col})

    sns.set(style='ticks')
    g = sns.FacetGrid(df, col='Theta', margin_titles=True, col_wrap=4, size=2.5)
    g.map(plt.scatter, 'U', 'V', s=1)
    g.set(xlim=(0, 1), ylim=(0, 1))
    g.fig.suptitle('Gumbel Copula')
    plt.show()
