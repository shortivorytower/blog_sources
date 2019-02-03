import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def clayton_copula_pdf_inverse(u, theta, t):
    return np.power(np.power(u, -theta) * (np.power(t, -theta / (theta + 1)) - 1.0) + 1.0, -1.0 / theta)


def generate_clayton_copula_2d(theta, sim_count=50):
    u_list = np.random.uniform(size=sim_count)
    t_list = np.random.uniform(size=sim_count)
    v_list = np.vectorize(lambda u, t: clayton_copula_pdf_inverse(u, theta, t))(u_list, t_list)
    return np.stack((u_list, v_list))


if __name__ == '__main__':

    np.random.seed(1)
    sim_count = 500

    u_col = np.array([], dtype=np.float32)
    v_col = np.array([], dtype=np.float32)
    theta_col = np.array([], dtype=np.float32)

    thetas = [-0.5, 1.0, 3.0, 10.0]

    for theta in thetas:
        sim_result = generate_clayton_copula_2d(theta, sim_count)
        theta_col = np.append(theta_col, np.full_like(sim_result[0], theta))
        u_col = np.append(u_col, sim_result[0])
        v_col = np.append(v_col, sim_result[1])

    df = pd.DataFrame({'Theta': theta_col, 'U': u_col, 'V': v_col})

    sns.set(style='ticks')
    g = sns.FacetGrid(df, col='Theta', margin_titles=True, col_wrap=4, size=2.5)
    g.map(plt.scatter, 'U', 'V', s=1)
    g.set(xlim=(0, 1), ylim=(0, 1))
    g.fig.suptitle('Clayton Copula')
    plt.show()
