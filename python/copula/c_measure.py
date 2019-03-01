import numpy as np
from scipy.optimize import brentq


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


def level_curve(generator_func, inverted_generator_func, u, t):
    """Get the level curve value of 'v' from given u at level t
    """
    v = inverted_generator_func(generator_func(t) - generator_func(u))
    return v


if __name__ == '__main__':
    theta = 3.0
    t = 0.33

    gumbel_generator = lambda u: np.power(-np.log(u), theta)
    gumbel_inverted_generator = lambda y: np.exp(-np.power(y, 1.0 / theta))
    c_measure = lambda r: r - r / theta * np.log(r)

    print('theta = {0}, t = {1},  Theoretical Gumbel C-Measure = {2}'.format(theta, t, c_measure(t)))

    np.random.seed(1)
    sim_count = 500000
    sim_result = generate_gumbel_copula_2d(theta, sim_count)
    u_list = sim_result[0]
    v_list = sim_result[1]
    gumbel_level_curve = level_curve(gumbel_generator, gumbel_inverted_generator, u_list, t)

    # set all negative values and NA to 1
    # so that we accept any simulated v values (which must be between 0 and 1)
    gumbel_level_curve[np.isnan(gumbel_level_curve)] = 1.0
    gumbel_level_curve[gumbel_level_curve < 0] = 1.0

    in_range_count = (v_list <= gumbel_level_curve).sum()

    print('C(u,v)<={0} : hit count={1} out of {2}'.format(t, in_range_count, sim_count))
    print('Simulated C-Measure = {0}'.format(in_range_count / sim_count))
