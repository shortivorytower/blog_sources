import numpy as np


def clayton_generator(theta, t):
    return 1.0 / theta * (np.power(t, -theta) - 1)


def differentiated_clayton_generator(theta, t):
    return -np.power(t, -theta - 1)


def clayton_c_measure(theta, t):
    return t - clayton_generator(theta, t) / differentiated_clayton_generator(theta, t)


def clayton_copula_pdf_inverse(u, theta, t):
    return np.power(np.power(u, -theta) * (np.power(t, -theta / (theta + 1)) - 1.0) + 1.0, -1.0 / theta)


def generate_clayton_copula_2d(theta, sim_count=50):
    u_list = np.random.uniform(size=sim_count)
    t_list = np.random.uniform(size=sim_count)
    v_list = np.vectorize(lambda u, t: clayton_copula_pdf_inverse(u, theta, t))(u_list, t_list)
    return np.stack((u_list, v_list))


def level_curve(generator_func, inverted_generator_func, u, t):
    """Get the level curve value of 'v' from given u at level t
    """
    v = inverted_generator_func(generator_func(t) - generator_func(u))
    return v



if __name__ == '__main__':

    theta = 3.0
    t = 0.35
    print('theta = {0}, t = {1},  Theoretical Clayton C-Measure = {2}'.format(theta, t, clayton_c_measure(theta, t)))

    clayton_generator = lambda t: np.power(t, -theta) - 1
    clayton_inverted_generator = lambda s: np.power(1.0 + s, -1.0 / theta)

    sim_count = 100000
    sim_result = generate_clayton_copula_2d(theta, sim_count)
    u_list = sim_result[0]
    v_list = sim_result[1]

    level_curve_v = level_curve(clayton_generator, clayton_inverted_generator, u_list, t)
    #print(u_list)
    #print(v_list)
    #print(level_curve_v)
    in_range_count = 0
    for v, level_v in zip(v_list, level_curve_v):
        if v <= level_v or np.isnan(level_v) or level_v < 0:
            in_range_count += 1

    print('C(u,v)<={0} : count={1}'.format(t,in_range_count))
    print('Simulated C-Measure = {0}'.format(in_range_count / sim_count))
