import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

annualization_factor = 250
path_steps = annualization_factor * 4
rho = 0.2
sigma_M = 0.15
sigma_A = 0.20
r_M = 0.3
beta = 0.8
M0 = 100.0
A0 = 100.0

dt = 1.0 / annualization_factor

np.random.seed(2231)
white_noise = np.random.normal(0, 1, (2, path_steps))

correlation_matrix = np.array([[1, rho],
                               [rho, 1]])

cholesky_decomp = numpy.linalg.cholesky(correlation_matrix)
correlated_wiener_process = (np.sqrt(dt) * cholesky_decomp.dot(white_noise)).cumsum(axis=1)

correlated_wiener_process_with_timestep = np.vstack((np.arange(correlated_wiener_process.shape[1])+1, correlated_wiener_process))

def model_func(w):
    timestep = w[0]
    t = timestep * dt
    W_M = w[1]
    W_t = w[2]
    M_t = M0 * np.exp((r_M - 0.5 * sigma_M ** 2) * t + sigma_M * W_M)
    A_t = A0 * np.exp((beta * r_M - 0.5 * (beta**2) * (sigma_M**2) - 0.5 * sigma_A**2 - beta * sigma_A * sigma_M * rho) * t + beta * sigma_M * W_M + sigma_A* W_t)
    return np.array([t, M_t, A_t])


simulation_path =np.apply_along_axis(model_func, 0, correlated_wiener_process_with_timestep)

# plot the result
x = simulation_path[0,:]
for y_ind in range(1, simulation_path.shape[0]):
    y = simulation_path[y_ind,:]
    plt.plot(x,y, label=f'{y_ind}', linewidth=1)
plt.title(r'$ \rho = {0}, \beta = {1}, r_M = {2}, \sigma_M = {3}, \sigma_A = {4} $'.format(rho, beta, r_M, sigma_M, sigma_A))
plt.ylim(bottom=0.0)
plt.legend()
plt.grid()
plt.show()
