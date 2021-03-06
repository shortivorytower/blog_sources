import numpy as np
from scipy.stats import norm

# these packages are required only for testing.
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

class BackwardParabolicPde:
    def __init__(self, mu_term, sigma_term, r_term):
        '''Create a backward parabolic PDE that can be computed with finite difference method. 
        All terms objects are lambda functions taking t and x argument
        '''
        self._mu_term = mu_term
        self._sigma_term = sigma_term
        self._r_term = r_term
    @property
    def mu(self):
        return self._mu_term
    @property
    def sigma(self):
        return self._sigma_term
    @property
    def r(self):
        return self._r_term

class FiniteDifferenceSolver:
    def __init__(self, pde, boundary_cond_T, boundary_cond_max_X, boundary_cond_min_X):
        self._pde = pde
        self._boundary_cond_T = boundary_cond_T
        self._boundary_cond_max_X = boundary_cond_max_X
        self._boundary_cond_min_X = boundary_cond_min_X
    
    def solve(self, x_max, x_min, t_max, steps_t, steps_x):
        dt = t_max / (steps_t - 1)
        dx = (x_max - x_min) / (steps_x +1)

        # clear the result storage
        solution_grid = []

        # n is time index, at 0 means t = t_max, at steps_t - 1 means t = 0
        # we are stepping from T, T-1.. all the way to time 0.
        for n in range(steps_t):
            valuation_layer = np.zeros(steps_x)
            a = np.zeros(steps_x)
            b = np.zeros(steps_x)
            c = np.zeros(steps_x)
            d = np.zeros(steps_x)

            # space index, at j = 0 means x = x_min + dx, j =spacesteps-1 means x = x_max - dx
            for j in range(steps_x):
                current_x = x_min + (j+1) * dx
                current_t = (steps_t - 1 - n) * dt

                if n == 0:
                    # at max t, the result layer is just the boundary condition at T
                    valuation_layer[j] = self._boundary_cond_T(current_t, current_x)
                else:
                    # if we are not at max t, we need to get the mu / sigma / r terms 
                    # and solve the tridiagonal matrix problem
                    mu_func = self._pde.mu
                    sigma_func = self._pde.sigma
                    r_func = self._pde.r

                    # compute the mu, sigma and r term given t and x
                    mu = mu_func(current_t, current_x)
                    sigma = sigma_func(current_t, current_x)
                    r = r_func(current_t, current_x)

                    # fill in a, b, c arrays
                    a[j] = -mu / (4.0 * dx) + sigma / (2.0 * dx * dx)
                    b[j] = -1.0 / dt - sigma / (dx * dx) - r / 2.0
                    c[j] = mu / (4.0 * dx) + sigma / (2.0 * dx * dx)

                    # get the last computed layer, in fact we don't need to keep all layers
                    prev_layer = solution_grid[-1]

                    # prepare the d array
                    prev_t = (steps_t - 1 - (n-1)) * dt

                    if j > 0 and j < steps_x - 1:
                        # if we are in between max and min x
                        d[j] = prev_layer[j-1] * (mu/(4.0*dx) - sigma / (2.0 * dx * dx)) + prev_layer[j] * (-1.0 / dt + sigma / (dx*dx) + r/2.0) + prev_layer[j+1] * (-mu/(4.0*dx) - sigma / (2.0 * dx*dx))
                    elif j == 0:
                        # if we are at min x
                        prev_v_j_minus1 = self._boundary_cond_min_X(prev_t, x_min)
                        original_d = prev_v_j_minus1 * (mu/(4.0*dx) - sigma / (2.0 * dx * dx)) + prev_layer[j] * (-1.0 / dt + sigma / (dx*dx) + r/2.0) + prev_layer[j+1] * (-mu/(4.0*dx) - sigma / (2.0 * dx*dx))
                        current_v_j_minus1 = self._boundary_cond_min_X(current_t, x_min)
                        d[j] = original_d - a[j] * current_v_j_minus1
                    elif j == steps_x - 1:
                        # if we are at max X
                        prev_v_j_plus1 = self._boundary_cond_max_X(prev_t, x_max)
                        original_d = prev_layer[j-1] * (mu/(4.0*dx) - sigma / (2.0 * dx * dx)) + prev_layer[j] * (-1.0 / dt + sigma / (dx*dx) + r/2.0) + prev_v_j_plus1 * (-mu/(4.0*dx) - sigma / (2.0 * dx*dx))
                        current_v_j_plus1 = self._boundary_cond_max_X(current_t, x_max)
                        d[j] = original_d - c[j] * current_v_j_plus1
            
            # now a, b, c, d arrays are all ready
            if n > 0:
                # if it is not at max T we need to solve the tridiagonal matrix
                valuation_layer = self.solve_tridiagonal_matrix(a, b, c, d)

            # append the result layer for this time step
            solution_grid.append(valuation_layer)
        
        # return the solution object.
        return FiniteDifferenceSolution(solution_grid, x_max, x_min, t_max, dt, dx, steps_t, steps_x,
                                        self._boundary_cond_max_X, self._boundary_cond_min_X)


    def solve_tridiagonal_matrix(self, a, b, c, r):
        '''Implement the Thomas Algorithm that solves the tridiagonal matrix problem.
        Refers to http://www.industrial-maths.com/ms6021_thomas.pdf for details and the naming convention of 
        the variables a, b, c, r, x, rho, and gamma.
        Note that all array indices are zero based in this piece of code. Value of a[1] means it is a[2] in the paper.
        '''
        if len(a)!=len(b) or len(b)!=len(c) or len(c)!=len(r):
            raise ValueError('input array size mismatched')

        rho = np.zeros(len(a))
        gamma = np.zeros(len(a)-1)
        # stage 1: fill up the rho and gamma
        for i in range(len(a)):
            if i==0:
                gamma[0] = c[0]/b[0]
                rho[0]=r[0]/b[0]
            else:
                if i<len(a)-1:
                    gamma[i] = c[i] / (b[i]-a[i]*gamma[i-1])
                rho[i] = (r[i] - a[i] * rho[i-1]) / (b[i]-a[i]*gamma[i-1])
        # stage 2: compute the result vector x
        x = np.zeros(len(a))
        x[-1]=rho[-1]
        for i in range(len(x)-2, -1, -1):
            x[i]=rho[i] - gamma[i] * x[i+1]
        
        return x

# helper class to locate the actual PDE value at T0
class FiniteDifferenceSolution:
    def __init__(self, solution_grid, x_max, x_min, t_max, dt, dx, steps_t, steps_x, boundary_cond_max_X, boundary_cond_min_X):
        self._boundary_cond_max_X = boundary_cond_max_X
        self._boundary_cond_min_X = boundary_cond_min_X
        self._x_max = x_max
        self._x_min = x_min
        self._solution_grid = solution_grid
        self._dt = dt
        self._dx = dx
        self._steps_t = steps_t
        self._steps_x = steps_x
        self._t_max = t_max

    def solution_at_t0(self, x0):
        grid_at_t0 = self._solution_grid[-1]

        # fill in the arrays of X and V points and then perform linear interpolation.
        v_pts = np.zeros(len(grid_at_t0))
        x_pts = np.zeros(len(grid_at_t0))
        for j in range(len(v_pts)):
            x_pts[j] = self.convert_j_to_x(j)
            v_pts[j] = grid_at_t0[j]

        if x0<=self._x_min:
            return self._boundary_cond_min_X(0.0, x0)
        elif x0>=self._x_max:
            return self._boundary_cond_max_X(0.0, x0)
        return np.interp(x0, x_pts, v_pts)

    @property
    def solution_grid(self):
        return self._solution_grid

    def convert_n_to_t(self, n):
        return (self._steps_t -1 - n) * self._dt

    def convert_j_to_x(self, j):
        return self._x_min + (j+1)*self._dx



if __name__ == '__main__':
    
    risk_free = 0.01
    vol = 0.4
    strike = 5.5
    tmat = 1.0
    spot = 5.0

    mu_term = lambda t, x: risk_free * x
    sigma_term = lambda t,x: 0.5 * vol * vol * x *x
    r_term = lambda t, x: risk_free

    pde = BackwardParabolicPde(mu_term, sigma_term, r_term)

    # specify call payoff boundary condition
    boundary_cond_T = lambda t, x: max(x - strike, 0.0)
    boundary_cond_max_x = lambda t, x: np.exp(-risk_free * (tmat-t)) * boundary_cond_T(t,x)
    boundary_cond_min_x = lambda t, x: np.exp(-risk_free * (tmat-t)) * boundary_cond_T(t,x)

    pde_solver = FiniteDifferenceSolver(pde, boundary_cond_T, boundary_cond_max_x, boundary_cond_min_x)
    
    start_time = datetime.now()
    solution = pde_solver.solve(10, 0, tmat, 50, 200)
    end_time = datetime.now()
    pde_result = solution.solution_at_t0(spot)
    print('PDE solution:', pde_result)
    print('elapse time:', end_time - start_time)

    d1 = (np.log(spot/strike) + (risk_free + 0.5 * vol * vol) * tmat)/(vol * np.sqrt(tmat))
    d2 = (np.log(spot/strike) + (risk_free - 0.5 * vol * vol) * tmat)/(vol * np.sqrt(tmat))
    call_price = spot*norm.cdf(d1) - strike * np.exp(-risk_free*tmat) * norm.cdf(d2)
    print('Close Form:', call_price)

    t_list = []
    x_list = []
    v_list = []
    for n in range(len(solution.solution_grid)):
        for j in range(len(solution.solution_grid[n])):
            t_list.append(solution.convert_n_to_t(n))
            x_list.append(solution.convert_j_to_x(j))
            v_list.append(solution.solution_grid[n][j])

    df = pd.DataFrame({'x': x_list, 't': t_list, 'V': v_list})

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_zlabel('V', fontsize=16)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('t', fontsize=16)
    surf = ax.plot_trisurf(df.x, df.t, df.V, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
