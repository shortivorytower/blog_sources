from pde_finite_difference_solver import BackwardParabolicPde

class FeynmanKacConverter(BackwardParabolicPde):
    def __init__(self, drift_diffusion_process, r_term):
        BackwardParabolicPde.__init__(self, drift_diffusion_process.mu, self.__half_diffusion_square, r_term)
        self._process = drift_diffusion_process

    def __half_diffusion_square(self, t, x):
        return 0.5 * self._process.sigma(t, x) ** 2.0


class DriftDiffusionProcess:
    def __init__(self, mu_term, sigma_term):
        ''' Encapsulate the stochastic process
        dXt = mu(t,x) dt + sigma(t,x) dWt
        '''
        self._mu_term = mu_term
        self._sigma_term = sigma_term
    @property
    def mu(self):
        return self._mu_term
    @property
    def sigma(self):
        return self._sigma_term        