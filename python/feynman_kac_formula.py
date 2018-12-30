from pde_finite_difference_solver import BackwardParabolicPde

class FeynmanKacConverter(BackwardParabolicPde):
    def __init__(self, drift_diffusion_process, r_term):
        BackwardParabolicPde.__init__(self, drift_diffusion_process.mu, self.__half_diffusion_square, r_term)
        self._process = drift_diffusion_process

    def __half_diffusion_square(self, t, x):
        return 0.5 * self._process.sigma(t, x) ** 2.0