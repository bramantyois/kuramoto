import numpy as np
from kuramoto.utils import set_seed


def kuramoto(omega, t, k, theta, seed=None):
    """
    Kuramoto Model
    
    omega: frequencies of oscillators
    t: time steps (not used)
    k: coupling strength
    theta: phases of oscillators
    """
    set_seed(seed)
    n_osc = len(omega)
    d_omega_dt = omega.copy()

    for i, j in np.ndindex((n_osc, n_osc)):
        d_omega_dt[i] += np.sin(theta[j] - theta[i])

    d_omega_dt *= k / n_osc

    return d_omega_dt
