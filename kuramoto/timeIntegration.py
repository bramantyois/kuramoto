import numpy as np
import numba

from neurolib.utils import model_utils as mu


def timeIntegration(params)-> np.ndarray:
    """
    setting up parameters for time integration
    
    :param params: Parameter dictionary of the model
    :type params: dict

    :return: Integrated activity of the model
    :rtype: (numpy.ndarray, )
    """

    dt = params["dt"]  # Time step for the Euler intergration (ms)
    duration = params["duration"]  # imulation duration (ms)
    RNGseed = params["seed"]  # seed for RNG

    # model parameters
    num_osc = params["num_osc"]  # number of oscillators
    theta = params["theta"]  # phases of oscillators
    omega = params["omega"]  # frequencies of oscillators
    k = params["k"]  # coupling strength

    # ornstein uhlenbeck noise param
    ou_tau = params["ou_tau"]  # noise time constant
    sigma_ou = params["sigma_ou"]  # noise strength
    
    # Initialization 
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)
    sqrt_dt = np.sqrt(dt)

    # Initializing noise. are params["x_ou"] and params["y_ou"] array?
    x_ou = params['x_ou']
    

