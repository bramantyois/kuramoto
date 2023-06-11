import numpy as np

from neurolib.utils.collections import dotdict


def loadDefaultParams(seed=None):
    params = dotdict({})

    ### runtime parameters
    params.dt = 0.1 
    params.duration = 2000  
    np.random.seed(seed)  
    params.seed = seed

    params.N = 64
    params.k = 16

    params.omega = np.random.normal(loc=np.pi, scale=np.pi, size=(params.N,))

    params.tau_ou = 5.0  # ms Timescale of the Ornstein-Uhlenbeck noise process
    params.sigma_ou = 0.0  # mV/ms/sqrt(ms) noise intensity
    params.theta_ou_mean = 0.0  # mV/ms (OU process) [0-5]

    # init values
    params.theta_init = np.random.uniform(low=0, high=2*np.pi, size=(params.N, 1))

    #params.theta_ou = np.zeros((params.N,))

    return params
    
