import numpy as np
import numba


def timeIntegration(params):
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
    N = params["num_osc"]  # number of oscillators
    omega = params["omega"]  # frequencies of oscillators
    k = params["k"]  # coupling strength

    # ornstein uhlenbeck noise param
    tau_ou = params["tau_ou"]  # noise time constant
    sigma_ou = params["sigma_ou"]  # noise strength
    x_ou_mean = params["x_ou_mean"]
    
    # ---------------------------
    # Seed the RNG
    # ---------------------------
    np.random.seed(RNGseed)
    
    # ---------------------------
    # time integration param
    # ---------------------------
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)
    sqrt_dt = np.sqrt(dt)

    # ---------------------------
    # Placeholders
    # ---------------------------
    x_s = np.zeros((N, len(t)+1))
    
    x_ou = params['x_ou']
    noise_x_ou = np.random.standard_normal(size=(N, len(t)))
    
    # ---------------------------
    # simulation param
    # ---------------------------
    t = np.arange(1, round(duration, 6) / dt + 1) * dt
    
    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data

    # ---------------------------
    # initial values
    # ---------------------------   

    # initial values for thetas
    x_s[:,0] = params['xs_init']
    
    timeIntegration_njit_elementwise(
        t,
        dt,
        sqrt_dt,
        N,
        omega,
        k,
        x_s,
        tau_ou,
        sigma_ou,
        x_ou,
        x_ou_mean,
        noise_x_ou,
    )


@numba.njit
def timeIntegration_njit_elementwise(
    t, 
    dt, 
    sqrt_dt,
    N,
    omega,
    k,
    
    x_s,
    
    tau_ou,
    sigma_ou,
    x_ou,
    x_ou_mean,
    noise_x_ou,
):
    """
    Kuramoto Model 
    """
    k_n = k/N
    for i in range(1, len(t)):
        # Ornstein-Uhlenbeck process
        x_ou[i] = x_ou[i-1] + (x_ou_mean-x_ou[i-1]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_x_ou[:, i-1]
        
        x_rhs = np.zeros((N, 1))
        for n, m in np.ndindex((N, N)):
            x_rhs[n] += k_n * np.sin(x_s[m,n-1] - x_s[n,n-1])

        x_s[:,i] = x_s[:,i-1] + dt * (
            omega +  
            x_rhs + 
            x_ou[i]) 

    return t, x_s, x_ou