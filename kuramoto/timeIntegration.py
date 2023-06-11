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
    N = params["N"]  # number of oscillators
    omega = params["omega"]  # frequencies of oscillators
    k = params["k"]  # coupling strength

    # ornstein uhlenbeck noise param
    tau_ou = params["tau_ou"]  # noise time constant
    sigma_ou = params["sigma_ou"]  # noise strength
    theta_ou_mean = params["theta_ou_mean"]
    
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
    theta_s = np.zeros((N, len(t)+1))
    
    theta_ou = np.zeros_like(theta_s)
    noise_theta_ou = np.random.standard_normal(size=(N, len(t)))
    
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
    theta_s[:,[0]] = params['theta_init']
    
    # ------------------------------------------------------------------------
    # global coupling parameters
    # ------------------------------------------------------------------------
#TO CHECK, added from other models

    # Connectivity matrix
    # Interareal relative coupling strengths (values between 0 and 1), Cmat(i,j) connnection from jth to ith
    Cmat = params["Cmat"]
    N = len(Cmat)  # Number of nodes
    
    k = params["k"]  # coupling strength
    #K_gl = params["K_gl"]  # global coupling strength ?
    
    # Interareal connection delay
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]

    if N == 1:
        Dmat = np.zeros((N, N))
    else:
        # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
        Dmat = mu.computeDelayMatrix(lengthMat, signalV)
        # no self-feedback delay
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt

    # do we need here Additive or diffusive coupling scheme ?
   
         
    timeIntegration_njit_elementwise(
        t,
        dt,
        sqrt_dt,
        N,
        omega,
        k,
        Cmat,
        Dmat,
        theta_s,
        tau_ou,
        sigma_ou,
        theta_ou,
        theta_ou_mean,
        noise_theta_ou,
    )


@numba.njit
def timeIntegration_njit_elementwise(
    t, 
    dt, 
    sqrt_dt,
    N,
    omega,
    k,
    Cmat,
    Dmat,
    theta_s,
    tau_ou,
    sigma_ou,
    theta_ou,
    theta_ou_mean,
    noise_theta_ou,
):
    """
    Kuramoto Model 
    """
    k_n = k/N

    for i in range(1, len(t)):
            # Ornstein-Uhlenbeck process
            theta_ou[i] = theta_ou[i-1] + (theta_ou_mean - theta_ou[i-1]) * dt / tau_ou 
                                    + sigma_ou * np.sqrt(dt) * noise_theta_ou[:, i-1]

            theta_rhs = np.zeros((N, 1))
            for n, m in np.ndindex((N, N)):
                theta_rhs[n] += k_n * Cmat[n, m] * np.sin(theta_s[m, n-1] - theta_s[n, n-1] - Dmat[n, m])

            theta_s[:, i] = theta_s[:, i-1] + dt * (omega + theta_rhs)

        return t, theta_s, theta_ou
