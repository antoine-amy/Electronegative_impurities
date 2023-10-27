import numpy as np
import scipy.constants as sc

# Constants


def arrhenius(D0, Ea, T):
    """
    Calculate the diffusion coefficient using the Arrhenius equation.
    From "Outgassing Model for Electronegative Impurites in Liquid Xenon for nEXO" 26-02-2020.
    
    Parameters:
    - D0: in cm2.s-1, diffusion constant at infinite temperature
    - Ea: in Joules, activation energy
    - T: in Kelvin, temperature
    
    Returns:
    - Diffusion coefficient D
    """
    return D0 * np.exp(-Ea / (sc.k * T))


def electron_lifetime(t, M, rho, n0, F, eta, R0, alpha, n_p=0):
    """
    Calculate the electron lifetime.
    From "Screening for Electronegative Impurities".
    
    Parameters:
    - t: in seconds, time
    - M: in kg, LXe mass
    - rho: in kg/liter, LXe density
    - n0: in ppb, initial impurity concentration
    - F: in liter/sec, xenon gas circulation flow rate
    - eta: purification efficiency
    - R0: in ppb liter/sec, total out-diffusion rate
    - alpha: in s, EXO-200 value, field dependant factor
    - n_p: in ppb, purifier output impurity concentration
    
    Returns:
    - Electron lifetime value
    """
    factor_exp = np.exp(-rho * eta * F * t / M)
    denominator = n0 * factor_exp + ((R0 + n_p * F) / (eta * F)) * (1 - factor_exp)

    # Guard against division by zero
    if denominator == 0:
        return 0

    return alpha / denominator


def XPM_electron_lifetime_fit(t, C_el, n0, R0, rho, M, n0_error=None, R0_error=None):
    """
    Calculate materials test XPM fit for electron lifetime.
    From "Screening for Electronegative Impurities".
    
    Parameters:
    - t: in seconds, time
    - C_el: in ppb/μs
    - n0: in ppb, initial impurity concentration
    - R0: in ppb liter/sec, total out-diffusion rate
    - rho: in kg/liter, LXe density
    - M: in kg, LXe mass
    - n0_error (optional): error associated with n0
    - R0_error (optional): error associated with R0
    
    Returns:
    - Main electron lifetime, and (optionally) lower and upper bounds due to errors
    """
    electron_lifetime = C_el / (n0 + R0 * rho * t / M)
    
    if n0_error is not None and R0_error is not None:
        electron_lifetime_upper = C_el / (n0 - n0_error + (R0 - R0_error) * rho * t / M)
        electron_lifetime_lower = C_el / (n0 + n0_error + (R0 + R0_error) * rho * t / M)
        return electron_lifetime, electron_lifetime_lower, electron_lifetime_upper
    
    return electron_lifetime



def steel_desorption(tau0, E, T, q0, t):
    """
    Calculate the steel desorption rate. From "Outgassing Model for Electronegative Impurites in Liquid Xenon for nEXO" 26-02-2020.
    
    Parameters:
    - tau0: in seconds, pre-exponential factor for relaxation time
    - E: in Joules, activation energy
    - T: in Kelvin, temperature
    - q0: initial absorbed impurities per unit area
    - t: in seconds, time
    
    Returns:
    - Steel desorption rate J
    """
    residence_time = tau0 * np.exp(E / sc.k * T) #the residence time of an impurity at an absorption site
    J = q0 / residence_time * np.exp(-t / residence_time)
    
    return J


def plastics_outgassing(c0, D0, Ea, T, d, t, max_iter=1000):
    """
    Computes the plastic outgassing rate using the diffusion equation solution. From "Outgassing Model for Electronegative Impurites in Liquid Xenon for nEXO" 26-02-2020.
    
    Parameters:
    - c0: 
    - D: 
    - d: in mm, thickness of the sample
    - t: in seconds, time
    - max_iter: maximum number of iterations for the summation (default is 1000)
    
    Returns:
    - Value of the plastic outgassing rate J
    """
    J = 0
    D=arrhenius(D0, Ea, T)
    for n in range(max_iter):
        term = np.exp(-(sc.pi * (2*n + 1) / d)**2 * D * t)
        J += term
    
    J *= (4 * c0 * D) / d
    return J


def plastics_outgassing_approximation(c0, D0, Ea, T, t, d=None, mode='short'):
    """
    Compute the plastic outgassing rate using either the short or long time approximation. From "Outgassing Model for Electronegative Impurites in Liquid Xenon for nEXO" 26-02-2020.

    Parameters:
    - c0: in ppb, initial gas concentration in the sample CHECK UNIT!!!
    - D0: in cm2.s-1, diffusion constant at infinite temperature
    - Ea: in Joules, activation energy
    - T: in Kelvin, temperature
    - t: in seconds, time
    - d: in mm, thickness, required for the 'long' mode
    - mode: either 'short' or 'long' to specify the approximation to use

    Returns:
    - Value of the plastic outgassing rate J based on the selected approximation
    """
    
    # Calculate the diffusion coefficient using the Arrhenius equation
    D = arrhenius(D0, Ea, T)
    
    # Short time approximation
    if mode == 'short':
        J = c0 * np.sqrt(D / t)
    
    # Long time approximation
    elif mode == 'long':
        if d is None:
            raise ValueError("Thickness 'd' must be specified for the 'long' mode.")
        J = (4 * c0 * D / d) * np.exp(-(sc.pi / d)**2 * D * t)
    
    else:
        raise ValueError("Invalid mode. Choose either 'short' or 'long'.")
    
    return J
