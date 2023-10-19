import numpy as np
import matplotlib.pyplot as plt

def tau(t, M, rho, n0, F, eta, R0, alpha, n_p): # Electron lifetime calculation
    #R0=R0*1e-9
    factor_exp=np.exp(-rho*eta*F*t/M)
    denominator=n0*factor_exp+((R0+n_p*F)/(eta*F))*(1-factor_exp)
    
    # Make sure you're not dividing by zero
    if denominator==0:
        return 0

    return alpha/denominator


def XPM_tau_fit(t, C_el, n0, R0, rho, M): # Materials tests XPM fit
    return C_el/(n0+R0*rho*t/M)