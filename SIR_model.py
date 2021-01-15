import numpy as np
from scipy.integrate import odeint
from cobaya.likelihood import Likelihood

# The SIR model differential equations.
def deriv_SIR(y, t, beta, gamma):
    S, I, R = y
    
    # Total population 
    N = S + I + R
    
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def integrate_SIR(S0, I0, R0, ndays, beta, gamma):
    # A grid of time points (in days)
    t = np.linspace(0, ndays, ndays)
    
    # Initial conditions vector
    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_SIR, y0, t, args=(beta, gamma))
    S, I, R = ret.T
    return S, I, R

# Generate fake data

# Total population, N.
N = 1000
# Number of days.
ndays = 160
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.2, 1./10 

_, I_data_noiseless, _ = integrate_SIR(S0, I0, R0, ndays, beta, gamma)
I_data = np.random.normal(I_data_noiseless, 0.1)

def sir_logp(beta, gamma):
    _, I_theory, _ = integrate_SIR(S0, I0, R0, ndays, beta, gamma)

    chi2 = np.sum((I_theory - I_data)**2.)
    return -chi2 / 2
