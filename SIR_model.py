import numpy as np
import pandas
import math
from scipy.integrate import odeint
from cobaya.likelihood import Likelihood

def read_data(path, country = 'United Kingdom'):
    df = pandas.read_csv(path)

    # Extract specific country. The -1 keeps only the whole country, 
    # as opposed to just a region. The 4: keeps eliminates the entries
    # that are not part of the time series
    data = df[df['Country/Region'] == country].iloc[-1][4:]

    # Convert into a numpy array
    data = data.to_numpy()

    # Find when first case occurs
    i_firstcase = np.where(data>0)[0][0]
    data = data[i_firstcase:]

    return data

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

# Read the data
I_data = read_data('./data/time_series_covid19_confirmed_global.csv')

# Total population, N.
N = 66.65*1e6 # Approximate population of the UK
# Number of days.
ndays = len(I_data)
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = I_data[0], 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

def sir_logp(beta, gamma):
    _, I_theory, _ = integrate_SIR(S0, I0, R0, ndays, beta, gamma)

    chi2 = np.sum((I_theory - I_data)**2./I_theory**2)
    return -chi2 / 2
