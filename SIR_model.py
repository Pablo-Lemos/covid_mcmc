import numpy as np
import pandas
import math
import sys
from scipy.integrate import odeint
from cobaya.likelihood import Likelihood

def read_data(path, country = 'United Kingdom'):
    '''  Read the data file 

    Parameters:
    -----------
    path: str
      The path to the data file
    country: str
      The country which we want to read. Defaults to 'United Kingdom'

    Returns:
    --------
    data: nd.array
      The data extracted from the file
    firstcase: nd.array
      The index corresponding to the first non-zero element
   
    '''
    df = pandas.read_csv(path)

    # Extract specific country. The -1 keeps only the whole country, 
    # as opposed to just a region. The 4: keeps eliminates the entries
    # that are not part of the time series
    data = df[df['Country/Region'] == country].iloc[-1][4:]

    # Convert into a numpy array
    data = data.to_numpy()

    # The data is cummulative, undo that:
    data = data[1:] - data[:-1]

    # Find when first case occurs
    firstcase = np.where(data>0)[0][0]

    return data, firstcase

def deriv_SIR(y, t, beta, gamma):
    ''' The SIR model differential equations

    Parameters: 
    -----------
    y: tuple
      A tuplee containing S, I, R  the number of susceptible, infected
      and recovered respectively
    t: float
      Time
    beta, gamma: floats
      The parameters of the model

    Returns:
    --------
    dSdt, dIdt, dRdt: floats
      The derivatives of S, I, R with respect to time.
    '''

    S, I, R = y
    
    # Total population 
    N = S + I + R
    
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def integrate_SIR(S0, I0, R0, ndays, beta, gamma):
    ''' Integrate the SIR model equations
  
    Parameters:
    -----------
    S0, I0, R0: floats
      The initial values of S, I, R
    ndays: int
      The number of days over which to integratee
    beta, gamma: floats
      The parameters of the model

    Returns:
    --------
    S, I, R: nd.arrays
      The evolution of S, I, R over time

    '''

    # A grid of time points (in days)
    t = np.linspace(0, ndays, ndays)
    
    # Initial conditions vector
    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_SIR, y0, t, args=(beta, gamma))
    S, I, R = ret.T
    return S, I, R

def logp_SIR(beta, gamma):
    ''' 
    Calculate the log likelihood for the SIR model given parameters

    Parameters: 
    -----------
    beta, gamma: floats
      The parameters of the model

    Returns:
    --------
    logp: float
      The log likelihood (-2*chisq)
    '''
  
    _, I_theory, _= integrate_SIR(S0, I0, R0, ndays, beta, gamma)
    
    epsilon = 1.e-10 # To avoid dividing by zero
    chi2_I = np.sum((I_theory - I_data)**2./(I_theory+epsilon)**2)
    return -0.5*(chi2_I)

if __name__ == 'main':
    print('This file is to be used with cobaya, not executed')

if __name__ == 'SIR_model':
    # Read the data
    try:
        I_data, i_firstcase = read_data('./data/time_series_covid19_confirmed_global.csv', country = 'United Kingdom')
        I_data = I_data[i_firstcase:]

        # Total population, N.
        N = 66.65*1e6 # Approximate population of the UK
        # Number of days.
        ndays = len(I_data)
        # Initial number of infected and recovered individuals, I0 and R0.
        I0, R0 = I_data[0], 0
        # Everyone else, S0, is susceptible to infection initially.
        S0 = N - I0 - R0

    except: 
        print('Data file not found')


