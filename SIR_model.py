# -*- coding: utf-8 -*-
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

def deriv_SIR(y, t, beta, gamma, d, epsilon, gammap, q):
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
    
    # First lockdown
    if 55<=t<=136:
      qq = q
    elif 279<=t<=306:
      qq = q
    elif t>=341:
      qq = q
    else:
      qq = 0

    #qq = q*(np.exp(-(t-95.5)**2/2./(40.5/2)**2) + np.exp(-(t-292.5)**2/2./(15/2)**2) + np.exp(-(t-393)**2/2./(52/2)**2))

    S, I, Q, R = y
    
    # Total population 
    N = S + I + R + Q
    beta /=N
    gammap = gamma
    
    # From https://www.sciencedirect.com/science/article/pii/S2468042720300439
    # For COVID 19, qprime can be set to zero
    dSdt = -beta * S * I - d*epsilon
    dIdt = beta * S * I  - gamma * I - qq*I
    dQdt = qq * I - gammap*Q
    dRdt = gamma * I + d*(1-epsilon) + gammap*Q
    return dSdt, dIdt, dQdt, dRdt

def integrate_SIR(S0, I0, Q0, R0, ndays, beta, gamma, d, epsilon, gammap, q):
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
    y0 = S0, I0, Q0, R0

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_SIR, y0, t, args=(beta, gamma, d, epsilon, gammap, q))
    S, I, Q, R = ret.T
    return S, I, Q, R


def logp_SIR(beta, gamma, d, epsilon, gammap, q):
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
    _, I_theory, _, _= integrate_SIR(S0, I0, Q0, R0, ndays, beta, gamma, d, epsilon,  gammap, q)
    
    I_theory = np.clip(I_theory, a_min = 1e-20, a_max = None)
    p = I_data*np.log((I_theory).astype(np.float32)) - I_theory - I_data.astype(np.float32)*np.log(I_data.astype(np.float32)) + I_data.astype(np.float32)
    return np.sum(p)
    
if __name__ == 'main':
    print('This file is to be used with cobaya, not executed')

if __name__ == 'SIR_model':
    # Read the data
    try:
        I_data, i_firstcase = read_data('./data/time_series_covid19_confirmed_global.csv', country = 'United Kingdom')
        I_data = I_data[i_firstcase:]
        #I_data = I_data[:150]
        I_data = np.clip(I_data, a_min = 1e-20, a_max = None)

        # Correct two big outliers
        I_data[152] = I_data[151]
        I_data[153] = I_data[154]

        # Total population, N.
        N = 66.65*1e6 # Approximate population of the UK
        # Number of days.
        ndays = len(I_data)
        # Initial number of infected and recovered individuals, I0 and R0.
        I0, Q0, R0 = I_data[0], 0, 0
        # Everyone else, S0, is susceptible to infection initially.
        S0 = N - I0 - Q0 - R0

    except: 
        print('Data file not found')


