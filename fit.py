import pandas as pd
import numpy as np
from ETS import ETS_M_Ad_M

def fit_extracter(params, series, exogen, yearly_seasonality = None):
    '''
    This function runs the optimal values threw the model to extract optimal (fitted) forecasts for the training data.
    In essence it is identical to the model function with the exception that it returns the full array of lists given by the
    ETS_M_Ad_M function.

    Parameters:

      Time independent smoothing parameters:
      alpha = level smoothing coefficient
      beta = trend smoothing coefficient
      gamma = seasonality smoothing coefficient
      omega = trend dampening coefficient

      Initial states computed above according to Hyndman 2008.
      level_initial = initial level
      slope_initial = initial trend
      seasonal_initial7 ... seasonal_initial = initial seasonal component where the number determines the lag of the dummy

      regression:
      exog = exogen variables (time dependent)
      reg = regression coefficient vector

    Return: The function returns the error, point forecast and states for every time point in separate lists:
      errors_list
      point forecast
      level_list
      slope_list
      seasonal_list
    '''

    #Note: the regression parameter has a variable length as due to the setting of before and after th enumber of exogen variables varies

    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    omega = params[3]
    level_initial = params[4]
    slope_initial = params[5]
    seasonal_initial = params[6:13]
    reg = (params[13:13+len(exogen.columns)])

    if yearly_seasonality == 'fourier':
        # defining the initial yearly seasonal components
        # Trick to lower the number of parameters to estimate:
        #   I dont estimate 365 dummies but rather 20 furier series weights and cumpute the dummies with them

        yearly = pd.DataFrame({'date': series.index})
        yearly = yearly.set_index(pd.PeriodIndex(series.index, freq='D'))

        # yearly seasonality with N=10
        # N=1
        yearly['yearly_sin365'] = np.sin(2 * np.pi * yearly.index.dayofyear / 365.25)
        yearly['yearly_cos365'] = np.cos(2 * np.pi * yearly.index.dayofyear / 365.25)
        # N=2
        yearly['yearly_sin365_2'] = np.sin(4 * np.pi * yearly.index.dayofyear / 365.25)
        yearly['yearly_cos365_2'] = np.cos(4 * np.pi * yearly.index.dayofyear / 365.25)
        # N=3
        yearly['yearly_sin365_3'] = np.sin(6 * np.pi * yearly.index.dayofyear / 365.25)
        yearly['yearly_cos365_3'] = np.cos(6 * np.pi * yearly.index.dayofyear / 365.25)
        # N=4
        yearly['yearly_sin365_4'] = np.sin(8 * np.pi * yearly.index.dayofyear / 365.25)
        yearly['yearly_cos365_4'] = np.cos(8 * np.pi * yearly.index.dayofyear / 365.25)
        # N=5
        yearly['yearly_sin365_5'] = np.sin(10 * np.pi * yearly.index.dayofyear / 365.25)
        yearly['yearly_cos365_5'] = np.cos(10 * np.pi * yearly.index.dayofyear / 365.25)
        # N=6
        yearly['yearly_sin365_6'] = np.sin(12 * np.pi * yearly.index.dayofyear / 365.25)
        yearly['yearly_cos365_6'] = np.cos(12 * np.pi * yearly.index.dayofyear / 365.25)
        # N=7
        yearly['yearly_sin365_7'] = np.sin(14 * np.pi * yearly.index.dayofyear / 365.25)
        yearly['yearly_cos365_7'] = np.cos(14 * np.pi * yearly.index.dayofyear / 365.25)
        # N=8
        yearly['yearly_sin365_8'] = np.sin(16 * np.pi * yearly.index.dayofyear / 365.25)
        yearly['yearly_cos365_8'] = np.cos(16 * np.pi * yearly.index.dayofyear / 365.25)
        # N=9
        yearly['yearly_sin365_9'] = np.sin(18 * np.pi * yearly.index.dayofyear / 365.25)
        yearly['yearly_cos365_9'] = np.cos(18 * np.pi * yearly.index.dayofyear / 365.25)
        # N=10
        yearly['yearly_sin365_10'] = np.sin(20 * np.pi * yearly.index.dayofyear / 365.25)
        yearly['yearly_cos365_10'] = np.cos(20 * np.pi * yearly.index.dayofyear / 365.25)

        # deleting date column as it is no longer required and should not be in the linear regression
        del yearly['date']

        # 1. compute the fourier series results from the weights times the cos(t) and sin(t) for t=0...365
        yearly_init = params[13+len(exogen.columns):33+len(exogen.columns)] * yearly.iloc[0:365, ]
        # 2. sum up the total yearly seasonality of each day by summing up all weighted trigonometric functional values
        yearly_init = 1 + yearly_init.sum(axis=1)
        # 3. define this array of 365 dummies as an array
        yearly_init = yearly_init
        # 4. turn the array around as we want the most recent seasonality effect to be at the end
        yearly_init = yearly_init[::-1]

        # yearly smoothing parameter
        epsilon = params[33+len(exogen.columns)]

    elif yearly_seasonality == 'dummies':
        yearly_init = (params[13 + len(exogen.columns):25 + len(exogen.columns)])

        epsilon = params[25 + len(exogen.columns)]

    if yearly_seasonality == 'fourier' or yearly_seasonality == 'dummies':
        results = ETS_M_Ad_M(alpha,beta,gamma,omega,level_initial,slope_initial,seasonal_initial,reg,series,exogen,yearly_seasonality,yearly_init,epsilon)
    else:
        results = ETS_M_Ad_M(alpha,beta,gamma,omega,level_initial,slope_initial,seasonal_initial,reg,series,exogen,yearly_seasonality,yearly_init=None,epsilon=None)

    return results
