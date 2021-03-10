import pandas as pd
import numpy as np
from ETS import seasonal_matrices


def forecasting(params, exogen, h, series_test, yearly_seasonality = None):
    '''
    This function runs the optimal values threw the model to extract optimal predictions for the evaluation data.
    In essence it is identical to the model function with the exception that it does not give the time series but the
    prediction horizon h. The computation of point forecast is done by passsing arguments to the ETS_M_Ad_M_forecast
    function below.

    Parameters:

      Time independent smoothing parameters:
      alpha = level smoothing coefficient
      beta = trend smoothing coefficient
      gamma = seasonality smoothing coefficient
      omega = trend dampening coefficient

      last period T fit states computed above according to Hyndman 2008.
      level_initial = period T fit level
      slope_initial = period T fit trend
      seasonal_initial7 ... seasonal_initial_HM = period T fit seasonal component where the number determines the lag of the dummy

      regression:
      exogen = exogen variables (time dependent)
      reg = regression coefficient vector

    Return: The function returns the point forecast and states for every time point in separate lists:
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
    seasonal_initial = np.vstack(params[6:13])
    reg = (params[13:13+len(exogen.columns)])

    if yearly_seasonality == 'fourier':
        epsilon = params[13+len(exogen.columns)]
        yearly = params[14+len(exogen.columns):379+len(exogen.columns)]

    elif yearly_seasonality == "dummies":
        yearly = (params[13+len(exogen.columns):25+len(exogen.columns)])

        epsilon = params[25+len(exogen.columns)]

    if yearly_seasonality == 'fourier' or yearly_seasonality == "dummies":
        results = ETS_M_Ad_M_forecast(alpha,beta,gamma,omega,level_initial,slope_initial,seasonal_initial,reg,h,exogen,series_test,yearly_seasonality,yearly, epsilon)
    else:
        results = ETS_M_Ad_M_forecast(alpha, beta, gamma, omega,level_initial, slope_initial, seasonal_initial, reg, h, exogen, series_test)

    return results


def ETS_M_Ad_M_forecast(alpha,beta,gamma,omega,level_initial,slope_initial,seasonal_initial,reg,h,exogen,series_test,yearly_seasonality = None, yearly_initial = None, epsilon = None):
    '''
    The ETS_M_Ad_M_forecast forecast function computes the forecast h steps ahead. As errors e are unavailable in prediction,
    this simplifies the process compared to the ETS_M_Ad_M function. The ETS_M_Ad_M_forecast does not calculate new state estimates
    or errors but merely forecasts and updates the seasonal states.

    Parameters:

      Time independent smoothing parameters:
      alpha = level smoothing coefficient
      beta = trend smoothing coefficient
      gamma = seasonality smoothing coefficient
      omega = trend dampening coefficient

      last period T fit states computed above according to Hyndman 2008.
      level_initial = period T fit level
      slope_initial = period T fit trend
      seasonal_initial7 ... seasonal_init = period T fit seasonal component where the number determines the lag of the dummy

      regression:
      exogen = exogen variables (time dependent)
      reg = regression coefficient vector

    Return: The function returns the point forecast and states for every time point in separate lists:
      point forecast
      level_list
      slope_list
      seasonal_list
    '''

    #computing the number of time points as the length of the forecasting vector

    t = h
    point_forecast = list()
    level_list = list()
    slope_list = list()
    seasonal_list = list()

    if yearly_seasonality == 'fourier' or yearly_seasonality == 'dummies':
        yearly_list = list()

    #Initilaisation

    level_past = level_initial
    slope_past = slope_initial
    seasonal_past = seasonal_initial

    if yearly_seasonality == 'fourier' or yearly_seasonality == 'dummies':
        yearly_past = yearly_initial

    #defining the seasonal matrices for the calculation of new state estimates

    if yearly_seasonality == 'fourier':
        weekly_transition_matrix, weekly_update_vector, yearly_transition_matrix, yearly_update_vector = seasonal_matrices(yearly_seasonality)
    else:
        weekly_transition_matrix, weekly_update_vector = seasonal_matrices(yearly_seasonality)


    for i in range(0,h):

        #compute one step ahead  forecast for timepoint t
        # Note: need -1 here exogen.iloc[i-1] as exogen contains the forecasting data and indexing starts at 0

        if yearly_seasonality == 'fourier':
            estimate = (level_past + omega * slope_past) * seasonal_past[0]  * yearly_past[0] + np.dot(reg, exogen.iloc[i]) * (level_past + omega * slope_past) * seasonal_past[0]  * yearly_past[0]
        elif yearly_seasonality == 'dummies':
            estimate = (level_past + omega * slope_past) * seasonal_past[0] * yearly_past[series_test.index.month[i] - 1] + np.dot(reg, exogen.iloc[i]) * (level_past + omega * slope_past) * seasonal_past[0] * yearly_past[series_test.index.month[i] - 1]
        else:
            estimate = (level_past + omega * slope_past) * seasonal_past[0] + np.dot(reg,exogen.iloc[i]) * (level_past + omega * slope_past) * seasonal_past[0]

        point_forecast.append(estimate)
        level_list.append(level_past)
        slope_list.append(slope_past)
        seasonal_list.append(seasonal_past[0])

        if yearly_seasonality == 'fourier':
            yearly_list.append(yearly_past[0])
        if yearly_seasonality == 'dummies':
            yearly_list = yearly_past


        #Updating
        #no changes in level (l) and slope (b) as they remain constant without new information
        #only changes in seasonality (s) as it cycles every 7 days, the effect of each individual seasonality is not updated

        seasonal_past = np.dot(weekly_transition_matrix,seasonal_past)

        if yearly_seasonality == 'fourier':
            yearly_past = np.dot(yearly_transition_matrix, yearly_past)

    if yearly_seasonality == 'fourier' or yearly_seasonality =='dummies':
        return  {'point forecast' : point_forecast,'level_list' : level_list, 'slope_list' : slope_list, 'seasonal_list' : seasonal_list, 'yearly_list' : yearly_list}
    else:
        return  {'point forecast' : point_forecast,'level_list' : level_list, 'slope_list' : slope_list, 'seasonal_list' : seasonal_list}




