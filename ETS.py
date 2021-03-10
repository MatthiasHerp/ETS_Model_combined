import pandas as pd
import numpy as np

def model(params, series, exogen,
          yearly_seasonality,
          alpha=None, beta=None, gamma=None, omega=None, epsilon=None, smoothing=None):

    """
    This function runs an ETS(M,Ad,M) model with exogen variables. This is an Error, Trend, Seasonality exponential smoothing
    model.The first M stands for multiplicative or relative errors, the Ad for an additive dampend trend and the last M for
    multiplicative seasonality. The model also contains additional exogen variables which are dummies for certain events.
    The actual computation of the fit model is done in the function ETS_M_Ad_M which further contains the functions
    calc_new_estimates, calc_error, save_estimates and seasonal_matrices. These are all explained in the following code.
    
    Parameters:

        params: model parameters

        series: the time series in a pandas Series format

        exog: the exogen variables in a pandas DataFrame format with each column being a variable and the time as its index
    
    Return: The function returns the sum of squared error of the fitted model. This allows the model to be inputed
    into an optimizer which minimizes the sum of squared residuals dependent on the input parameters (params).
    """
    # defining all model parameters from the params vector
    # Note that the seasonal and exogen variable parameters are vectors while the other parameters are scalars

    if smoothing:
        alpha = alpha
        beta = beta
        gamma = gamma
        omega = omega
        epsilon = epsilon
        level_initial = params[0]
        slope_initial = params[1]
        seasonal_initial = params[2:9] #prior we hade a np.vstack around this and this broke it down i think

        # added len(exogen) as now we have variable number of exogen variables due to days before and after

        reg = (params[9:9 + len(exogen.columns)])

    else:

        alpha = params[0]
        beta = params[1]
        gamma = params[2]
        omega = params[3]
        level_initial = params[4]
        slope_initial = params[5]
        seasonal_initial = params[6:13]




        #added len(exogen) as now we have variable number of exogen variables due to days before and after

        reg = (params[13:13+len(exogen.columns)])

    #defining the initial yearly seasonal components as a fourier series

    if yearly_seasonality == "fourier":

        # defining the index as a date variable which will become relevant for subsequent computation
        yearly = pd.DataFrame({'date': series.index})
        yearly = yearly.set_index(pd.PeriodIndex(series.index, freq='D'))

        # yearly seasonality with N=10 fourier series elements

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

        if smoothing:

            # 1. compute the fourier series results from the weights times the cos(t) and sin(t) for t=0...365

            yearly_init = params[9 + len(exogen.columns):29 + len(exogen.columns)] * yearly.iloc[0:365]

            # 2. sum up the total yearly seasonality of each day by summing up all weighted trigonometric functional values

            yearly_init = 1 + yearly_init.sum(axis=1)

            # 3. define this array of 365 dummies as an array

            yearly_init = yearly_init              #np.vstack(yearly_init)

            # 4. turn the array around as we want the most recent seasonality effect to be at the end

            yearly_init = yearly_init[::-1]

            # yearly smoothing parameter

            epsilon = epsilon

        else:

            # 1. compute the fourier series results from the weights times the cos(t) and sin(t) for t=0...365

            yearly_init = params[13+len(exogen.columns):33+len(exogen.columns)] * yearly.iloc[0:365]

            # 2. sum up the total yearly seasonality of each day by summing up all weighted trigonometric functional values

            yearly_init = 1 + yearly_init.sum(axis=1)

            # 3. define this array of 365 dummies as an array

            yearly_init = yearly_init              #np.vstack(yearly_init)

            # 4. turn the array around as we want the most recent seasonality effect to be at the end

            yearly_init = yearly_init[::-1]

            # yearly smoothing parameter

            epsilon = params[33+len(exogen.columns)]

    #Built in exception that gives out the parameters and the error sum if an error in the model occurs
    #Note that the exception gives back yearly seasonality parameters if they are specified in the model

    #For the dummy model we have 12 dummies and a smoothing parameter

    elif yearly_seasonality == "dummies":

        if smoothing:
            yearly_init = (params[9+len(exogen.columns):21+len(exogen.columns)])
            epsilon = epsilon
        else:
            yearly_init = (params[13+len(exogen.columns):25+len(exogen.columns)])
            epsilon = params[25+len(exogen.columns)]


    try:
        if yearly_seasonality == "fourier" or yearly_seasonality == 'dummies':
            results = ETS_M_Ad_M(alpha,beta,gamma,omega,level_initial,slope_initial,seasonal_initial,reg,series,exogen,yearly_seasonality,yearly_init,epsilon)
        else:
            results = ETS_M_Ad_M(alpha,beta,gamma,omega,level_initial,slope_initial,seasonal_initial,reg,series,exogen,yearly_seasonality,yearly_init=None,epsilon=None)

    except:
        if yearly_seasonality == "fourier":
            print('alpha:', alpha, 'beta:', beta, 'gamma:', gamma, 'omega:', omega, level_initial, slope_initial,
                  seasonal_initial, 'reg:', reg,'Fourier weights:', params[13+len(exogen.columns):33+len(exogen.columns)],'epsilon:', params[33+len(exogen.columns)])
            if error_sum:
                print('error_sum:', error_sum)
        if yearly_seasonality == "dummies":
            print('alpha:', alpha, 'beta:', beta, 'gamma:', gamma, 'omega:', omega, level_initial, slope_initial,
                  seasonal_initial, 'reg:', reg,'monthly dummies:', params[13+len(exogen.columns):25+len(exogen.columns)],'epsilon:', params[25+len(exogen.columns)])
            if error_sum:
                print('error_sum:', error_sum)
        else:
            print('alpha:', alpha, 'beta:', beta, 'gamma:', gamma, 'omega:', omega, level_initial, slope_initial,
                  seasonal_initial, 'reg:', reg)
            if error_sum:
                print('error_sum:', error_sum)


    error_list = results['errors_list']

    error_list = [number ** 2 for number in error_list]

    error_sum = sum(error_list)

    return error_sum


def calc_new_estimates(level_past, slope_past, seasonal_past, alpha, beta, omega, gamma, e, weekly_transition_matrix, weekly_update_vector, series, i,
                       yearly_seasonality = None, epsilon = None, yearly_past = None, yearly_transition_matrix = None, yearly_update_vector = None):

    """
    This function updates the state estimates of the ETS(M,Ad,M) model level_past, slope_past, seasonal_past by the innovations/errors
    of each period. It is a part of the loop of the fit calculator of the model. Note that it also moves up the dummies
    in the seasonality vector. The Inputs are all past states, the smoothing parameters and the weekly_transition_matrix and weekly_update_vector
    required to update the current dummy, put it at the bottom of the vector and move the dummies up one position each period.

    Parameters:

      Past state estimates:
      level_past = past level
      slope_past = past trend
      seasonal_past = past seasonal dummy vector

      Time independent smoothing parameters:
      alpha = level smoothing coefficient
      beta = trend smoothing coefficient
      gamma = seasonality smoothing coefficient
      omega = trend dampening coefficient

      constant matrix and vector for the seasonality:
      weekly_transition_matrix = serves the purpose of pushing all dummies up one position
                                 while the current dummy goes to the bottom
      weekly_update_vector = is zero for all parameters except the current dummy in last position, which is updated by e.

    Return: The function returns the updated states:
      level = updated level
      slope = updated trend
      seasonal = updated seasonality vector
    """

    #Note: i added series and i to the passed parameters so that i can find the correct month for updating the estimate

    level = (level_past + omega * slope_past) * (1 + alpha * e)
    slope = omega * slope_past + beta * (level_past + omega * slope_past) * e

    #First we update the estimate and then we transition the seasonal vector so that the top value goes to the bottom

    seasonal_past = seasonal_past * ( np.ones(7) + weekly_update_vector * gamma * e)
    seasonal = np.dot(weekly_transition_matrix,seasonal_past)

    #before: seasonal = np.dot(weekly_transition_matrix,seasonal_past) + weekly_update_vector * gamma * e


    if yearly_seasonality == 'fourier':

        yearly_past = yearly_past * (np.ones(365) + yearly_update_vector * epsilon * e)
        yearly = np.dot(yearly_transition_matrix, yearly_past)
        #before: yearly_past = yearly_past + yearly_update_vector * epsilon * e
        #yearly = np.dot(yearly_transition_matrix, yearly_past)

        #before:
        #yearly = np.dot(yearly_transition_matrix, yearly_past) + yearly_update_vector * epsilon * e
        #I made it into two steps because first I want to update the vector as it is, so update the today forecast value by the today error
        #then I want to transition the vector so that this upper value goes down
        #hope it makes more sense


    if yearly_seasonality == 'dummies':
        yearly_past[series.index.month[i]-1] = yearly_past[series.index.month[i]-1] * (1 + epsilon * e)
        yearly = yearly_past

        #before: yearly_past[series.index.month[i]-1] = yearly_past[series.index.month[i]-1] + epsilon * e
        #yearly = yearly_past


    if yearly_seasonality == 'fourier' or yearly_seasonality == 'dummies':
        return level,slope,seasonal,yearly
    else:
        return level, slope, seasonal

def calc_error(level_past, slope_past, seasonal_past, omega, series, i, reg, exogen,
               yearly_seasonality = None, yearly_past = None):

    """
    This function calculates the point forecast, the relativ and absolute forecasting error of the ETS(M,Ad,M) model.
    The Inputs are all past states and the trend dampening factor, the time point i, the time series to estimate y as well as  the
    exogen variables and their regressors.
    It is a part of the loop of the fit calculator of the model and thus time i dependent. The absolute errors are computed
    for the sum of squared errors. Note that the sum of squared errors could also be computed with the relativ errors.
    Here the exogen variables are included. In the computation of the forecast a term is added where the regressors are multiplied
    by their coefficients and the estimate purely based on the ETS Model. This allows us to have multiplicative effects of events
    without running into the issue of having an estimate of zero on days without events.

    Parameters:

      Past state estimates:
      level_past = past level
      slope_past = past trend
      seasonal_past = past seasonal dummy vector

      Time independent smoothing parameters:
      omega = trend dampening coefficient

      time dependent:
      series = time series
      i = current time point

      regression:
      exogen = exogen variables (time dependent)
      reg = regression coefficient vector for the exogen variables


    Return: The function returns the point forecast, the relativ and absolute error:
      estimate = point forecast
      e = relativ error
      e_absolute = absolute error
    """

    if yearly_seasonality == 'fourier':
        estimate = (level_past + omega * slope_past) * seasonal_past[0] * yearly_past[0] * ( 1 + np.dot(reg, exogen.iloc[i]) )

    elif yearly_seasonality == 'dummies':
        estimate = (level_past + omega * slope_past) * seasonal_past[0] * yearly_past[series.index.month[i]-1] * ( 1 + np.dot(reg, exogen.iloc[i]) )

    else:

        estimate = (level_past + omega * slope_past) * seasonal_past[0] * ( 1 + np.dot(reg, exogen.iloc[i]) )

    e = (series[i] - estimate) / estimate
    e_absolute = series[i] - estimate


    return estimate, e, e_absolute


def save_estimates(errors_list, point_forecast, level_list, slope_list, seasonal_list, e_absolute, estimate, level_past, slope_past, seasonal_past,
                   yearly_seasonality = None, yearly_list = None, yearly_past = None):

    """
    This function simply appends the state estimates, the point forecast and the absolute error of each period
    to previously defined lists in the ETS(M,Ad,M) model. The Inputs are all past states, the point forecast,
    the absolute error and their respective lists. It is a part of the loop of the fit calculator of the model.
    Sidenote: The Function has no difference to the model without exogen variables.

    Parameters:

      Past state estimates:
      level_past = past level
      slope_past = past trend
      seasonal_past = past seasonal dummy vector

      estimate = point forecast
      e_absolute = absolute error

      Lists accoridng to the above variables:
      errors_list
      point_forecast
      level_list
      slope_list
      seasonal_list

    Return: The function returns the updated Lists.
      errors_list
      point_forecast
      level_list
      slope_list
      seasonal_list
    """

    errors_list.append(e_absolute)
    point_forecast.append(estimate)
    level_list.append(level_past)
    slope_list.append(slope_past)
    seasonal_list.append(seasonal_past[0])

    if yearly_seasonality == 'fourier':
        yearly_list.append(yearly_past[0])
    elif yearly_seasonality == 'dummies':
        yearly_list = yearly_past

    #for dummies the yearly list only contains the 12 dummies thats why i store all dummies each time they are updated

    if yearly_seasonality == 'fourier' or yearly_seasonality == 'dummies':
        return errors_list,point_forecast,level_list,slope_list,seasonal_list,yearly_list
    else:
        return errors_list,point_forecast,level_list,slope_list,seasonal_list



def seasonal_matrices(yearly_seasonality = None):

    '''
    This function simply defines the weekly transition matrix and weekly updating matrix needed in the computation of
    new weekly seasonality dummies. The function is part of the initialisation where it passes the matrices which are
    then used in the loop for the computation of new state estimates.
    Sidenote: The Function has no difference to the model without exogen variables.

    Parameters:
    Sidenote: It has no inputs although it can be made more general at which point it would include a scalar as input containing the
       length of the seasonality (here 7).

    Return: It returns the above weekly transition matrix and the weekly updating matrix:
       weekly_transition_matrix
       weekly_update_vector

    '''

    #defining weekly transition matrix:
    #1. defining first column of zeros (1 row to short)

    col_1 = np.vstack(np.zeros(6))

    #2. defining identity matrix 1 row and column to small

    col_2_6 = np.identity(6)

    #3. adding the 1 column and the identity matrix, now all states are updated to jump up one step in the state vector

    matrix_6 = np.hstack((col_1,col_2_6))

    #4. creating a final row in which the current state is put in last place and will be added by an update

    row_7 = np.concatenate((1,np.zeros(6)), axis = None)

    #5. adding the last row to the matrix to make it complete

    weekly_transition_matrix = np.vstack((matrix_6,row_7))

    #defining the weekly updating vector

    weekly_update_vector = np.concatenate((1,np.zeros(6)), axis = None)

    # defining yearly transition matrix:
    # 1. defining first column of zeros (1 row to short)

    if yearly_seasonality == 'fourier':

        col_1 = np.vstack(np.zeros(364))

        # 2. defining identity matrix 1 row and column to small

        col_2_365 = np.identity(364)

        # 3. adding the 1 column and the identity matrix, now all states are updated to jump up one step in the state vector

        matrix_364 = np.hstack((col_1, col_2_365))

        # 4. creating a final row in which the current state is put in last place and will be added by an update

        row_365 = np.concatenate((1, np.zeros(364)), axis=None)

        # 5. adding the last row to the matrix to make it complete

        yearly_transition_matrix = np.vstack((matrix_364, row_365))

        # defining the yearly updating vector

        yearly_update_vector = np.concatenate((1,np.zeros(364)), axis=None)

    if yearly_seasonality == 'fourier':
        return weekly_transition_matrix, weekly_update_vector, yearly_transition_matrix, yearly_update_vector
    else:
        return weekly_transition_matrix, weekly_update_vector


def ETS_M_Ad_M(alpha,beta,gamma,omega,level_initial,slope_initial,seasonal_initial,reg,series,exogen,yearly_seasonality,yearly_init,epsilon):

    '''
    This function computes the fit of the ETS(M,Ad,M) model with exogen variables for given initial and smoothing parameters.
    It is given these inputs by the model function and itself contains the functions calc_new_estimates,
    calc_error, save_estimates and seasonal_matrices. It first defines time t as the length of the series.
    Further it creates lists for parameters to return. Then it initialises by setting th initial states to be the past states.
    This allows the loop to start where one step ahead forecasts and errors are computed. This is followed by an update
    of the states with the new errors and then a redefinition of the states which in turn restarts the loop for the next period.

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

      series = time series

      regression:
      exogen = exogen variables (time dependent)
      reg = regression coefficient vector

    Return: The function returns lists of the fit errors, the point forecasts and the states:
        errors_list
        point_forecast
        level_list
        slope_list
        seasonal_list
    '''

    t = len(series)
    errors_list = list()
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
        yearly_past = yearly_init

    #defining the seasonal matrices for the calculation of new state estimates

    if yearly_seasonality == 'fourier':
        weekly_transition_matrix,weekly_update_vector,yearly_transition_matrix,yearly_update_vector = seasonal_matrices(yearly_seasonality)
    else:
        weekly_transition_matrix,weekly_update_vector = seasonal_matrices(yearly_seasonality)


    for i in range(0,t):

        #compute one step ahead  forecast for timepoint i

        if yearly_seasonality == 'fourier' or yearly_seasonality == 'dummies':
            estimate, e, e_absolute = calc_error(level_past, slope_past, seasonal_past, omega, series, i, reg, exogen,yearly_seasonality, yearly_past)
        else:
            estimate, e, e_absolute = calc_error(level_past, slope_past, seasonal_past, omega, series, i, reg, exogen)


        #save estimation error for Likelihood computation as well as the states and forecasts (fit values)

        if yearly_seasonality == 'fourier' or yearly_seasonality == 'dummies':
            errors_list,point_forecast,level_list,slope_list,seasonal_list,yearly_list = save_estimates(errors_list, point_forecast, level_list, slope_list, seasonal_list, e_absolute, estimate, level_past, slope_past, seasonal_past, yearly_seasonality, yearly_list, yearly_past)
        else:
            errors_list,point_forecast,level_list,slope_list,seasonal_list = save_estimates(errors_list, point_forecast, level_list, slope_list, seasonal_list, e_absolute, estimate, level_past, slope_past, seasonal_past)



        #Updating all state estimates with the information set up to time point i

        if yearly_seasonality == 'fourier':
            level,slope,seasonal,yearly = calc_new_estimates(level_past, slope_past, seasonal_past, alpha, beta, omega, gamma, e, weekly_transition_matrix, weekly_update_vector, series, i, yearly_seasonality, epsilon, yearly_past, yearly_transition_matrix, yearly_update_vector)
        elif yearly_seasonality == 'dummies':
            level, slope, seasonal, yearly = calc_new_estimates(level_past, slope_past, seasonal_past, alpha, beta,omega, gamma, e, weekly_transition_matrix,weekly_update_vector, series, i, yearly_seasonality, epsilon, yearly_past, yearly_transition_matrix=None,yearly_update_vector=None)
        else:
            level,slope,seasonal = calc_new_estimates(level_past, slope_past, seasonal_past, alpha, beta, omega, gamma, e, weekly_transition_matrix, weekly_update_vector, series, i)


        #denote updated states from i as past states for time point i+1 in the next iteration of the loop

        level_past = level
        slope_past = slope
        seasonal_past = seasonal

        if yearly_seasonality == 'fourier' or yearly_seasonality == 'dummies':
            yearly_past = yearly



    if yearly_seasonality == 'fourier' or yearly_seasonality == 'dummies':
        return  {'errors_list' : errors_list, 'point forecast' : point_forecast, 'level_list' : level_list, 'slope_list' : slope_list, 'seasonal_list' : seasonal_list, 'yearly_list' : yearly_list}
    else:
        return  {'errors_list' : errors_list, 'point forecast' : point_forecast, 'level_list' : level_list, 'slope_list' : slope_list, 'seasonal_list' : seasonal_list}



