import numpy as np
import pandas as pd
from scipy.optimize import minimize

def model_optimization(params, series, exogen, before, after, model,
                       alpha, beta, gamma, omega, epsilon, smoothing,
                       yearly_seasonality):
    '''
    This function computes optimal parameters for the model by a Maximum Likelihood approach.
    Following Hyndman 2008 p.69 the Adjusted Least Squared (ALS) estimate is equal to the ML estimate
    in the case of homoskedastic errors. As we incorperate the heteroskedasticity of the series threw
    multiplicativ components this assumption is valid. The computation works by passing
    parameters as well as a series and exogen variables to the model. The model then computes an error and gives it
    back to the optimizer which minimizes the error by means of changing the parameter threw a
    Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS) under a set of bounds for the parameters. These bounds are based
    on Forecasting by exponential smoothing (Hyndman et al. 2008) p.25 and Chapters 5,10. In essence we restrict the smoothing
    parameters to be between 0 and 1. The level and slopes are not restricted. The weekly seasonality effects are restricted
    between 0 and infinity as they are multiplicative. Thus they can raise sales to infinity and lower them close to zero.
    Finally the exogen variables which are also multiplicative, are restricted between -1 and infinity. Like the daily
    seasonality they can raise sales to infinity and lower them close to zero. The lower bound difference is due to the
    additiv component of the exogen variables in the model. Note that as users can choose how many days before and after events
    they include in the model there is a loop which extends the bounds list to include these exogen variables.

    Parameters:

        params: initial parameters for the optimization

        series: the time series in a pandas Series format

        exog: the exogen variables in a pandas DataFrame format with each column being a variable and the time as its index

        before: Array [] of the length of the columns of exogen. Each number says how many days in the past are to be considered
               for the events. The numbers are to be arranged in the order of the columns of the events.

        after: Array [] of the length of the columns of exogen. Each number says how many days in the future are to be
               considered for the events. The numbers are to be arranged in the order of the columns of the events.

        model: the model for which the parameters are optimized.

    Return: Results containing information about the optimization such as the optimal parameters,
    the status of the optimization and the Sum of squared errors at the optimum.
    '''

    #Defining bounds
    #Note: leaving out the smoothing parameter bounds if they are not part of the optimization

    if smoothing:
        bounds = [(-np.inf,np.inf),(-np.inf,np.inf),
                  (0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),
                  (0.000001,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf)]
    else:
        bounds = [(0.000001,0.9999),(0.000001,0.9999),(0.000001,0.9999),(0.000001,0.9999),(-np.inf,np.inf),(-np.inf,np.inf),
                  (0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),(0.000001,np.inf),
                  (0.000001,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf),(-1,np.inf)]

    #adding one bound for each additional day before and after

    for i in range(0,sum(before)+sum(after)):
        bounds.append((-1,np.inf))

    # appending fourier series bounds after the exogen bounds
    if yearly_seasonality == 'fourier':
        yearly_bounds = (0.000001, 10)
        for i in range(1, 21):
            bounds.append(yearly_bounds)

        #if smoothing is set we dont need bounds for epsilon, otherwise we do
        if smoothing is None:
            bounds.append((0.000001, 0.9999))

    #appending 12 bounds for yearly seasonality dummies
    if yearly_seasonality == 'dummies':
        yearly_bounds = (0.000001, 10)
        for i in range(1, 13):
            bounds.append(yearly_bounds)

        # if smoothing is set we dont need bounds for epsilon, otherwise we do
        if smoothing is None:
            bounds.append((0.000001, 0.9999))

    #Optimization

    if smoothing:
        results = minimize(model, params, args=(series, exogen, yearly_seasonality, alpha, beta, gamma, omega, epsilon, smoothing), method='L-BFGS-B', bounds=bounds)

        #In case of smoothing we add the smoothing parameters to the parameter set after optimizing as we need the m for computation latter, but they are hyperparameters so not part of the optimization.
        #Depending on the model we add epsilon to the optimal parameters or not

        if yearly_seasonality == 'fourier' or yearly_seasonality == 'dummies':
            Optimal_Parameters = np.concatenate((alpha,beta,gamma,omega,results.x,epsilon),axis=None)
        else:
            Optimal_Parameters = np.concatenate((alpha, beta, gamma, omega, results.x), axis=None)

        Optimization_success = results.success
        Optimum_SSE = results.fun
        Iterations_of_function = results.nit
        Evaluations_of_Derivativ = results.nfev

        #I need to change the parameters that are given back here so that they include the smoothing parameters, otherwise fit and forecast wont work

    else:
        results = minimize(model, params, args=(series, exogen, yearly_seasonality), method='L-BFGS-B', bounds=bounds)

        Optimal_Parameters = results.x
        Optimization_success = results.success
        Optimum_SSE = results.fun
        Iterations_of_function = results.nit
        Evaluations_of_Derivativ = results.nfev
    
    return Optimal_Parameters,Optimization_success,Optimum_SSE,Iterations_of_function,Evaluations_of_Derivativ
