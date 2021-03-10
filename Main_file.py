import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from days_around_events import days_around_events
from ETS import model
from fit import fit_extracter
from forecast import forecasting
from Initialisation import Initial_Parameter_calculater
from Optimizer import model_optimization
from Evaluation import eval_metrics

'''
series must be an pd.Series with a date as index

'''

def ETS(series, exogen, test_dataset_length, before=[0,0,0,0,0], after=[0,0,0,0,0],
        alpha = 0.1, beta = 0.01, omega = 0.99, gamma = 0.01, epsilon = 0.01, smoothing=None,
        yearly_seasonality=None, Optimal_Parameters=None):

    #preparing the series by
    #1.checking that the index is a datetime variable
    #2.defining training and testing dataset

    series.index = pd.to_datetime(series.index)
    series_training = series[:-test_dataset_length]
    series_test = series[-test_dataset_length:]

    #preparing the exogen data by
    #1.Include days before and after events into the exogen data set
    #2.defining training and testing dataset

    exogen = days_around_events(exogen, before, after)
    exogen_training = exogen.iloc[:(len(series)-31)]
    exogen_test= exogen.iloc[len(series)-31:]

    #calculate initial parameters
    print("computing initial parameters")
    Initial_Parameters = Initial_Parameter_calculater(series_training, exogen_training, alpha, beta, omega, gamma, epsilon, smoothing, yearly_seasonality)

    print(Initial_Parameters)

    #if no optimal parameters are passed they are computed
    #Note: we set all Optimization results to None as default

    Optimization_success, Optimum_SSE, Iterations_of_function, Evaluations_of_Derivativ = None,None,None,None

    if Optimal_Parameters == None:

        #optimize Model with initial parameters
        print("computing optimal parameters")
        Optimal_Parameters,Optimization_success,Optimum_SSE,Iterations_of_function,Evaluations_of_Derivativ = model_optimization(Initial_Parameters, series_training, exogen_training, before, after, model,
                                                                                                                                 alpha, beta, gamma, omega, epsilon, smoothing,
                                                                                                                                 yearly_seasonality)

    # calculating the fit values and storing them as a

    print("computing fit values")
    fit = fit_extracter(Optimal_Parameters, series_training, exogen_training, yearly_seasonality)
    fit_values = pd.Series(np.concatenate(fit['point forecast'], axis=None))
    fit_values.index = series_training.index


    #defining forecasting parameters depending on the model flavor
    #1.extracting the last (most recent) values of the states for forecasting
    #2.reversing the seasonals as the first one input into the forecast function needs to be the oldest seasonality
    #3.defining forecastin parameters

    l_values = fit['level_list'][len(fit['level_list']) - 1:]
    b_values = fit['slope_list'][len(fit['slope_list']) - 1:]
    s_values = fit['seasonal_list'][len(fit['seasonal_list']) - 7:]

    if yearly_seasonality == 'fourier':
        yearly_values = fit['yearly_list'][len(fit['yearly_list']) - 365:]
        forecast_parameters = np.concatenate([Optimal_Parameters[0:4], l_values, b_values, s_values, Optimal_Parameters[13:13+len(exogen.columns)],Optimal_Parameters[33+len(exogen.columns)], yearly_values],axis=None)
    elif yearly_seasonality == 'dummies':
        yearly_values = fit['yearly_list']
        forecast_parameters = np.concatenate([Optimal_Parameters[0:4], l_values, b_values, s_values, Optimal_Parameters[13:13 + len(exogen.columns)],
             Optimal_Parameters[25+len(exogen.columns)], yearly_values], axis=None)
    else:
        forecast_parameters = np.concatenate([Optimal_Parameters[0:4], l_values, b_values, s_values, Optimal_Parameters[13:13+len(exogen.columns)]],axis=None)

    #calculating forecasts and storing them as a series

    forecast = forecasting(forecast_parameters, exogen_test, test_dataset_length, series_test, yearly_seasonality)
    forecast_values = pd.Series(np.concatenate(forecast['point forecast'], axis=None))
    forecast_values.index = series_test.index

    if Optimization_success and Optimum_SSE and Iterations_of_function and Evaluations_of_Derivativ:
        results = {"initial_parameters": Initial_Parameters,
                   "optimal_parameters": Optimal_Parameters,
                   "Optimization_success": Optimization_success,
                   "Optimum_SSE": Optimum_SSE,
                   "Iterations_of_function": Iterations_of_function,
                   "Evaluations_of_Derivativ": Evaluations_of_Derivativ,
                   "fit_values": fit_values,
                   "forecast_values": forecast_values}
    else:
        results = { "initial_parameters" : Initial_Parameters,
                    "optimal_parameters" : Optimal_Parameters,
                    "fit_values" : fit_values,
                    "forecast_values": forecast_values}

    return results
'''
# setting working directory
os.chdir("C:/Users/mah/Desktop/M5_Wallmart_Challenge")
os.getcwd()

data = pd.read_csv("Revenue_Store_Category.csv")
data = pd.DataFrame(data)

series = pd.Series(data.iloc[:, 1])
series.index = data["date"]

# reading in the exogen variables which are the SNAP, Sporting, Cultural, National and Religious events
exogen = pd.read_csv("exogen_variables.csv", index_col='date')

before = [0, 0, 0, 0, 0]
after = [0, 0, 0, 0, 0]
'''
'''
# Getting Optimal Parameters of a series
# optimal parameters for first series in non yearly model
Optimal_Parameters = [8.28772801e-02, 1.00000000e-06, 1.00000000e-06, 9.99900000e-01,
               9.32285791e+02, 3.00938004e+00, 1.40184833e+00, 1.32193141e+00,
               1.05916303e+00, 9.58923845e-01, 9.51769248e-01, 9.73350948e-01,
               1.14057097e+00, 4.36378073e-02, -5.93421785e-02, -9.14167167e-02,
               -5.68305306e-02, 8.90162445e-03]


Model1 = ETS(series, exogen, 31, before, after, Optimal_Parameters=Optimal_Parameters)
print(Model1.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model1.get("forecast_values")))

Model1_optimized = ETS(series, exogen, 31, before, after)
print(Model1_optimized.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model1_optimized.get("forecast_values")))

Model1_optimized_smoothed = ETS(series, exogen, 31, before, after,alpha = 0.3, beta = 0.3, omega = 0.99, gamma = 0.1, epsilon = 0.1)
print(Model1_optimized_smoothed.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model1_optimized_smoothed.get("forecast_values")))

Model1_optimized_smoothed_T = ETS(series, exogen, 31, before, after,alpha = 0.1, beta = 0.1, omega = 0.99, gamma = 0.01, epsilon = 0.01, smoothing=True)
print(Model1_optimized_smoothed_T.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model1_optimized_smoothed_T.get("forecast_values")))

# Plotting fitted, forecasted and actual values
# Plot the results
graph = plt.figure(figsize=(15, 5))
plt.plot(series, color='blue')
plt.plot(Model1_optimized_smoothed_T.get("fit_values"), color="green")
plt.plot(Model1_optimized_smoothed_T.get("forecast_values"), color="red")
plt.xlabel("date")
plt.ylabel(str(data.columns[1]))
plt.legend(("realization", "fitted"),loc="upper left")
plt.title(str(data.columns[1]))
plt.show()



# Getting Optimal Parameters of a series
# optimal parameters for first series in fourier model
Optimal_Parameters = [8.57960426e-02, 1.00000000e-06, 1.00000000e-02, 9.99208607e-01,
               9.32285538e+02, 6.03447509e+00, 1.29503879e+00, 1.22132165e+00,
               9.78133335e-01, 8.86786533e-01, 8.80357178e-01, 9.00898037e-01,
               1.05397638e+00, 4.31177211e-02, -7.99988063e-02, -9.49761276e-02,
               -4.83860449e-02, 3.90980900e-03, -5.74413623e-02, 6.68286993e-02,
               1.10715189e-02, 4.90847997e-03, -7.93778641e-03, 2.92884586e-03,
               -1.48064268e-02, 7.59873962e-03, -6.29558272e-03, 1.22923705e-02,
               2.10812135e-02, 5.40841810e-03, 1.71701997e-02, 3.75883988e-04,
               -2.93339298e-04, -2.04232159e-03, -9.26423646e-03, -1.30144687e-03,
               4.20606441e-03, 6.53773692e-04, 0.1]


Model2 = ETS(series, exogen, 31, before, after, yearly_seasonality="fourier", Optimal_Parameters=Optimal_Parameters)
print(Model2.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model2.get("forecast_values")))

Model2 = ETS(series, exogen, 31, before, after, yearly_seasonality="fourier")
print(Model2.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model2.get("forecast_values")))

Model2 = ETS(series, exogen, 31, before, after, yearly_seasonality="fourier",alpha = 0.3, beta = 0.1, omega = 0.99, gamma = 0.1, epsilon = 0.1)
print(Model2.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model2.get("forecast_values")))

Model2 = ETS(series, exogen, 31, before, after, yearly_seasonality="fourier",alpha = 0.3, beta = 0.1, omega = 0.99, gamma = 0.1, epsilon = 0.1, smoothing=True)
print(Model2.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model2.get("forecast_values")))


Optimal_Parameters = [8.28772801e-02, 1.00000000e-06, 1.00000000e-06, 9.99900000e-01,
               9.32285791e+02, 3.00938004e+00, 1.40184833e+00, 1.32193141e+00,
               1.05916303e+00, 9.58923845e-01, 9.51769248e-01, 9.73350948e-01,
               1.14057097e+00, 4.36378073e-02, -5.93421785e-02, -9.14167167e-02,
               -5.68305306e-02, 8.90162445e-03,1,1,1,1,1,1,1,1,1,1,1,1,0.2]



Model3 = ETS(series, exogen, 31, before, after, yearly_seasonality="dummies", Optimal_Parameters=Optimal_Parameters)
print(Model3.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model3.get("forecast_values")))

Model3 = ETS(series, exogen, 31, before, after, yearly_seasonality="dummies")
print(Model3.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model3.get("forecast_values")))

Model3 = ETS(series, exogen, 31, before, after, yearly_seasonality="dummies", alpha = 0.3, beta = 0.1, omega = 0.99, gamma = 0.1, epsilon = 0.1)
print(Model3.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model3.get("forecast_values")))


Model3 = ETS(series, exogen, 31, before, after, yearly_seasonality="dummies", alpha = 0.2, beta = 0.1, omega = 0.99, gamma = 0.001, epsilon = 0.001, smoothing=True)
print(Model3.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model3.get("forecast_values")))
# Plotting fitted, forecasted and actual values
# Plot the results
graph = plt.figure(figsize=(15, 5))
plt.plot(series, color='blue')
plt.plot(Model3.get("fit_values"), color="green")
plt.plot(Model3.get("forecast_values"), color="red")
plt.xlabel("date")
plt.ylabel(str(data.columns[1]))
plt.legend(("realization", "fitted"),loc="upper left")
plt.title(str(data.columns[1]))
plt.show()


Model1_optimized = ETS(series, exogen, 31, before, after)
print(Model1_optimized.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model1_optimized.get("forecast_values")))

Model1_optimized_smoothed = ETS(series, exogen, 31, before, after,alpha = 0.3, beta = 0.3, omega = 0.99, gamma = 0.1, epsilon = 0.1)
print(Model1_optimized_smoothed.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model1_optimized_smoothed.get("forecast_values")))
'''

'''
#e has a lot of values more than one

Model1_optimized_smoothed_T = ETS(series, exogen, 31, before, after,alpha = 0.1, beta = 0.1, omega = 0.99, gamma = 0.01, epsilon = 0.01, smoothing=True)
print(Model1_optimized_smoothed_T.get("optimal_parameters"))
print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model1_optimized_smoothed_T.get("forecast_values")))
'''
'''
# Plotting fitted, forecasted and actual values
# Plot the results
graph = plt.figure(figsize=(15, 5))
plt.plot(series, color='blue')
plt.plot(Model1.get("fit_values"), color="green")
plt.plot(Model1.get("forecast_values"), color="red")
plt.xlabel("date")
plt.ylabel(str(data.columns[1]))
plt.legend(("realization", "fitted"),loc="upper left")
plt.title(str(data.columns[1]))
plt.show()



# Getting Optimal Parameters of a series
# optimal parameters for first series in fourier model
Optimal_Parameters = [8.57960426e-02, 1.00000000e-06, 1.00000000e-02, 9.99208607e-01,
               9.32285538e+02, 6.03447509e+00, 1.29503879e+00, 1.22132165e+00,
               9.78133335e-01, 8.86786533e-01, 8.80357178e-01, 9.00898037e-01,
               1.05397638e+00, 4.31177211e-02, -7.99988063e-02, -9.49761276e-02,
               -4.83860449e-02, 3.90980900e-03, -5.74413623e-02, 6.68286993e-02,
               1.10715189e-02, 4.90847997e-03, -7.93778641e-03, 2.92884586e-03,
               -1.48064268e-02, 7.59873962e-03, -6.29558272e-03, 1.22923705e-02,
               2.10812135e-02, 5.40841810e-03, 1.71701997e-02, 3.75883988e-04,
               -2.93339298e-04, -2.04232159e-03, -9.26423646e-03, -1.30144687e-03,
               4.20606441e-03, 6.53773692e-04, 0.1]


Model2 = ETS(series, exogen, 31, before, after, yearly_seasonality="fourier", Optimal_Parameters=Optimal_Parameters)

# Plotting fitted, forecasted and actual values
# Plot the results
graph = plt.figure(figsize=(15, 5))
plt.plot(series, color='blue')
plt.plot(Model2.get("fit_values"), color="green")
plt.plot(Model2.get("forecast_values"), color="red")
plt.xlabel("date")
plt.ylabel(str(data.columns[1]))
plt.legend(("realization", "fitted"),loc="upper left")
plt.title(str(data.columns[1]))
plt.show()

'''
'''
Optimal_Parameters = [8.28772801e-02, 1.00000000e-06, 1.00000000e-06, 9.99900000e-01,
               9.32285791e+02, 3.00938004e+00, 1.40184833e+00, 1.32193141e+00,
               1.05916303e+00, 9.58923845e-01, 9.51769248e-01, 9.73350948e-01,
               1.14057097e+00, 4.36378073e-02, -5.93421785e-02, -9.14167167e-02,
               -5.68305306e-02, 8.90162445e-03,1,1,1,1,1,1,1,1,1,1,1,1,0.2]



Model3 = ETS(series, exogen, 31, before, after, yearly_seasonality="dummies", Optimal_Parameters=Optimal_Parameters)
'''
'''
# Plotting fitted, forecasted and actual values
# Plot the results
graph = plt.figure(figsize=(15, 5))
plt.plot(series, color='blue')
plt.plot(Model3.get("fit_values"), color="green")
plt.plot(Model3.get("forecast_values"), color="red")
plt.xlabel("date")
plt.ylabel(str(data.columns[1]))
plt.legend(("realization", "fitted"),loc="upper left")
plt.title(str(data.columns[1]))
plt.show()

print("Model without yearly seasonality:",eval_metrics(series[len(series)-31:],Model1.get("forecast_values")))
print("Model with fourier:",eval_metrics(series[len(series)-31:],Model2.get("forecast_values")))
'''






'''
Model with Optimal Parameters runs and gives back:

[0.0828772801, 1e-06, 1e-06, 0.9999, 932.285791, 3.00938004, 1.40184833, 1.32193141, 1.05916303, 0.958923845, 0.951769248, 0.973350948, 1.14057097, 0.0436378073, -0.0593421785, -0.0914167167, -0.0568305306, 0.00890162445]
Model without yearly seasonality: (237.89143585433723, 189.53203772864418, 0.7135471606738739)


Model optimizing and creating its own startvalues gives back:

[ 9.24245015e-02  2.09459268e-03  1.41325386e-02  3.59874369e-03
  9.32284507e+02  5.86423600e-03  1.39266373e+00  1.30610066e+00
  1.03121556e+00  9.11210694e-01  8.93604604e-01  9.59371909e-01
  1.12280480e+00  4.38927934e-02 -5.41290300e-02 -1.00237073e-01
 -7.18604855e-02  4.94341862e-03]
Model without yearly seasonality: (237.92496120932657, 183.1333314273646, 0.7134664170373178)


Model getting different starting smoothing parameters but still using them in the optimization gives back:

[ 9.24452495e-02  2.33356618e-01  1.41261453e-02  1.00000000e-06
  9.32285141e+02  5.96103059e-03  1.39205766e+00  1.30554434e+00
  1.03073662e+00  9.10799518e-01  8.93194445e-01  9.58944862e-01
  1.12232815e+00  4.38904422e-02 -5.40920858e-02 -1.00234208e-01
 -7.18644681e-02  4.94011553e-03]
Model without yearly seasonality: (237.92455045218045, 183.1327574259691, 0.7134674063880326)




'''