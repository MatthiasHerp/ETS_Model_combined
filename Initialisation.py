import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statistics import mean

def Initial_Parameter_calculater(series,exogen,
                                 alpha, beta, gamma, omega, epsilon, smoothing,
                                 yearly_seasonality):

    '''
    This function calculates initial parameters for the optimization of our ETS Model for our series.
    The Calculation is odne according to Forecasting by exponential smoothing (Hyndman et al. 2008) p.23-24.
    First, the initial seasonal parameters are calculated. This is done by computing a 7 lags moving average and then an
    additional 2 lag moving average on the resulting data. These results are used to detrend the series. Finally the
    average of the detrended values are used to obtain the initial seasonal parameters.
    Second, the initial Level and slope parameters are calculated. This is done by calculating a linear regression with
    a time dependent trend on first ten seasonally adjusted values. The model intercept becomes the initial level parameter.
    The initial slope is calculated by dividing the model trend through the mean of the first ten values of the series.
    The division is done as our model has a multiplicativ trend.
    The initial parameters for the exogen effects are calculated similar to the slope coefficient. We calculate a regression
    with each exogen variable as an explanatory variable. Then we divide the resulting coefficients by the mean of the series
    to obtain our initial parameters. Note that we use regress onto entire series as we have few observations for some events.
    Finally note that the smoothing parameters are set at 0.01 for beta and gamma and at 0.99 for omega. This assumes a
    consistent level, trend and seasonal effect, as small alpha, beta and gamma values mean weak adjustments of the
    level, slope and seasonal components to forecasting errors. A high omega value assumes a weak dampening of the trend
    as it is close to a value of 1 which would be a consistent trend.


    Parameters:

        series: the time series in a pandas Series format

        exogen: the exogen variables in a pandas DataFrame format with each column being a variable and the time as its index

    Return: an array of starting parameters for the model optimization
    '''

    #Initial seasonal Component

    #Computing Moving Average

    f = series[:371].rolling(window=7).mean()
    f = f.rolling(window=2).mean()

    #Detrending for multiplicative model
    #skip first 7 values of both series as they are used to start the moving average and only go till the 365 time point

    detrended_series = series[7:371]/f[7:]
    detrended_series.index = pd.to_datetime(detrended_series.index, format='%Y-%m-%d')

    #Check what weekday the first observation is and store it in order to get the
    #initial seasonal parameters in the right order.

    Daynumber = pd.to_datetime(series.index, format='%Y-%m-%d')[0].weekday()

    #grouping detrended series by the day of the week and computing the means

    weekyday_means = detrended_series.groupby(detrended_series.index.dayofweek).mean()

    #Define all inital seasonal values.
    #Note:The oldes value is the current seasonal.

    weekly_initial = np.zeros(7)
    for i in range(0, 7):
        weekly_initial[i] = weekyday_means[abs(Daynumber - i)]


    #Normalizing the seasonal indices so they add to m (m=7 as we have weekly seasonality).
    #done by dividing them all by there total sum and multiplying with m.

    total = sum(weekly_initial)

    multiplier = 7 / total

    weekly_initial  = weekly_initial * multiplier

    #Initial Level and slope components

    #creating a dataframe containing the first 10 values seasonaly adjusted (values) and a time index (t)

    first_10 = pd.DataFrame()
    first_10['values'] = np.zeros(10)
    first_10['t'] = range(0,10)

    #computing the seasonal adjustment
    #first by creating a data frame with the first 10 seasonal adjustments

    weekly_initial_10 = np.concatenate((weekly_initial,weekly_initial[0:3]))
    weekly_initial_10 = pd.DataFrame(weekly_initial_10, columns=['inits'])

    #computing the seasonally adjusted values

    for i in range(0,10):
        first_10.at[i,'values'] = series[i] / weekly_initial_10.at[i,'inits']

    #Computing the Linear regression with the first 10 seasonally adjusted values

    reg = LinearRegression().fit(first_10['t'].values.reshape(-1,1),first_10['values'].values.reshape(-1,1))

    #Initial level component is equal to the intercept

    level_initial = reg.intercept_[0]
    
    #Intial slope component is equal to the regression coefficient

    slope_initial = reg.coef_[0] / mean(series[0:10])

    #Initial values for the regressors

    reg2 = LinearRegression().fit(exogen,series)

    #defining values for starting parameters of the exogen variables
    #as we have a model with multiplicative effect i divide the coefficients by the mean over the time period

    exogen_initial_parameters = reg2.coef_[0:exogen.shape[1]] / mean(series)

    #Initial Yearly Seasonality effects
    #This part of the code is only executed if the user specifies the yearly seasonality to be modelled by a fourier series

    if yearly_seasonality == "fourier":

        #defining the index as a date variable which will become relevant for subsequent computation

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

        # add week day dummies so that weekly seasonality is accounted for

        yearly['Monday'] = np.zeros(len(yearly))
        for i in range(0, len(yearly)):
            if yearly.index.dayofweek[i] == 0:
                yearly['Monday'][i] = 1
            else:
                yearly['Monday'][i] = 0

        yearly['Tuesday'] = np.zeros(len(yearly))
        for i in range(0, len(yearly)):
            if yearly.index.dayofweek[i] == 1:
                yearly['Tuesday'][i] = 1
            else:
                yearly['Tuesday'][i] = 0

        yearly['Wensday'] = np.zeros(len(yearly))
        for i in range(0, len(yearly)):
            if yearly.index.dayofweek[i] == 2:
                yearly['Wensday'][i] = 1
            else:
                yearly['Wensday'][i] = 0

        yearly['Thursday'] = np.zeros(len(yearly))
        for i in range(0, len(yearly)):
            if yearly.index.dayofweek[i] == 3:
                yearly['Thursday'][i] = 1
            else:
                yearly['Thursday'][i] = 0

        yearly['Friday'] = np.zeros(len(yearly))
        for i in range(0, len(yearly)):
            if yearly.index.dayofweek[i] == 4:
                yearly['Friday'][i] = 1
            else:
                yearly['Friday'][i] = 0

        yearly['Saturday'] = np.zeros(len(yearly))
        for i in range(0, len(yearly)):
            if yearly.index.dayofweek[i] == 5:
                yearly['Saturday'][i] = 1
            else:
                yearly['Saturday'][i] = 0

        yearly['Sunday'] = np.zeros(len(yearly))
        for i in range(0, len(yearly)):
            if yearly.index.dayofweek[i] == 6:
                yearly['Sunday'][i] = 1
            else:
                yearly['Sunday'][i] = 0

        #Linear regression to estimate initial yearly seasonality parameters
        #we regress the 20 sin(t) and cos(t) from our fourier series + 7 weekly dummies (as control) on the entire 1 year fit data
        #this gets us estimates for the fourier series weights in the first year

        reg3 = LinearRegression().fit(yearly[0:365], series[0:365])

        #deviding the resulting coefficients by the mean of the data over that period
        #reason: we have multiplicative thus relative seasonality, in our regression we have absolute
        #so we divide by series mean to get relative estimates

        yearly_initial = reg3.coef_ / mean(series)

        # we ommit our 7 daily seasonality estimates so that we only have the yearly estimates for the optimization

        yearly_initial = yearly_initial[:-7]


    if yearly_seasonality == 'dummies':
        # Initial parameters for yearly seasonality modelled by monthly dummies

        yearly_initial = series.groupby(series.index.month).mean() / mean(series)

    #Defining Starting Parameters array
    #The first values are the smoothing parameters: alpha, beta, gamma, omega
    #The If loop gives back the parameters with our without yearly seasonality.

    if yearly_seasonality == "fourier" or yearly_seasonality == "dummies":

        if smoothing:
            Starting_Parameters = np.concatenate((level_initial,
                                                  slope_initial,
                                                  weekly_initial,
                                                  exogen_initial_parameters,
                                                  yearly_initial), axis=None)
        else:
            Starting_Parameters = np.concatenate((alpha,
                                                  beta,
                                                  omega,
                                                  gamma,
                                                  level_initial,
                                                  slope_initial,
                                                  weekly_initial,
                                                  exogen_initial_parameters,
                                                  yearly_initial,
                                                  epsilon), axis=None)
    else:
        if smoothing:
            Starting_Parameters = np.concatenate((level_initial,
                                                  slope_initial,
                                                  weekly_initial,
                                                  exogen_initial_parameters), axis=None)
        else:
            Starting_Parameters = np.concatenate((alpha,
                                                  beta,
                                                  gamma,
                                                  omega,
                                                  level_initial,
                                                  slope_initial,
                                                  weekly_initial,
                                                  exogen_initial_parameters), axis=None)


    return Starting_Parameters
