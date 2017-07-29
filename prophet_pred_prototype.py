# Project/Competition: https://www.kaggle.com/c/web-traffic-time-series-forecasting/
# Basic prophet experiments - just one page TS examined
# Inspired by https://www.kaggle.com/tunguz/forecast-example-w-prophet-median/notebook

import pandas as pd
import numpy as np
import fbprophet as fbpro
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
N = 60 # number of days for test split
i = 5000 # one example time series to train

# read the training data.
# I'm going to fill the NaN values with 0 since the dataset does not distinguish between 0 and missing.
# We'll have to deal with these later.
print("Reading training data into memory")
all_data = pd.read_csv('input/train_1.csv').fillna(0).T # read data in a transponed form to make it pand ready
key = pd.read_csv('input/key_1.csv')
ssub = pd.read_csv('input/sample_submission_1.csv')

print(all_data.head())

print("Splitting all data into training and validaion sets")
train, test = all_data.iloc[0:-N,:], all_data.iloc[-N:,:]
all_data = None

train_cleaned = train.T.iloc[:,1:].T

#smoothen outliers that are out of 1.5*std with rolling median of 56 days
std_mult = 1.5 # std smoothening multiplier
observations_for_rolling_median = 56
print("Select one page data as a data frame")
data = train_cleaned.iloc[:,i].to_frame()
data.columns = ['visits']

print(data.info())
print(data.tail())

# fix to Series.rolling(window=50,min_periods=1,center=False).median()
print("smoothen outliers that are out of 1.5*std with rolling median of 56 days")
# FIXME: fix to series to avoid the warning on rolling_median to be deprecated in future releases
data['median'] = pd.rolling_median(data.visits, observations_for_rolling_median, min_periods=1)

data.loc[np.abs(data.visits-data.visits.median())>=(std_mult*data.visits.std()),'visits'] = \
    data.loc[np.abs(data.visits-data.visits.median())>=(std_mult*data.visits.std()),'median']
data.index = pd.to_datetime(data.index)

print(data.tail())

#prophet expects the  label names to be 'ds' (date and time in ts) and 'y' (value)
print("Model and forecast with Prophet")
X = pd.DataFrame(index=range(0,len(data)))
X['ds'] = data.index
X['y'] = data['visits'].values
X.tail()

model = fbpro.Prophet(yearly_seasonality=True)
model.fit(X)
future = model.make_future_dataframe(periods=N)
future.tail()

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

model.plot(forecast)

# cross-validation on the testing set
print("Do cross-validation on the validation set")
y_truth = test.iloc[:,i].values
y_forecasted = forecast.iloc[-N:,2].values

denominator = (np.abs(y_truth) + np.abs(y_forecasted))
diff = np.abs(y_truth - y_forecasted) / denominator
diff[denominator == 0] = 0.0
print(200 * np.mean(diff))

print(200 * np.median(diff))