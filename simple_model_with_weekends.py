# Project/Competition: https://www.kaggle.com/c/web-traffic-time-series-forecasting/
# Simple benchmark prediction with median (median by page and weekdays, no mean smoothening)
#
# Note: this is inspired by
# - https://www.kaggle.com/clustifier/weekend-weekdays/
# -

import pandas as pd
import numpy as np

print('Reading train data...')
train = pd.read_csv("input/train_1.csv")

print('Pre-processing and feature engineering train data...')
train_flattened = pd.melt(train[list(train.columns[-49:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')
train_flattened['weekend'] = ((train_flattened.date.dt.dayofweek) // 5 == 1).astype(float)

print('Reading key data...')
test = pd.read_csv("input/key_1.csv")

print('Processing key data...')
test['date'] = test.Page.apply(lambda a: a[-10:])
test['Page'] = test.Page.apply(lambda a: a[:-11])
test['date'] = test['date'].astype('datetime64[ns]')
test['weekend'] = ((test.date.dt.dayofweek) // 5 == 1).astype(float)

print('Calculating medians...')
train_page_per_dow = train_flattened.groupby(['Page','weekend']).median().reset_index()

print('Prepare submission dataframe...')
test = test.merge(train_page_per_dow, how='left')
test.loc[test.Visits.isnull(), 'Visits'] = 0

print('Output submission dataframe...')
test[['Id','Visits']].to_csv('output/mad.csv', index=False)