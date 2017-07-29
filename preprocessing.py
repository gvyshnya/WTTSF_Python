# Project/Competition: https://www.kaggle.com/c/web-traffic-time-series-forecasting
# this EDA analysis has been inspired by
# - https://www.kaggle.com/muonneutrino/wikipedia-traffic-data-exploration/notebook
# - https://www.kaggle.com/indrajit/short-data-exploration-for-desktop-mobile-data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import collections as co

# functions

def get_language(page):
    """Parse the language abbreviation out of the page url of a Wikipedia page"""
    res = re.search('[a-z][a-z].wikipedia.org',page)  #regexp search
    if res:
        return res[0][0:2]
    return 'na'


# read the training data.
# I'm going to fill the NaN values with 0 since the dataset does not distinguish between 0 and missing.
# We'll have to deal with these later.
print("Reading training data into memory")
train = pd.read_csv('input/train_1.csv').fillna(0)
print(train.head())

# To save some memory, visit values will downcast everything to an integer.
# In Pandas, you can't automatically set columns with NaN values to integer types on reading the file,
# so we do it on the next step. This should reduce the size in memory from 600 Mbyte to 300 Mbyte.
# Views are an integer type anyway so this isn't losing any info.
# We might want our predictions to be floating point, though.
print("Down-casting the visit values to integers")
for col in train.columns[1:]:
    train[col] = pd.to_numeric(train[col],downcast='integer')
print(train.head())

# get some info about the training set
print("Output the information about the training data frame")
print(train.info())

# Traffic Influenced by Page Language so add the new feature
print("Add page language feature to the training data frame")
train['lang'] = train.Page.map(get_language)
print(co.Counter(train.lang))

#here we will seperate and make columns for various components of the project
components = pd.DataFrame([i.split("_")[-3:] for i in train["Page"]])
components.columns = ['Project', 'Access', 'Agent']
train[['Project', 'Access', 'Agent']] = components[['Project', 'Access', 'Agent']]
cols = train.columns.tolist()
cols = cols[-3:] + cols[:-3]
train = train[cols]
print(train.head())