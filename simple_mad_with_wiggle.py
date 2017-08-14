"""
    Project: https://www.kaggle.com/c/web-traffic-time-series-forecasting
    Purpose: simple MAD submission with weekends/monday, weights and future trend wiggling
    Inspired by: https://www.kaggle.com/chechir/weekend-flag-median-with-wiggle
"""
import pandas as pd
import datetime as dt

def get_raw_data():
    train = pd.read_csv("input/train_1.csv", low_memory = True)
    test = pd.read_csv("input/key_1.csv", low_memory = True)
    return train, test

def transform_data(train, test):
    train_flattened = pd.melt(train[list(train.columns[-49:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
    train_flattened = get_features(train_flattened)
    test['date'] = test.Page.apply(lambda a: a[-10:])
    test['Page'] = test.Page.apply(lambda a: a[:-11])
    test = get_features(test)
    return train_flattened, test

def get_features(df):
    df['date'] = df['date'].astype('datetime64[ns]')
    df['weekend'] = ((df.date.dt.dayofweek) // 5 == 1).astype(float)
    # see https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.dt.dayofweek.html
    df['monday'] = (df.date.dt.dayofweek == 0).astype(float)
    return df

def predict_using_median_geommean(train, test, weight=0.8):
    agg_train_monday = train.groupby(['Page','weekend','monday']).median().reset_index()
    agg_train_monday = agg_train_monday.rename(columns={'Visits':'mondayVisits'})
    train = train.drop(['monday'], axis=1)
    agg_train_weekend = train.groupby(['Page','weekend']).median().reset_index()

    test = test.merge(agg_train_weekend, how='left')
    test = test.merge(agg_train_monday, how='left')
    test['visits'] = test['Visits'].values*weight + test['mondayVisits'].values*(1-weight)
    return test

def wiggle_preds(df, adj1, adj2):
    second_term_ixs = df['date'] > '2017-02-01'
    adjusted = df['Visits'].values + df['Visits'].values*adj1
    adjusted[second_term_ixs] = df['Visits'].values[second_term_ixs] + df['Visits'].values[second_term_ixs]*adj2
    df['Visits'] = adjusted
    df.loc[df.Visits.isnull(), 'Visits'] = 0
    return df

if __name__ == '__main__':
    start_exec_time = dt.datetime.now()
    print("Started at ", start_exec_time)
    print('Reading data...', dt.datetime.now())
    train, test = get_raw_data()
    print('Transforming raw data...', dt.datetime.now())
    train, test = transform_data(train, test)

    weight = 0.84
    adj1 = 0.02
    adj2 = 0.05

    print('Starting median prediction with weights ...', dt.datetime.now())
    test_with_preds = predict_using_median_geommean(train, test, weight=weight)
    print('Wiggling prediction ...')
    test_with_preds = wiggle_preds(test_with_preds, adj1, adj2)

    test_with_preds[['Id','Visits']].to_csv('output/sub_mads_weight_{}.csv'.format(weight), index=False)
    print(test_with_preds[['Id', 'Visits']].head())
    print(test_with_preds[['Id', 'Visits']].tail())

    end_exec_time = dt.datetime.now()
    elapsed_time = end_exec_time - start_exec_time
    print("Completed mad weighted prediction...")
    print("Elapsed time: ", elapsed_time)
