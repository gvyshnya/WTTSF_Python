# Project/Competition: https://www.kaggle.com/c/web-traffic-time-series-forecasting
# this EDA analysis has been inspired by
# - https://www.kaggle.com/muonneutrino/wikipedia-traffic-data-exploration/notebook
# - https://www.kaggle.com/indrajit/short-data-exploration-for-desktop-mobile-data
# - https://www.kaggle.com/headsortails/wiki-traffic-forecast-exploration-wtf-eda

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import collections as co
import scipy.fftpack as fftp

# functions

def get_language(page):
    """Parse the language abbreviation out of the page url of a Wikipedia page"""
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res[0][0:2]
    return 'na'


def plot_with_fft(key):
    """This function draws the plot of the time series with Fast Fourier Transform (FFT) applied"""
    fig = plt.figure(1, figsize=[15, 5])
    plt.ylabel('Views per Page')
    plt.xlabel('Day')
    plt.title(labels[key])
    plt.plot(days, sums[key], label=labels[key])

    fig = plt.figure(2, figsize=[15, 5])
    fft_complex = fftp.fft(sums[key])
    fft_mag = [np.sqrt(np.real(x) * np.real(x) + np.imag(x) * np.imag(x)) for x in fft_complex]
    fft_xvals = [day / days[-1] for day in days]
    npts = len(fft_xvals) // 2 + 1
    fft_mag = fft_mag[:npts]
    fft_xvals = fft_xvals[:npts]

    plt.ylabel('FFT Magnitude')
    plt.xlabel(r"Frequency [days]$^{-1}$")
    plt.title('Fourier Transform')
    plt.plot(fft_xvals[1:], fft_mag[1:], label=labels[key])
    # Draw lines at 1, 1/2, and 1/3 week periods
    plt.axvline(x=1. / 7, color='red', alpha=0.3)
    plt.axvline(x=2. / 7, color='red', alpha=0.3)
    plt.axvline(x=3. / 7, color='red', alpha=0.3)

    print(plt.show())


def plot_entry(key, idx):
    """ This function will plot the line graph for the individual page's time series
    Note: it is dependent on global objects defined in the main execution loop as follows
    - lang_sets
    - train
    """
    data = lang_sets[key].iloc[idx, 1:]
    fig = plt.figure(1, figsize=(10, 5))
    plt.plot(days, data)
    plt.xlabel('day')
    plt.ylabel('views')
    plt.title(train.iloc[lang_sets[key].index[idx], 0])

    print(plt.show())

#################################################
# Main executable part of the script
#################################################

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

# Is Traffic Influenced by Page Language?
print("Add page language feature to the training data frame")
train['lang'] = train.Page.map(get_language)
print(co.Counter(train.lang))

# There are 7 languages plus the media pages. The languages used here are: English, Japanese, German, French, Chinese,
# Russian, and Spanish. This will make any analysis of the URLs difficult since there are four different writing
# systems to be dealt with (Latin, Cyrillic, Chinese, and Japanese). Here, I will create dataframes for the different
# types of entries. I will then calculate the sum of all views. I would note that because the data comes from several
# different sources, the sum will likely be double counting some of the views.
print("Calculate visits by language across the training data frame")
lang_sets = {}
lang_sets['en'] = train[train.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = train[train.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = train[train.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = train[train.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = train[train.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = train[train.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = train[train.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = train[train.lang=='es'].iloc[:,0:-1]

sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]

# So then how does the total number of views change over time?
# I'll plot all the different sets on the same plot.
print("Review how the total number of views change over time")
days = [r for r in range(sums['en'].shape[0])]

fig = plt.figure(1, figsize=[10, 10])
print(fig.show())
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels = {'en': 'English', 'ja': 'Japanese', 'de': 'German',
          'na': 'Media', 'fr': 'French', 'zh': 'Chinese',
          'ru': 'Russian', 'es': 'Spanish'
          }

for key in sums:
    plt.plot(days, sums[key], label=labels[key])

plt.legend()
print(plt.show())

########################
# Results:
########################
# English shows a much higher number of views per page, as might be expected since Wikipedia is a US-based site.
# There is a lot more structure here than I would have expected. The English and Russian plots show very large spikes
# around day 400 (around August 2016), with several more spikes in the English data later in 2016. My guess is that
# this is the effect of both the Summer Olympics in August and the election in the US.
# There's also a strange feature in the English data around day 200.
# The Spanish data is very interesting too. There is a clear periodic structure there, with a ~1 week fast period and
# what looks like a significant dip around every 6 months or so.
#######################

#######################
# Periodic Structure and FFTs¶
#######################
# Since it looks like there is some periodic structure here, I will plot each of these separately so that the scale
# is more visible. Along with the individual plots, I will also look at the magnitude of the
# Fast Fourier Transform (FFT). Peaks in the FFT show us the strongest frequencies in the periodic signal.
print("Review seasonality and FFTs over time")
for key in sums:
    plot_with_fft(key)

#######################
# Results:
#######################
# From this we see that while the Spanish data has the strongest periodic features, most of the other languages show
# some periodicity as well. For some reason the Russian and media data do not seem to show much. I plotted red lines
# where a period of 1, 1/2, and 1/3 week would appear. We see that the periodic features are mainly at 1 and 1/2 week.
# This is not surprising since browsing habits may differ on weekdays compared to weekends, leading to peaks
# in the FFTs at frequencies of n/(1 week) for integer n. So, we've learned now that page views are not at all smooth.
# There is some regular variation from day to day, but there are also large effects that can happen quite suddenly.
# A model likely will not be able to predict the sudden spikes unless it can be fed more information about
# what is going on in the world that day.
########################

########################
# Individual Entry Data
########################
# Plot the data for some individual entries.
# I've picked some entries to look at, but there's not necessarily anything special about them.

# First, let's look at some English pages.
print("Plot graphs for some English pages")
idx = [1, 5, 10, 50, 100, 250,500, 750, 1001, 1500, 2000, 3000, 4000, 5000]
for i in idx:
    plot_entry('en',i)

#######################
# Result:
#######################
# We see that for individual pages, the data is also not smooth. There are sudden gigantic spikes, large shifts in
# the mean number of views, and other things. We can also clearly see the effects of current events on Wikipedia views.
# The 2016 North Korean nuclear test occurred, and a Wikipedia page was quickly constructed and received a huge number
# of views in a short time. The number of views mostly decayed away in 1 or 2 weeks.
# Hunky Dory received a large number of viewers around the beginning of 2016, corresponding to the death of David Bowie.
# The page about the shooting competition at the 2016 Olympics had a small number of views and then suddenly a lot
# right around the Olympics.
# There are also some oddities, like two huge spikes in the data for Fiji Water, and the sudden long-term increases
# in traffic to "Internet of Things" and "Credit score." Maybe there were some news stories about Fiji
# water on those days. For the others, maybe there was a change in search engine behavior or maybe some new links
# appeared in very visible locations.
####################################

# Now let's look at some Spanish entries.
print("Look at some Spanish pages")
idx = [1, 5, 10, 50, 100, 250, 500, 750, 1001, 1500, 2000, 3000, 4000, 5000]
for i in idx:
    plot_entry('es',i)

########################
# Results:
########################
# This shows even more extreme short-term spikes than the English data. If some of these are just one or two days
# before reverting back to the mean, they may be a sign that something is wrong with the data. To deal with extremely
# short spikes, which we almost certainly won't be able to predict, something like a median filter can be used to
# remove them.
# We see something very curious here, though. We see that a very strong periodic structure appears only in certain
# pages. The plots showing the strongest periodic structure actually all have something in common - they all seem to
# have something to do with health topics. The weekly structure might make sense if it's related to people seeing
# doctors and then consulting Wikipedia. The longer (~6 month) structure is harder to explain, especially
# without having any browser demographic information.
##############################

# Now let's look at some French entries
print("Look at some Spanish pages")
idx = [1, 5, 10, 50, 100, 250, 500, 750, 1001, 1500, 2000, 3000, 4000, 5000]
for i in idx:
    plot_entry('fr',i)

########################
# Results:
########################
# The French plots show more of the same. Wikipedia views again are hugely dependent on whether or not something is in
# the news. Leicester FC won the Premier League and received many page views around the championship.
# The Olympics caused a huge spike in traffic to their page. Christmas actually shows some interesting structure,
# with views steadily increasing throughout Advent.
########################

########################
# How does the aggregated data compare to the most popular pages?
########################
# I mentioned some of the potential problems with the aggregated data, so I'll now look at the most popular pages,
# which are generally going to be the main pages for the languages in this dataset.
# For each language get highest few pages
print("Look at top visited pages for each of the languages")
npages = 5 # number of top pages to select
top_pages = {}
for key in lang_sets:
    print(key)
    sum_set = pd.DataFrame(lang_sets[key][['Page']])
    sum_set['total'] = lang_sets[key].sum(axis=1)
    sum_set = sum_set.sort_values('total',ascending=False)
    print(sum_set.head(10))
    top_pages[key] = sum_set.index[0]
    print('\n\n')

for key in top_pages:
    fig = plt.figure(1,figsize=(10,5))
    cols = train.columns
    cols = cols[1:-1]
    data = train.loc[top_pages[key],cols]
    plt.plot(days,data)
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.title(train.loc[top_pages[key],'Page'])
    print(plt.show())

######################
# Results:
######################
# Comparing these to the aggregated data, we see that things are mostly pretty similar. I'm actually quite surprised
# that the Olympics would have such a huge effect on a site like Wikipedia. I would say that the Japanese, Spanish,
# and media data differ the most.
# For media pages, this is expected since most people will access pages via links from other sites rather than
# through the main page or search function.
# The fact that some of the languages show large differences between the main page and the aggregated data suggests
# that the dataset is perhaps not very representative of all traffic to Wikipedia.
#####################

######################################################################
# Quick Analysis by Project, Type of Access, and Type of Agent
######################################################################

#here we will seperate and make columns for various components of the project
components = pd.DataFrame([i.split("_")[-3:] for i in train["Page"]])
components.columns = ['Project', 'Access', 'Agent']
train[['Project', 'Access', 'Agent']] = components[['Project', 'Access', 'Agent']]
cols = train.columns.tolist()
cols = cols[-3:] + cols[:-3]
train = train[cols]
train.head()

# here wil well make three different dataframes to understand each one of them
df_access = train.groupby(['Access'])[cols].mean()
df_access = df_access.T

df_project = train.groupby(['Project'])[cols].mean()
df_project = df_project.T

df_agent = train.groupby(['Agent'])[cols].mean()
df_agent = df_agent.T

# Basic Time Series Visualization
# Let's plot the data for project, agent and access type to see how they vary over the period of time.

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (20,20), sharex=True)
df_access.plot(ax = ax1)
df_agent.plot(ax = ax2)
df_project.plot(ax = ax3)

# Smoothing Graph Using Moving Averages
# The graphs that we plotted have too much noise and hence not readable.
# We will improve smoothness by taking rolling mean/ moving averages to understand their behavior.

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (20,20), sharex=True)
pd.rolling_mean(df_access, window=14).plot(ax = ax1)
pd.rolling_mean(df_agent, window=14).plot(ax = ax2)
pd.rolling_mean(df_project, window=14).plot(ax = ax3)

############################################################
# Results/Takeaways:
############################################################
# We will note down key takeaways as soon as we find something, that way we will not be missing on anything!
# Here is what I get from these graphs:
# 1) Major traffic comes from desktop (which is surprising to me)
# 2) Whenever mobile traffic goes up, desktop traffic takes a dip. Slight negative correlation can be checked
#    numerically but seems pretty evident via graphs
# 3) Spider as an agent has less contribution as compared to all access
# 4) English language has maximum traffic followed by commons (pretty intuitive as commons will
#    majorly have English content)
# 5) All the other languages apart from English seem to have very less variation in the traffic.
#    This is an important point as English projects will have significant impact in predicting future traffic
#############################################################

#############################################################
# Study Impact of Access type on English Projects
#############################################################

#basically we will filtr out English project along with each type of access
df_all = train[(train['Access'] == 'all-access') & (train['Project'] == 'en.wikipedia.org')]
df_all = df_all.groupby(['Access'])[cols].mean()
df_all = df_all.T

df_desktop = train[(train['Access'] == 'desktop') & (train['Project'] == 'en.wikipedia.org')]
df_desktop = df_desktop.groupby(['Access'])[cols].mean()
df_desktop = df_desktop.T

df_mobile = train[(train['Access'] == 'mobile-web') & (train['Project'] == 'en.wikipedia.org')]
df_mobile = df_mobile.groupby(['Access'])[cols].mean()
df_mobile = df_mobile.T

df_all.head()

#scaling all three dataframes to understand impact. Min-Max scalling to be used.
df_desktop['desktop'] = (df_desktop.desktop - df_desktop.desktop.min())/\
                        (df_desktop.desktop.max() - df_desktop.desktop.min())
df_mobile['mobile-web'] = (df_mobile['mobile-web'] - df_mobile['mobile-web'].min())/\
                          (df_mobile['mobile-web'].max() - df_mobile['mobile-web'].min())
df_all['all-access'] = (df_all['all-access'] - df_all['all-access'].min())/\
                       (df_all['all-access'].max() - df_all['all-access'].min())

# Smoothing Graph Using Moving Averages, again
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (20,20), sharex=True)
pd.rolling_mean(df_all, window=14).plot(ax = ax1, style = 'g--')
pd.rolling_mean(df_desktop, window=14).plot(ax = ax2, style = 'g--')
pd.rolling_mean(df_mobile, window=14).plot(ax = ax3, style = 'g--')

print('On an average the traffic on all access is {}'.format(df_all['all-access'].mean()))
print('On an average the traffic on desktop is {}'.format(df_desktop['desktop'].mean()))
print('Deviation on desktop is {}'.format(df_desktop['desktop'].std()))
print('On an average the traffic on mobile is {}'.format(df_mobile['mobile-web'].mean()))
print('Deviation on mobile is {}'.format(df_mobile['mobile-web'].std()))

####################################
# Results/Takeaway
####################################
# 1) Average traffic(on MinMax Scale) on the desktop is 0.195 whereas the same for mobile web is 0.223
# 2) Although, mobile access seems more consistent with lower standard deviation of 0.131 whereas the same for desktops
#    is 0.174
# 3) One thing which is clear from the graphs is that desktop traffic is smoother if you remove one abrupt peak that
#    appears around 2016-08-04