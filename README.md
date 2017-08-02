# Overview
This repo contains materials of project per the competition of https://www.kaggle.com/c/web-traffic-time-series-forecasting/

**Note:** This is a work-in-progress (WIP) repo, and it will be updated as more contributions committed

# Exploratory Data Analysis

## Results Summary
The summary of Exploratory Data Analysis (EDA) is presented below

- Short-term spikes (across a week?) may require median filter smoothening
- Wikipedia views again are hugely dependent on whether or not something is in the news
- The plots for pages showing the strongest periodic structure (ES, FR, EN) actually all have something in common
a.	They all seem to have something to do with health topics. 
b.	The weekly structure might make sense if it's related to people seeing doctors and then consulting Wikipedia. 
- The longer (~6 month) structure is harder to explain, especially without having any browser demographic information.
- We could potentially smooth out some of the unpredictable spikes and could maybe use high-view pages to reduce effects of statistical fluctuations on low-view pages
- The data for related topics seems to be correlated and that topics that are in the news get a lot of traffic, so this maybe points to some ways to improve things. 
- Unfortunately, with the different languages, training a model to identify related topics may be quite difficult.
- Trying to cluster similar topics using just the visit data rather than the page name might help us out a bit.
- Major traffic comes from desktop (which is surprising to me)
- Whenever mobile traffic goes up, desktop traffic takes a dip. Slight negative correlation can be checked numerically but seems pretty evident via graphs
- Spider as an agent has less contribution as compared to all access
- English language has maximum traffic followed by commons (pretty intuitive as commons will majorly have English content)
- All the other languages apart from English seem to have very less variation in the traffic. 
- Major takeaways from English traffic analysis by agent and by type of the access
a.	Average traffic (on MinMax Scale) on the desktop is 0.195 whereas the same for mobile web is 0.223
b.	Although, mobile access seems more consistent with lower standard deviation of 0.131 whereas the same for desktops is 0.174
c.	One thing which is clear from the graphs is that desktop traffic is smoother if you remove one abrupt peak that appears around 2016-08-04


## Misc observations

- NA and 0 are not distinguished in the raw train_1.csv data

# Files and Folders
This repo contains the following WIP artifacts

- eda.py - the file with track of EDA for the initial data (see EDA summary above)
- preprocessing.py - a separate file with data pre-processing routines
- simple_model.py - a simple moving-average forecasting model (is a good starting point for benchmarking)
- simple_model_with_weekends.py - a simple average median forecasting model, calculating median by page and type of day (weekend/regular day) (is a good starting point for benchmarking, scores top 35% result on the public leaderboard as of Jul 30, 2017)
- simple_model_with_all_extra_features.py - more advanced variation of the above-mentioned (locales, holidays by locales, outlier removal via winsorize transformation)
- prophet_pred_prototype.py - a prototype of Prophet-based TS forecasting (NB: this script does not produce a forecast submission file)
- ProphetModeller.py - a high-level class wrapper over TS forecasting algorithms behind Prophet library/tool 

**Note:** Prophet is the Facebook-developed open-source lib for producing high quality forecasts for time series data that has multiple seasonality with linear or non-linear growth (https://github.com/facebookincubator/prophet)
