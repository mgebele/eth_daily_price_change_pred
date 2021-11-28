#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#START-Classification-Models---Data-Set-Variations" data-toc-modified-id="START-Classification-Models---Data-Set-Variations-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>START Classification Models - Data Set Variations</a></span></li><li><span><a href="#Import-Data-Sets" data-toc-modified-id="Import-Data-Sets-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import Data Sets</a></span><ul class="toc-item"><li><span><a href="#Data-Set-I." data-toc-modified-id="Data-Set-I.-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Data Set I.</a></span></li></ul></li><li><span><a href="#import-btc-data" data-toc-modified-id="import-btc-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>import btc data</a></span></li><li><span><a href="#calculate-meier-multiple:-200-moving-average-current-price" data-toc-modified-id="calculate-meier-multiple:-200-moving-average-current-price-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>calculate meier multiple: 200 moving average current price</a></span></li><li><span><a href="#transfers-to/from-exchanges-diff-und-sliding-window" data-toc-modified-id="transfers-to/from-exchanges-diff-und-sliding-window-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>transfers to/from exchanges diff und sliding window</a></span></li><li><span><a href="#kick-out-meyer-multiple-eth-because-of-data-loss" data-toc-modified-id="kick-out-meyer-multiple-eth-because-of-data-loss-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>kick out meyer multiple eth because of data loss</a></span></li><li><span><a href="#Bring-df_dataset1-and-df_dataset_btc-together" data-toc-modified-id="Bring-df_dataset1-and-df_dataset_btc-together-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Bring df_dataset1 and df_dataset_btc together</a></span><ul class="toc-item"><li><span><a href="#Prediction-Shift" data-toc-modified-id="Prediction-Shift-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Prediction Shift</a></span><ul class="toc-item"><li><span><a href="#Data-Set-I." data-toc-modified-id="Data-Set-I.-7.1.1"><span class="toc-item-num">7.1.1&nbsp;&nbsp;</span>Data Set I.</a></span></li></ul></li></ul></li><li><span><a href="#get-yearly-btc-dollar-increase-for-traditional-investment" data-toc-modified-id="get-yearly-btc-dollar-increase-for-traditional-investment-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>get yearly btc dollar increase for traditional investment</a></span></li><li><span><a href="#Classification-Setup" data-toc-modified-id="Classification-Setup-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Classification Setup</a></span><ul class="toc-item"><li><span><a href="#Create-Data-Set-Copies-for-Trading-Strategy" data-toc-modified-id="Create-Data-Set-Copies-for-Trading-Strategy-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Create Data Set Copies for Trading Strategy</a></span></li><li><span><a href="#Categorization---Data-Set-I." data-toc-modified-id="Categorization---Data-Set-I.-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Categorization - Data Set I.</a></span></li></ul></li><li><span><a href="#Model-Preparation---Data-Set-I." data-toc-modified-id="Model-Preparation---Data-Set-I.-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Model Preparation - Data Set I.</a></span><ul class="toc-item"><li><span><a href="#Train-Test-Split---Data-Set-I." data-toc-modified-id="Train-Test-Split---Data-Set-I.-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Train Test Split - Data Set I.</a></span><ul class="toc-item"><li><span><a href="#Check-of-Class-Balance" data-toc-modified-id="Check-of-Class-Balance-10.1.1"><span class="toc-item-num">10.1.1&nbsp;&nbsp;</span>Check of Class Balance</a></span></li></ul></li></ul></li><li><span><a href="#Feature-Selection-random-forest" data-toc-modified-id="Feature-Selection-random-forest-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Feature Selection random forest</a></span></li><li><span><a href="#define-columns-to-keep-for-prediction" data-toc-modified-id="define-columns-to-keep-for-prediction-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>define columns to keep for prediction</a></span></li><li><span><a href="#Classification-Models---Data-Set-I." data-toc-modified-id="Classification-Models---Data-Set-I.-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Classification Models - Data Set I.</a></span></li><li><span><a href="#Hyperparam-tuning-ranfo" data-toc-modified-id="Hyperparam-tuning-ranfo-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Hyperparam tuning ranfo</a></span><ul class="toc-item"><li><span><a href="#XGBoost---Data-Set-I." data-toc-modified-id="XGBoost---Data-Set-I.-14.1"><span class="toc-item-num">14.1&nbsp;&nbsp;</span>XGBoost - Data Set I.</a></span></li><li><span><a href="#Logistic-Regression---Data-Set-I." data-toc-modified-id="Logistic-Regression---Data-Set-I.-14.2"><span class="toc-item-num">14.2&nbsp;&nbsp;</span>Logistic Regression - Data Set I.</a></span></li><li><span><a href="#Random-Forest-Classifier-1" data-toc-modified-id="Random-Forest-Classifier-1-14.3"><span class="toc-item-num">14.3&nbsp;&nbsp;</span>Random Forest Classifier 1</a></span></li><li><span><a href="#Random-Forest-Classifier---Data-Set-I." data-toc-modified-id="Random-Forest-Classifier---Data-Set-I.-14.4"><span class="toc-item-num">14.4&nbsp;&nbsp;</span>Random Forest Classifier - Data Set I.</a></span></li></ul></li><li><span><a href="#Trading-Strategy---Data-Set-I" data-toc-modified-id="Trading-Strategy---Data-Set-I-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>Trading Strategy - Data Set I</a></span><ul class="toc-item"><li><span><a href="#Trading-Strategy-in-Combination-with-Logistic-Regression" data-toc-modified-id="Trading-Strategy-in-Combination-with-Logistic-Regression-15.1"><span class="toc-item-num">15.1&nbsp;&nbsp;</span>Trading Strategy in Combination with Logistic Regression</a></span></li></ul></li><li><span><a href="#Compare-with-year-2017" data-toc-modified-id="Compare-with-year-2017-16"><span class="toc-item-num">16&nbsp;&nbsp;</span>Compare with year 2017</a></span></li><li><span><a href="#Compare-with-year-2018" data-toc-modified-id="Compare-with-year-2018-17"><span class="toc-item-num">17&nbsp;&nbsp;</span>Compare with year 2018</a></span></li><li><span><a href="#Compare-with-year-2019" data-toc-modified-id="Compare-with-year-2019-18"><span class="toc-item-num">18&nbsp;&nbsp;</span>Compare with year 2019</a></span></li><li><span><a href="#Compare-with-year-20" data-toc-modified-id="Compare-with-year-20-19"><span class="toc-item-num">19&nbsp;&nbsp;</span>Compare with year 20</a></span></li><li><span><a href="#Save-Results-as-csv" data-toc-modified-id="Save-Results-as-csv-20"><span class="toc-item-num">20&nbsp;&nbsp;</span>Save Results as csv</a></span></li><li><span><a href="#Deepdive-in-prediction" data-toc-modified-id="Deepdive-in-prediction-21"><span class="toc-item-num">21&nbsp;&nbsp;</span>Deepdive in prediction</a></span></li><li><span><a href="#wie-war-der-gewinn/verlust-über-den-tag?-jede-stunde!-get-data:" data-toc-modified-id="wie-war-der-gewinn/verlust-über-den-tag?-jede-stunde!-get-data:-22"><span class="toc-item-num">22&nbsp;&nbsp;</span>wie war der gewinn/verlust über den tag? jede stunde! get data:</a></span></li><li><span><a href="#List-of-Data-Types" data-toc-modified-id="List-of-Data-Types-23"><span class="toc-item-num">23&nbsp;&nbsp;</span>List of Data Types</a></span></li><li><span><a href="#name-btc-feats-different" data-toc-modified-id="name-btc-feats-different-24"><span class="toc-item-num">24&nbsp;&nbsp;</span>name btc feats different</a></span></li></ul></div>

# # START Classification Models - Data Set Variations

# In[1]:


# Import of Modules and Packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from pycaret.classification import *

# Deep Neural Networks
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
import numpy as np

# PyCaret
import pycaret


# In[2]:


# Adjustment of Decimal Places
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', None)


# # Import Data Sets

# ## Data Set I.

# In[3]:


os.getcwd()


# In[4]:

featimpo_all = [6.0, 6.1, 6.15, 6.2, 6.25,
                6.3, 6.35, 6.4, 6.45, 6.5, 6.55, 6.6]

for featimpo in featimpo_all:

    os.chdir('{}JupyterLabDir\\Rest\\MA BTC\\Markus_Code_MA_final\\20201010'.format(
        os.getcwd().split('JupyterLabDir')[0]))

    # In[5]:

    datacsvpath = 'df_eth_tier1_tier2.csv'

    # In[6]:

    # CSV Import of Data Set I. - Blockchain Data
    df_dataset1 = pd.read_csv(datacsvpath, parse_dates=True, index_col='Date')

    # # import btc data

    # In[7]:

    df_dataset_btc = pd.read_csv(
        'df_btc_tier1_tier2.csv', parse_dates=True, index_col='Date')
    # List of Data Types
    l_dtypes_btc = df_dataset_btc.dtypes
    # name btc feats different
    df_dataset_btc.columns = [
        str(col) + '_btc' for col in df_dataset_btc.columns]

    # In[8]:

    # Merging all Blockchain DataFrame Metrics
    # issued = The total amount of new coins added to the current supply, i.e. minted coins or new coins released to the network.
    # current = The total amount of all coins ever created/issued, i.e. the circulating supply.
    df_dataset_btc = df_dataset_btc[["addresses/active_count_btc",
                                     "Closed Price USD_btc",
                                     "blockchain/block_count_btc",
                                     "supply/current_btc",
                                     "supply/issued_btc",
                                     "transactions/transfers_volume_from_exchanges_sum_btc",
                                     "Daily Return in USD_btc",
                                     ]]

    # In[9]:

    df_dataset_btc["addresses/active_count_btc_cumsum"] = df_dataset_btc["addresses/active_count_btc"].cumsum()

    # In[10]:

    df_dataset_btc["market/price_usd_close_btc_diff"] = df_dataset_btc["Closed Price USD_btc"].diff()
    df_dataset_btc[["Closed Price USD_btc",
                    "market/price_usd_close_btc_diff",
                    "addresses/active_count_btc_cumsum"]]

    # In[ ]:

    # # calculate meier multiple: 200 moving average current price

    # In[11]:

    df_dataset_btc = df_dataset_btc.sort_index(ascending=True)
    df_dataset_btc

    # In[12]:

    # min_periods = 1
    df_dataset_btc["200movingaverage_btc"] = pd.Series.rolling(
        df_dataset_btc["Closed Price USD_btc"], window=200,).mean()
    df_dataset_btc["mayer_multiple_btc"] = df_dataset_btc["Closed Price USD_btc"] / \
        df_dataset_btc["200movingaverage_btc"]
    df_dataset_btc["mayer_multiple_btc"]

    # In[13]:

    df_dataset_btc["blockchain/block_count_btc_cumsum"] = df_dataset_btc["blockchain/block_count_btc"].cumsum()

    # In[14]:

    # somehow there are blocks missing so i just added the number of missing blocks
    df_dataset_btc["blockchain/block_count_btc_cumsum"] = df_dataset_btc["blockchain/block_count_btc_cumsum"]+141381

    # In[15]:

    # we are only +- 35 precise
    df_dataset_btc["blockchain/block_count_btc_cumsum"].loc['2016-07-09']

    # In[16]:

    df_dataset_btc["blockchain/block_count_btc_cumsum"].loc['2012-11-28']

    # In[17]:

    # 0 = halving! 1 = no halving
    df_dataset_btc['halving'] = np.where(
        df_dataset_btc["blockchain/block_count_btc_cumsum"] % 210000 <= 50, 0, 1)

    # In[18]:

    #df_dataset_btc["halving"] = df_dataset_btc["blockchain/block_count_btc_cumsum"]

    # In[19]:

    df_dataset_btc["halving"]

    # In[20]:

    # the higher this value the
    df_dataset_btc["days_after_halving"] = df_dataset_btc.groupby(
        (df_dataset_btc["halving"] == 0).cumsum()).cumcount()
    df_dataset_btc["days_after_halving"].sort_values()

    # In[21]:

    df_dataset_btc["days_after_halving"].tail(20)

    # In[22]:

    df_dataset_btc[["addresses/active_count_btc",
                    "Closed Price USD_btc",
                    "blockchain/block_count_btc",
                    "supply/current_btc",
                    "supply/issued_btc"]]

    # In[23]:

    df_dataset_btc["stock_to_flow_ratio_btc"] = df_dataset_btc["supply/current_btc"] / \
        (df_dataset_btc["supply/issued_btc"] * 365)

    # In[24]:

    df_dataset1["stock_to_flow_ratio"] = df_dataset1["supply/current"] / \
        (df_dataset1["supply/issued"] * 365)

    # In[25]:

    # min_periods = 1
    df_dataset1["200movingaverage"] = pd.Series.rolling(
        df_dataset1["Closed Price USD"], window=200,).mean()
    df_dataset1["mayer_multiple"] = df_dataset1["Closed Price USD"] / \
        df_dataset1["200movingaverage"]
    df_dataset1["mayer_multiple"]

    # In[26]:

    import plotly.graph_objects as go

    # # Create traces
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_dataset_btc.index,
    #                          y=df_dataset_btc["Closed Price USD_btc"],
    #                          mode='lines',
    #                          name='price_usd_close_btc'))
    # fig.add_trace(go.Scatter(x=df_dataset_btc.index,
    #                          y=df_dataset_btc["200movingaverage_btc"],
    #                          mode='lines',
    #                          name='200movingaverage_btcv'))
    # fig.add_trace(go.Scatter(x=df_dataset_btc.index,
    #                          y=df_dataset_btc["mayer_multiple_btc"]*5000,
    #                          mode='lines',
    #                          name='mayer_multiple_btc*5000'))
    # fig.add_trace(go.Scatter(x=df_dataset1.index,
    #                          y=df_dataset1["mayer_multiple"]*5000,
    #                          mode='lines',
    #                          name='mayer_multiple*5000'))

    # fig.show()

    # # transfers to/from exchanges diff und sliding window

    # In[27]:

    df_dataset1["transfers_volume_from_exchanges_sum30movave"] = pd.Series.rolling(
        df_dataset1["transactions/transfers_volume_from_exchanges_sum"], window=30).mean()
    df_dataset1["transfers_volume_from_exchanges_sum7movave"] = pd.Series.rolling(
        df_dataset1["transactions/transfers_volume_from_exchanges_sum"], window=7).mean()
    df_dataset1["transfers_volume_from_exchanges_sum3movave"] = pd.Series.rolling(
        df_dataset1["transactions/transfers_volume_from_exchanges_sum"], window=3).mean()
    df_dataset_btc["transfers_volume_from_exchanges_sum30movave_btc"] = pd.Series.rolling(
        df_dataset_btc["transactions/transfers_volume_from_exchanges_sum_btc"], window=30).mean()
    df_dataset_btc["transfers_volume_from_exchanges_sum7movave_btc"] = pd.Series.rolling(
        df_dataset_btc["transactions/transfers_volume_from_exchanges_sum_btc"], window=7).mean()
    df_dataset_btc["transfers_volume_from_exchanges_sum3movave_btc"] = pd.Series.rolling(
        df_dataset_btc["transactions/transfers_volume_from_exchanges_sum_btc"], window=3).mean()

    # In[28]:

    df_dataset1 = df_dataset1.sort_index(ascending=True)
    df_dataset_btc = df_dataset_btc.sort_index(ascending=True)
    df_dataset1["transactions/transfers_volume_from_exchanges_sum_diff"] = df_dataset1["transactions/transfers_volume_from_exchanges_sum"].diff()
    df_dataset_btc["transactions/transfers_volume_from_exchanges_sum_diff_btc"] = df_dataset_btc["transactions/transfers_volume_from_exchanges_sum_btc"].diff()
    df_dataset1 = df_dataset1.sort_index(ascending=False)
    df_dataset_btc = df_dataset_btc.sort_index(ascending=False)

    # In[29]:

    df_dataset1 = df_dataset1.sort_index(ascending=True)
    df_dataset_btc = df_dataset_btc.sort_index(ascending=True)

    # In[30]:

    for x in df_dataset1.columns:
        df_dataset1["{}_adiff".format(x)] = df_dataset1["{}".format(x)].diff()

    # In[31]:

    df_dataset1 = df_dataset1.sort_index(ascending=False)
    df_dataset_btc = df_dataset_btc.sort_index(ascending=False)

    # # kick out meyer multiple eth because of data loss

    # In[32]:

    #df_dataset1=df_dataset1.drop('mayer_multiple', axis=1)

    # # Bring df_dataset1 and df_dataset_btc together

    # In[33]:

    df_dataset1 = pd.concat(
        [df_dataset1, df_dataset_btc], axis=1, join='inner')
    df_dataset1

    # In[34]:

    df_dataset1.columns

    # In[ ]:

    # In[35]:

    # Daily Return in USD in row 2011-07-16 means the change in "btc dollar price" from 2011-07-15 23:59 to 2011-07-16 23:59
    # in row 2011-07-17 means the change in "btc dollar price" from 2011-07-16 23:59 to 2011-07-17 23:59

    df_dataset1[["market/price_usd_close_btc_diff",
                 "Closed Price USD_btc",
                 "Closed Price USD",
                 'Daily Return in USD', ]]

    # In[36]:

    df_dataset1["Daily Return in USD"]

    # In[37]:

    # recalculate the correct "Daily Return in USD" for Friday-Monday problem
    #df_dataset1["Daily Return in USD"] = df_dataset1["Closed Price USD"].diff()

    # In[38]:

    df_dataset1[["Daily Return in USD", "Closed Price USD"]]

    # In[39]:

    df_dataset1["Log Price in USD"] = pd.DataFrame(
        np.log(df_dataset1['Closed Price USD']))
    df_dataset1["Volatility Daily Log Return in Percent 5D"] = 0

    # In[40]:

    # Data Set Info
    df_dataset1.info()

    # In[41]:

    # List of Data Types
    l_dtypes = df_dataset1.dtypes

    # In[42]:

    # Counter
    counter = 0

    # Cross-Check of Data Types
    for i in l_dtypes:
        if (i == float):
            counter = counter+1
        else:
            counter = counter

    # Number of Float Data Types
    print(counter)

    # In[43]:

    # Data Types
    df_dataset1.dtypes

    # In[44]:

    # Summary Statistics
    df_dataset1.describe()

    # In[45]:

    # Dataset Tail
    df_dataset1.tail()

    # In[46]:

    len(df_dataset1.columns)

    # ## Prediction Shift

    # In[47]:

    for x in df_dataset1.columns:
        print(x)

    # ### Data Set I.

    # In[48]:

    df_dataset1.columns

    # In[49]:

    df_dataset1[['Daily Return in USD', 'Log Price in USD',
                 'Volatility Daily Log Return in Percent 5D', 'Closed Price USD']]

    # In[50]:

    df_dataset1 = df_dataset1.sort_index(ascending=False)

    # In[ ]:

    # In[51]:

    # Shift of Target Column by 1 Day
    df_dataset1['Daily Return in USD_ohneshift'] = df_dataset1['Daily Return in USD']
    df_dataset1["percentage_daily_return_bef_shift"] = df_dataset1["Daily Return in USD_ohneshift"] / \
        df_dataset1["Closed Price USD"]

    # In[ ]:

    # In[ ]:

    # In[52]:

    df_dataset1['Daily Return in USD'] = df_dataset1['Daily Return in USD'].shift(
        1)

    # In[53]:

    df_dataset1[["market/price_usd_close_btc_diff",
                 "Closed Price USD_btc",
                 "Closed Price USD",
                 'Daily Return in USD']]

    # In[ ]:

    # In[54]:

    df_dataset1["percentage_daily_return_bef_shift_btc"] = df_dataset1["Daily Return in USD_btc"] / \
        df_dataset1["Closed Price USD_btc"]

    # In[55]:

    df_dataset1[['Daily Return in USD', 'addresses/active_count', 'Log Price in USD',
                 'Volatility Daily Log Return in Percent 5D', 'Closed Price USD']]

    # In[56]:

    df_dataset1['market/price_usd_close_cummax'] = df_dataset1['Closed Price USD'].cummax()
    df_dataset1['price_usd_close_percent_of_maxtilnow'] = df_dataset1['Closed Price USD'] / \
        df_dataset1['market/price_usd_close_cummax']

    # In[ ]:

    # In[57]:

    # Check Shift
    df_dataset1[['Log Price in USD',
                 'Volatility Daily Log Return in Percent 5D', 'Closed Price USD', "percentage_daily_return_bef_shift"]].tail()

    # In[58]:

    df_dataset1 = df_dataset1.drop(
        'Volatility Daily Log Return in Percent 5D', axis=1)

    # In[59]:

    # Check Empty Cells
    df_dataset1.isnull().sum()

    # In[60]:

    # daily return: von gestern auf heute die preisveränderung deswegen shift später
    df_dataset1[df_dataset1.isna().any(axis=1)]

    # In[ ]:

    # In[ ]:

    # In[61]:

    # Re-Check Empty Cells
    df_dataset1.isnull().sum()

    # In[62]:

    df_dataset1[["market/price_usd_close_btc_diff",
                 "Closed Price USD_btc",
                 "Closed Price USD",
                 'Daily Return in USD', ]]

    # In[63]:

    #df_dataset1['market/price_usd_close'] = df_dataset1["Closed Price USD"]

    # # get yearly btc dollar increase for traditional investment

    # In[64]:

    print("year 2017")
    print(df_dataset1.loc["2017-12-31"]["Closed Price USD"]
          [0] / df_dataset1.loc["2017-01-01"]["Closed Price USD"][0])
    print("year 2018")
    print(df_dataset1.loc["2018-12-31"]["Closed Price USD"]
          [0] / df_dataset1.loc["2018-01-01"]["Closed Price USD"][0])
    print("year 2019")
    print(df_dataset1.loc["2019-12-31"]["Closed Price USD"]
          [0] / df_dataset1.loc["2019-01-01"]["Closed Price USD"][0])
    print("year 2020")
    print(df_dataset1.loc["2020-12-31"]["Closed Price USD"]
          [0] / df_dataset1.loc["2020-01-01"]["Closed Price USD"][0])

    # # Classification Setup

    # ## Create Data Set Copies for Trading Strategy

    # In[65]:

    # Data Set Copy for Trading Extension
    df_dataset1_copy = df_dataset1.copy()
    df_dataset1_copy[['Closed Price USD',
                      'Daily Return in USD', 'Daily Return in Percent',
                      'Daily Log Return in Percent', 'Log Price in USD',
                      'Daily Return in USD_ohneshift', 'percentage_daily_return_bef_shift',
                      'market/price_usd_close_cummax',
                      'price_usd_close_percent_of_maxtilnow']].head(10)

    # In[66]:

    df_dataset1_copy.columns

    # ## Categorization - Data Set I.

    # In[67]:

    # Categorization
    # Class 1 for all Returns >= 0 USD
    df_dataset1.loc[df_dataset1['Daily Return in USD'] >= 0, 'Class'] = 1
    # Class 0 for all Returns < 0 USD
    df_dataset1.loc[df_dataset1['Daily Return in USD'] < 0, 'Class'] = 0

    # In[68]:

    # Data Set Head
    df_dataset1.head()

    # In[69]:

    for x in range(len(df_dataset1.columns)):
        print(x, df_dataset1.columns[x])

    # In[70]:

    # Delete Original Target Column
    df_dataset1 = df_dataset1.drop(['Daily Return in USD'], axis=1)

    # In[71]:

    # Check Class Balance
    df_dataset1['Class'].value_counts()

    # In[72]:

    # Data Set Head
    df_dataset1.head()

    # In[73]:

    df_dataset1["addresses/active_count_cumsum"] = df_dataset1["addresses/active_count"][::-1].cumsum()
    # 1 active adress has the price:
    df_dataset1["price_per_one_active_address_cumsum"] = df_dataset1["Closed Price USD"] / \
        df_dataset1["addresses/active_count_cumsum"]
    df_dataset1["dailyreturn_per_one_active_address_cumsum"] = df_dataset1["Daily Return in USD_ohneshift"] / \
        df_dataset1["addresses/active_count_cumsum"]

    # In[74]:

    df_dataset1["dailyreturn_per_one_active_address_cumsum"].iloc[0]

    # In[75]:

    df_dataset1.to_csv("ethtarget_4btc_hptuning.csv",
                       sep='\t', index=True, index_label='Date')

    # In[76]:

    # TODO btc retrun of next 24h if active address is x (same than eth)
    # map btc active addresses with eth

    # In[77]:

    #df_dataset1=df_dataset1[['addresses/new_non_zero_count', 'addresses/active_count', 'addresses/sending_count', 'addresses/receiving_count', 'addresses/count', 'blockchain/block_height', 'blockchain/block_count', 'blockchain/block_interval_mean', 'blockchain/block_interval_median', 'blockchain/block_size_sum', 'blockchain/block_size_mean', 'fees/volume_sum', 'fees/volume_mean', 'fees/volume_median', 'fees/gas_used_sum', 'fees/gas_used_mean', 'fees/gas_used_median', 'fees/gas_price_mean', 'fees/gas_price_median', 'fees/gas_limit_tx_mean', 'fees/gas_limit_tx_median', 'indicators/sopr', 'market/price_drawdown_relative', 'market/marketcap_usd', 'mining/difficulty_latest', 'mining/hash_rate_mean', 'supply/current', 'transactions/count', 'transactions/rate', 'transactions/transfers_count', 'transactions/transfers_rate', 'transactions/transfers_volume_sum', 'transactions/transfers_volume_mean', 'transactions/transfers_volume_median', 'addresses/non_zero_count', 'addresses/min_point_zero_1_count', 'addresses/min_point_1_count', 'addresses/min_1_count', 'addresses/min_10_count', 'addresses/min_100_count', 'addresses/min_1k_count', 'addresses/min_10k_count', 'addresses/min_32_count', 'distribution/balance_1pct_holders', 'distribution/gini', 'distribution/herfindahl', 'distribution/supply_contracts', 'transactions/transfers_volume_to_exchanges_sum', 'transactions/transfers_volume_from_exchanges_sum', 'transactions/transfers_volume_exchanges_net', 'transactions/transfers_to_exchanges_count', 'transactions/transfers_from_exchanges_count', 'transactions/transfers_volume_to_exchanges_mean', 'transactions/transfers_volume_from_exchanges_mean', 'distribution/balance_exchanges', 'fees/fee_ratio_multiple', 'indicators/net_unrealized_profit_loss', 'indicators/unrealized_profit', 'indicators/unrealized_loss', 'indicators/cdd', 'indicators/liveliness', 'indicators/average_dormancy', 'indicators/asol', 'indicators/msol', 'indicators/nvt', 'indicators/nvts', 'indicators/velocity', 'market/marketcap_realized_usd', 'market/mvrv', 'mining/thermocap', 'mining/marketcap_thermocap_ratio', 'mining/revenue_sum', 'mining/revenue_from_fees', 'supply/profit_relative', 'supply/profit_sum', 'supply/loss_sum', 'supply/active_24h', 'supply/active_1d_1w', 'supply/active_1w_1m', 'supply/active_1m_3m', 'supply/active_3m_6m', 'supply/active_6m_12m', 'supply/active_1y_2y', 'supply/active_2y_3y', 'supply/active_3y_5y', 'supply/active_5y_7y', 'supply/active_7y_10y', 'supply/active_more_10y', 'supply/issued', 'supply/inflation_rate', 'transactions/contract_calls_internal_count', 'Closed Price USD', 'Daily Return in Percent', 'Daily Log Return in Percent', 'Log Price in USD', '200movingaverage', 'mayer_multiple', 'addresses/active_count_btc', 'Closed Price USD_btc', 'addresses/active_count_btc_cumsum', 'market/price_usd_close_btc_diff', '200movingaverage_btc', 'mayer_multiple_btc', 'Daily Return in USD_ohneshift', 'percentage_daily_return_bef_shift', 'market/price_usd_close_cummax', 'price_usd_close_percent_of_maxtilnow', 'Class', 'addresses/active_count_cumsum', 'price_per_one_active_address_cumsum', 'dailyreturn_per_one_active_address_cumsum']]

    # In[78]:

    len(df_dataset1.columns)

    # In[79]:

    games_to_predict = 20

    # Delete Empty Row
    predict_change_for24h = df_dataset1.head(games_to_predict)
    # delete first column because contains null value for target!
    df_dataset1 = df_dataset1.iloc[games_to_predict:]
    predict_change_for24h

    # In[80]:

    len(predict_change_for24h.columns)

    # # Model Preparation - Data Set I.

    # ## Train Test Split - Data Set I.

    # In[81]:

    df_dataset1 = df_dataset1.dropna()

    # In[82]:

    # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
    predictors_columnnames = df_dataset1.drop(['Class'], axis=1)
    predictors = df_dataset1.drop(['Class'], axis=1).values.astype(np.float32)

    # In[83]:

    # Setup of Categorized Target Variable
    target = df_dataset1['Class'].astype(np.float32)

    # In[ ]:

    # In[84]:

    len(df_dataset1.columns)

    # In[85]:

    predictors.shape

    # In[86]:

    df_dataset1_allbut2020 = df_dataset1.loc[df_dataset1.index.year != 2020]
    df_dataset1_2020 = df_dataset1.loc[df_dataset1.index.year == 2020]
    df_dataset1_allbut2019 = df_dataset1.loc[df_dataset1.index.year != 2019]
    df_dataset1_2019 = df_dataset1.loc[df_dataset1.index.year == 2019]
    df_dataset1_allbut2018 = df_dataset1.loc[df_dataset1.index.year != 2018]
    df_dataset1_2018 = df_dataset1.loc[df_dataset1.index.year == 2018]
    df_dataset1_allbut2017 = df_dataset1.loc[df_dataset1.index.year != 2017]
    df_dataset1_2017 = df_dataset1.loc[df_dataset1.index.year == 2017]

    # In[87]:

    # Setup of Categorized Target Variable
    y_traintarget_allbut2017 = df_dataset1_allbut2017['Class'].astype(
        np.float32)
    y_testtarget_2017 = df_dataset1_2017['Class'].astype(np.float32)
    # Setup of Categorized Target Variable
    y_traintarget_allbut2018 = df_dataset1_allbut2018['Class'].astype(
        np.float32)
    y_testtarget_2018 = df_dataset1_2018['Class'].astype(np.float32)
    # Setup of Categorized Target Variable
    y_traintarget_allbut2019 = df_dataset1_allbut2019['Class'].astype(
        np.float32)
    y_testtarget_2019 = df_dataset1_2019['Class'].astype(np.float32)
    # Setup of Categorized Target Variable
    y_traintarget_allbut2020 = df_dataset1_allbut2020['Class'].astype(
        np.float32)
    y_testtarget_2020 = df_dataset1_2020['Class'].astype(np.float32)

    # In[ ]:

    # In[ ]:

    # In[88]:

    # Predictor Dimensions
    predictors.shape

    # In[ ]:

    # In[89]:

    for x in range(len(predictors[0])):
        print(x, predictors[0][x])

    # In[90]:

    # Target Dimensions
    target.shape

    # In[91]:

    # Target Dimensions
    predictors.shape

    # In[92]:

    # Train-Test-Split Function to Create Training and Test Data Set
    X_train, X_test, y_train, y_test = train_test_split(
        predictors, target, test_size=0.3, random_state=41, stratify=target)

    # In[93]:

    # Dimensions of Predictors for Training
    X_train.shape

    # In[94]:

    # Dimensions of Target for Training
    y_train.shape

    # In[95]:

    # Number of Predictors
    n_nodes = predictors.shape[1]
    n_nodes

    # ### Check of Class Balance

    # In[96]:

    # Check Training Target of Class 0
    y_train[y_train == 0].count()

    # In[97]:

    # Check Training Target of Class 1
    y_train[y_train == 1].count()

    # In[98]:

    # Check Test Target of Class 0
    y_test[y_test == 0].count()

    # In[99]:

    # Check Test Target of Class 1
    y_test[y_test == 1].count()

    # In[ ]:

    # In[ ]:

    # In[ ]:

    # # Feature Selection random forest

    # In[100]:

    import pandas as pd
    from sklearn.feature_selection import SelectFromModel

    # # # define RFE
    # sel = SelectFromModel(RandomForestClassifier(bootstrap=True, min_samples_leaf=5,
    #                               min_samples_split=2, n_estimators=234,
    #                               max_features= 'auto', max_depth= None))

    # # define RFE
    sel = SelectFromModel(RandomForestClassifier(bootstrap=False, min_samples_leaf=4,
                                                 min_samples_split=5, n_estimators=200))

    # fit RFE
    sel.fit(X_train, y_train)
    # summarize all features
    sel.get_support()

    # summarize all features
    print(sel.threshold_)
    print(sel.estimator_.feature_importances_*1000)

    for i in range(X_train.shape[1]):
        print('Column: %f, Selected %s, col %s' % (sel.estimator_.feature_importances_[
              i]*1000, sel.get_support()[i], predictors_columnnames.columns[i]))

    # In[101]:

    df_dataset1_columns_keep_imp = []
    df_dataset1_columns_keep = []
    rank_delete_list = []
    df_dataset1_columns_delete_imp = []
    df_dataset1_columns_delete = []

    for i in range(X_train.shape[1]):
        # if sel.get_support()[i] == False:
        if sel.estimator_.feature_importances_[i]*1000 <= featimpo:
            rank_delete_list.append(i)
            df_dataset1_columns_delete_imp.append(
                [predictors_columnnames.columns[i], sel.estimator_.feature_importances_[i]*1000])
            df_dataset1_columns_delete.append(
                predictors_columnnames.columns[i])

        else:
            print('Column: %d, Selected %s, col %s' %
                  (i, sel.get_support()[i], predictors_columnnames.columns[i]))
            df_dataset1_columns_keep_imp.append(
                [predictors_columnnames.columns[i], sel.estimator_.feature_importances_[i]*1000])
            df_dataset1_columns_keep.append(predictors_columnnames.columns[i])

    # In[102]:

    df_dataset1_columns_keep

    # In[103]:

    # sort list of lists by second element
    df_dataset1_columns_delete.sort(key=lambda x: x[1])

    # In[104]:

    df_dataset1_columns_delete

    # In[105]:

    X_train.shape[1]

    # In[106]:

    predictors.shape

    # In[107]:

    df_dataset1 = df_dataset1[df_dataset1_columns_keep]
    predictors = np.delete(predictors, rank_delete_list, 1)

    # In[108]:

    df_dataset1_allbut2020 = df_dataset1.loc[df_dataset1.index.year != 2020]
    df_dataset1_2020 = df_dataset1.loc[df_dataset1.index.year == 2020]
    df_dataset1_allbut2019 = df_dataset1.loc[df_dataset1.index.year != 2019]
    df_dataset1_2019 = df_dataset1.loc[df_dataset1.index.year == 2019]
    df_dataset1_allbut2018 = df_dataset1.loc[df_dataset1.index.year != 2018]
    df_dataset1_2018 = df_dataset1.loc[df_dataset1.index.year == 2018]
    df_dataset1_allbut2017 = df_dataset1.loc[df_dataset1.index.year != 2017]
    df_dataset1_2017 = df_dataset1.loc[df_dataset1.index.year == 2017]

    # In[109]:

    print(len(df_dataset1_allbut2020.columns))
    print(len(df_dataset1_allbut2019.columns))
    print(len(df_dataset1_allbut2018.columns))
    print(len(df_dataset1_2018.columns))

    # In[110]:

    predictors.shape

    # In[111]:

    anzahlfeats = predictors.shape[1]

    # In[112]:

    # Instantiate Classifier Logistic Regression
    lreg = LogisticRegression(max_iter=1200)

    # Train Classifier
    lreg.fit(X_train, y_train)

    # Prediction of Classes
    y_pred = lreg.predict(X_test)

    # Display Confusion Matrix and Classification Report
    print(confusion_matrix(y_test, y_pred))
    #     print(classification_report(y_test, y_pred))

    # Prediction of Probabilities
    y_pred_prob = lreg.predict_proba(X_test)[:, 1]

    # Create Data Frame of Actual Values - y_test
    df_y_test = pd.DataFrame(y_test)

    # Add Predicted Probabilities to Data Frame
    df_y_test["y_pred_prob"] = y_pred_prob

    # Add Predicted Classes (0 or 1) to Data Frame to Cross-Check Predicted Probabilities
    df_y_test["y_pred"] = y_pred

    # In[ ]:

    # In[113]:

    df_y_test = df_y_test.sort_index()
    df_y_test

    # In[114]:

    df_y_test[df_y_test['Class'] != df_y_test['y_pred']]

    # In[115]:

    df_y_test[df_y_test['Class'] == df_y_test['y_pred']]

    # In[116]:

    import plotly.graph_objects as go

    # Create traces
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_y_test.index, y=df_y_test['Class'],
    #                          mode='lines+markers',
    #                          name='Class - reality'))
    # fig.add_trace(go.Scatter(x=df_y_test.index, y=df_y_test['y_pred_prob'],
    #                          mode='lines+markers',
    #                          name='y_pred'))
    # fig.show()

    # In[117]:

    # Select Actual Daily Returns in USD from the Data Set Copy
    df_dataset1_copy_dailyreturns = df_dataset1_copy[['Daily Return in USD']]

    # # Merge Predictions and Daily Returns Data Frame on the Column "Date"
    # df_y_test_withdailyreturns = pd.merge(
    #     df_y_test, df_dataset1_copy_dailyreturns, on='Date')

    # df_dataset1_copy_percentage = df_dataset1_copy[['percentage_daily_return']]
    # df_y_test_withdailyreturns = pd.merge(
    #     df_y_test_withdailyreturns, df_dataset1_copy_percentage, on='Date')

    # # Add Bitcoin Price in USD
    # df_y_test_withdailyreturns = pd.merge(
    #     df_y_test_withdailyreturns, df_dataset1['market/price_usd_close'], on='Date')

    # df_y_test_withdailyreturns['y_pred_prob_bereinigt'] = np.where(
    #     df_y_test_withdailyreturns['y_pred_prob'] < 0.5, 1-df_y_test_withdailyreturns['y_pred_prob'], df_y_test_withdailyreturns['y_pred_prob'])

    # df_y_test_withdailyreturns["Einsatz"] = einsatz * \
    #     df_y_test_withdailyreturns["y_pred_prob_bereinigt"]  # **2

    # In[ ]:

    # # define columns to keep for prediction

    # In[118]:

    df_dataset1_columns_delete

    # In[119]:

    df_dataset1

    # In[120]:

    df_dataset1 = df_dataset1.dropna()

    # # Classification Models - Data Set I.

    # In[121]:

    # Function To Show Accuracy Mean and Accuracy Standard Deviation

    def show_scores(scores):
        # Scores Series
        print('Scores:', scores)
        # Score Mean
        print('Mean:', np.mean(scores))
        # Score Standard Deviation
        print('Standard Deviation:', np.std(scores))

    # # Hyperparam tuning ranfo

    # In[122]:

    train_features = X_train
    train_labels = y_train
    test_features = X_test
    test_labels = y_test

    # In[123]:

    from numpy import inf

    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        #print("predictions {} _test_labels {} ".format(predictions, test_labels))
        errors = int(np.sum((predictions == test_labels) == False))
        alle = len(test_labels.values)
        x = errors / alle
        mape = 100 * errors / alle
        accuracy = 100 - mape
        print('Model Performance')
        print('Amount Errors total {} of {} '.format(np.mean(errors), alle))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        return accuracy

    base_model = RandomForestClassifier(n_estimators=500)
    base_model.fit(train_features, train_labels)
    base_accuracy = evaluate(base_model, test_features, test_labels)

    # In[124]:

    predictors.shape

    # In[125]:

    target.shape

    # In[ ]:

    # In[126]:

    {'subsample': 0.7, 'objective': 'binary:logistic',
     'n_estimators': 600, 'min_child_weight': 7, 'max_depth': 2,
     'learning_rate': 0.1, 'gamma': 4, 'colsample_bytree': 1.0,
     'base_score': 0.8}

    # In[127]:

    # from xgboost import XGBClassifier

    # multievaluate = []

    # # base_model = RandomForestClassifier(n_estimators = 500, min_samples_leaf = 4)
    # base_model = XGBClassifier(base_score=0.8, colsample_bytree=1.0, gamma=4,
    #               learning_rate=0.1, max_depth=2,
    #               min_child_weight=7, n_estimators=600,
    #               objective='binary:logistic',
    #               subsample=0.7)

    # for x in range(10):
    #   # Split Data into Training and Testing
    #   X_train, X_test, y_train, y_test = train_test_split(
    #     predictors, target, test_size=0.3, stratify=target)
    #   train_features = X_train
    #   train_labels = y_train
    #   test_features = X_test
    #   test_labels = y_test

    #   base_model.fit(train_features, train_labels)

    #   multievaluate.append(evaluate(base_model, test_features, test_labels))

    # # mean of 10 accuracy runs
    # print("MEAN ACC:")
    # np.mean( multievaluate )

    # In[128]:

    # from sklearn.model_selection import RandomizedSearchCV
    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # # Create the random grid
    # param_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    # print(param_grid)

    # In[129]:

    # from sklearn.model_selection import GridSearchCV

    # # Create a based model
    # model = RandomForestClassifier()
    # # Instantiate the grid search model
    # grid_search = GridSearchCV(estimator = model, param_grid = param_grid,
    #                           cv = 3, n_jobs = -1, verbose = 2)

    # print(grid_search)

    # In[130]:

    # # Fit the grid search to the data
    # grid_search.fit(train_features, train_labels)
    # grid_search.best_params_

    # In[131]:

    # best_params = grid_search.best_params_
    # best_params

    # In[132]:

    # multievaluate = []

    # for x in range(10):
    #   # Split Data into Training and Testing
    #   X_train, X_test, y_train, y_test = train_test_split(
    #     predictors, target, test_size=0.3, stratify=target)
    #   train_features = X_train
    #   train_labels = y_train
    #   test_features = X_test
    #   test_labels = y_test

    #   multievaluate.append(evaluate(best_grid, test_features, test_labels))

    # # mean of 10 accuracy runs
    # print("MEAN ACC:")
    # np.mean( multievaluate )

    # In[133]:

    # import xlsxwriter
    # # Adjustment of Decimal Places
    # pd.options.display.float_format = '{:.2f}'.format
    # pd.options.display.max_columns = None
    # pd.options.display.max_rows = None

    # In[134]:

    # import time
    # import statistics

    # result_to_csv = []

    # result_to_csv.append(("np.mean( multievaluate )", np.mean( multievaluate ) ))

    # df_zusammen = pd.DataFrame(result_to_csv)
    # df_zusammen["best_grid"] = pd.Series([best_grid])
    # df_zusammen["best_params"] = pd.Series([best_params])
    # df_zusammen["param_grid"] = pd.Series([param_grid])
    # df_zusammen["X_train.shape"] = pd.Series([X_train.shape])
    # df_zusammen["X_test.shape"] = pd.Series([X_test.shape])
    # df_zusammen["datacsvpath"] = pd.Series([datacsvpath])
    # df_zusammen["df_dataset1.columns"] = pd.Series([df_dataset1.columns.tolist()])

    # out_path = "hptuning_{}_{}_ppd{}.xlsx".format(int(time.time()), str(model)[:10], str(round(  np.mean( multievaluate ), 2)) )

    # writer = pd.ExcelWriter(out_path , engine='xlsxwriter')
    # df_zusammen.to_excel(writer, sheet_name='Tabelle1')
    # writer.save()

    # In[ ]:

    # In[ ]:

    # ## XGBoost - Data Set I.

    # In[135]:

    # Test Accuracy
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    # Instantiate Classifier - LogisticRegression
    model = XGBClassifier()

    accuarylist = []

    for x in range(4):
        # Split Data into Training and Testing
        X_train, X_test, y_train, y_test = train_test_split(
            predictors, target, test_size=0.3, stratify=target)

        # Train Classifier
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        accuarylist.append(accuracy * 100.0)

    meanacc = np.mean(accuarylist)

    # In[136]:

    # Prediction of Classes
    y_pred = model.predict(X_test)

    # In[137]:

    # Prediction of Probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # In[138]:

    # Select Predicted Probabilities
    y_pred_prob[1:11]

    # In[139]:

    # Select Actual Classes
    np.array(y_test[1:11])

    # In[140]:

    # Train Accuracy
    model.score(X_train, y_train)

    # In[141]:

    # Test Accuracy
    for x in range(10):
        print(model.score(X_test, y_test))

    # In[142]:

    # Output of Confusion Matrix
    print(confusion_matrix(y_test, y_pred))

    # In[143]:

    # Output of Classification Report
    print(classification_report(y_test, y_pred))

    # In[144]:

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # In[145]:

    # 10-fold Cross Validation including Stratification
    skf = StratifiedKFold(shuffle=True, n_splits=10)

    # In[146]:

    # Compute and print AUC Score
    print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

    # In[147]:

    # Cross-Validation for AUC scores
    cv_auc = cross_val_score(model, predictors, target,
                             cv=skf, scoring='roc_auc')

    # In[148]:

    # Print List of AUC scores
    print("AUC scores computed using 10-fold cross-validation: {}".format(cv_auc))

    # In[149]:

    # Cross-Validation for Accuracy Scores
    cv_acc = cross_val_score(model, predictors, target,
                             cv=skf, scoring='accuracy')

    # In[150]:

    # Print List of Accuracy Scores
    print("Accuracy scores computed using 10-fold cross-validation: {}".format(cv_acc))

    # In[151]:

    # Accuracy Results - Mean and Standard Deviation
    show_scores(cv_acc)

    # ## Logistic Regression - Data Set I.

    # In[152]:

    # Instantiate Classifier - LogisticRegression
    lreg = LogisticRegression(C=1000, max_iter=1200,
                              penalty='l1', solver='liblinear')

    # In[153]:

    # Instantiation of MinMaxScaler
    norm_scaler = MinMaxScaler()

    # Training and Application of Feature Scaler on Training Data of Predictors
    X_train_scaled = pd.DataFrame(norm_scaler.fit_transform(X_train))

    # Application of Feature Scaler on Test Data of Predictors
    X_test_scaled = norm_scaler.transform(X_test)

    # In[154]:

    # Train Classifier
    lreg.fit(X_train_scaled, y_train)

    # In[155]:

    # Prediction of Classes
    y_pred = lreg.predict(X_test_scaled)

    # In[156]:

    # Prediction of Probabilities
    y_pred_prob = lreg.predict_proba(X_test_scaled)[:, 1]

    # In[157]:

    # Select Predicted Probabilities
    y_pred_prob[1:11]

    # In[158]:

    # Select Actual Classes
    np.array(y_test[1:11])

    # In[159]:

    # Train Accuracy
    lreg.score(X_train_scaled, y_train)

    # In[160]:

    # Test Accuracy
    lreg.score(X_test_scaled, y_test)

    # In[161]:

    # Output of Confusion Matrix
    print(confusion_matrix(y_test, y_pred))

    # In[162]:

    # Output of Classification Report
    print(classification_report(y_test, y_pred))

    # In[163]:

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # In[164]:

    # 10-fold Cross Validation including Stratification
    skf = StratifiedKFold(shuffle=True, n_splits=10)

    # In[165]:

    # Compute and print AUC Score
    print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

    # In[166]:

    # # Cross-Validation for AUC scores
    # cv_auc = cross_val_score(lreg, predictors_scaled, target, cv=skf, scoring='roc_auc')

    # In[167]:

    # # Print List of AUC scores
    # print("AUC scores computed using 10-fold cross-validation: {}".format(cv_auc))

    # In[168]:

    # Cross-Validation for Accuracy Scores
    #cv_acc = cross_val_score(lreg, predictors_scaled, target, cv=skf, scoring='accuracy')

    # In[169]:

    # # Print List of Accuracy Scores
    # print("Accuracy scores computed using 10-fold cross-validation: {}".format(cv_acc))

    # In[170]:

    # # Accuracy Results - Mean and Standard Deviation
    # show_scores(cv_acc)

    # ## Random Forest Classifier 1

    # In[171]:

    # Test Accuracy
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    # # 5 run {'bootstrap': True, 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 234}
    # rfc = RandomForestClassifier(bootstrap= True, max_depth=None, max_features='auto', min_samples_leaf=5,
    #                        min_samples_split=2, n_estimators=234)

    lreg = RandomForestClassifier(bootstrap=False, min_samples_leaf=4,
                                  min_samples_split=5, n_estimators=200)
    accuarylist = []

    for x in range(2):
        # Split Data into Training and Testing
        X_train, X_test, y_train, y_test = train_test_split(
            predictors, target, test_size=0.3, stratify=target)

        # Train Classifier
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        accuarylist.append(accuracy * 100.0)

    meanacc = np.mean(accuarylist)

    # In[172]:

    # Prediction of Classes
    y_pred = model.predict(X_test)

    # In[173]:

    # Prediction of Probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # In[174]:

    # Select Predicted Probabilities
    y_pred_prob[1:11]

    # In[175]:

    # Select Actual Classes
    np.array(y_test[1:11])

    # In[176]:

    # Train Accuracy
    model.score(X_train, y_train)

    # In[177]:

    # Test Accuracy
    for x in range(10):
        print(model.score(X_test, y_test))

    # In[178]:

    # Output of Confusion Matrix
    print(confusion_matrix(y_test, y_pred))

    # In[179]:

    # Output of Classification Report
    print(classification_report(y_test, y_pred))

    # In[180]:

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # In[181]:

    # 10-fold Cross Validation including Stratification
    skf = StratifiedKFold(shuffle=True, n_splits=10)

    # In[182]:

    # Compute and print AUC Score
    print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

    # In[183]:

    # Cross-Validation for AUC scores
    cv_auc = cross_val_score(model, predictors, target,
                             cv=skf, scoring='roc_auc')

    # In[184]:

    # Print List of AUC scores
    print("AUC scores computed using 10-fold cross-validation: {}".format(cv_auc))

    # In[185]:

    # Cross-Validation for Accuracy Scores
    cv_acc = cross_val_score(model, predictors, target,
                             cv=skf, scoring='accuracy')

    # In[186]:

    # Print List of Accuracy Scores
    print("Accuracy scores computed using 10-fold cross-validation: {}".format(cv_acc))

    # In[187]:

    # Accuracy Results - Mean and Standard Deviation
    show_scores(cv_acc)

    # In[188]:

    # Accuracy Results - Mean and Standard Deviation
    show_scores(cv_acc)

    # ## Random Forest Classifier - Data Set I.

    # In[189]:

    # Instantiate Classifier - Random Forest
    rfc = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                 criterion='gini', max_depth=None, max_features='auto',
                                 max_leaf_nodes=None, max_samples=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=5, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0, n_estimators=234,
                                 n_jobs=None, oob_score=False, random_state=None,
                                 verbose=0, warm_start=False)

    # In[190]:

    # Train Classifier
    rfc.fit(X_train, y_train)

    # In[191]:

    # Prediction of Classes
    y_pred = rfc.predict(X_test)

    # In[192]:

    # Prediction of Probabilities
    y_pred_prob = rfc.predict_proba(X_test)[:, 1]

    # In[193]:

    # Select Predicted Probabilities
    y_pred_prob[1:11]

    # In[194]:

    # Train Accuracy
    rfc.score(X_train, y_train)

    # In[195]:

    # Test Accuracy
    rfc.score(X_test, y_test)

    # In[196]:

    # Output of Confusion Matrix
    print(confusion_matrix(y_test, y_pred))

    # In[197]:

    # Output of Classification Report
    print(classification_report(y_test, y_pred))

    # In[198]:

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # In[199]:

    # 10-fold Cross Validation including Stratification
    skf = StratifiedKFold(shuffle=True, n_splits=10)

    # In[200]:

    # Compute and print AUC score
    print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

    # In[201]:

    # Cross-Validation for AUC scores
    cv_auc = cross_val_score(rfc, predictors, target,
                             cv=skf, scoring='roc_auc')

    # In[202]:

    # Print List of AUC scores
    print("AUC scores computed using 10-fold cross-validation: {}".format(cv_auc))

    # In[203]:

    # Cross-Validation for Accuracy Scores
    cv_acc = cross_val_score(rfc, predictors, target,
                             cv=skf, scoring='accuracy')

    # In[204]:

    # Print List of Accuracy Scores
    print("Accuracy scores computed using 10-fold cross-validation: {}".format(cv_acc))

    # In[205]:

    # Accuracy Results - Mean and Standard Deviation
    show_scores(cv_acc)

    # # Trading Strategy - Data Set I

    # ## Trading Strategy in Combination with Logistic Regression

    # In[206]:

    target

    # In[207]:

    predictors

    # In[208]:

    for x in df_dataset1.columns:
        print(x)

    # In[209]:

    df_dataset1_copy[['Daily Return in USD']]

    # In[210]:

    # Create Lists To Store All Calculated Values For Every Variable of Every Run
    # Specified Period
    # Threshold 0.5
    spec_list_05_total = []
    spec_list_05_profit = []
    spec_list_05_loss = []
    spec_list_05_trans = []
    spec_list_05_acc = []
    spec_list_05_preds_corr = []
    spec_list_05_preds = []
    # Threshold 0.6
    spec_list_06_total = []
    spec_list_06_profit = []
    spec_list_06_loss = []
    spec_list_06_trans = []
    spec_list_06_acc = []
    spec_list_06_preds_corr = []
    spec_list_06_preds = []
    # Threshold 0.7
    spec_list_07_total = []
    spec_list_07_profit = []
    spec_list_07_loss = []
    spec_list_07_trans = []
    spec_list_07_acc = []
    spec_list_07_preds_corr = []
    spec_list_07_preds = []
    # Threshold 0.8
    spec_list_08_total = []
    spec_list_08_profit = []
    spec_list_08_loss = []
    spec_list_08_trans = []
    spec_list_08_acc = []
    spec_list_08_preds_corr = []
    spec_list_08_preds = []

    # Entire Period
    # Threshold 0.5
    list_05_total = []
    list_05_profit = []
    list_05_loss = []
    list_05_trans = []
    list_05_acc = []
    list_05_preds_corr = []
    list_05_preds = []
    # Threshold 0.6
    list_06_total = []
    list_06_profit = []
    list_06_loss = []
    list_06_trans = []
    list_06_acc = []
    list_06_preds_corr = []
    list_06_preds = []
    # Threshold 0.7
    list_07_total = []
    list_07_profit = []
    list_07_loss = []
    list_07_trans = []
    list_07_acc = []
    list_07_preds_corr = []
    list_07_preds = []
    # Threshold 0.8
    list_08_total = []
    list_08_profit = []
    list_08_loss = []
    list_08_trans = []
    list_08_acc = []
    list_08_preds_corr = []
    list_08_preds = []

    # In[211]:

    df_dataset1_copy["percentage_daily_return"] = df_dataset1_copy["Daily Return in USD"] / \
        df_dataset1_copy["Closed Price USD"]
    df_dataset1_copy["percentage_daily_return"]

    # In[212]:

    einsatz = 1000
    prestakepercentlist = []
    zinseszins = []
    profitpercent = []
    current_profitpercent = []
    spec_start_date = '2019-09-09'

    prestakepercentlist51 = []
    prestakepercentlist52 = []
    prestakepercentlist53 = []

    profitpercent51 = []
    profitpercent52 = []
    profitpercent53 = []

    current_profitpercent51 = []
    current_profitpercent52 = []
    current_profitpercent53 = []

    zinseszins51 = []
    zinseszins52 = []
    zinseszins53 = []

    current_zinseszins = []
    current_zinseszins51 = []
    current_zinseszins52 = []
    current_zinseszins53 = []

    # For Loop to Simulate K-fold Cross Validation
    for x in range(1, 100):

        # Display Number of Runs
        print("Run:", x)

        # Split Data into Training and Testing
        X_train, X_test, y_train, y_test = train_test_split(
            predictors, target, test_size=0.3, stratify=target)

        # Amount of Columns
        n_nodes = predictors.shape[1]
        lreg = RandomForestClassifier(bootstrap=False, min_samples_leaf=4,
                                      min_samples_split=5, n_estimators=200)
    #     lreg = RandomForestClassifier(bootstrap=True, min_samples_leaf=5,
    #                                   min_samples_split=2, n_estimators=234,
    #                                   max_features= 'auto', max_depth= None)

    #     lreg = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
    #                        criterion='gini', max_depth=None, max_features='auto',
    #                        max_leaf_nodes=None, max_samples=None,
    #                        min_impurity_decrease=0.0, min_impurity_split=None,
    #                        min_samples_leaf=5, min_samples_split=2,
    #                        min_weight_fraction_leaf=0.0, n_estimators=234,
    #                        n_jobs=None, oob_score=False, random_state=None,
    #                        verbose=0, warm_start=False)

        # Train Classifier
        lreg.fit(X_train, y_train)

        # Prediction of Classes
        y_pred = lreg.predict(X_test)

        # Display Confusion Matrix and Classification Report
        print(confusion_matrix(y_test, y_pred))
    #     print(classification_report(y_test, y_pred))

        # Prediction of Probabilities
        y_pred_prob = lreg.predict_proba(X_test)[:, 1]

        # Create Data Frame of Actual Values - y_test
        df_y_test = pd.DataFrame(y_test)

        # Add Predicted Probabilities to Data Frame
        df_y_test["y_pred_prob"] = y_pred_prob

        # Add Predicted Classes (0 or 1) to Data Frame to Cross-Check Predicted Probabilities
        df_y_test["y_pred"] = y_pred

        # Select Actual Daily Returns in USD from the Data Set Copy
        df_dataset1_copy_dailyreturns = df_dataset1_copy[[
            'Daily Return in USD']]

        # Merge Predictions and Daily Returns Data Frame on the Column "Date"
        df_y_test_withdailyreturns = pd.merge(
            df_y_test, df_dataset1_copy_dailyreturns, on='Date')

        df_dataset1_copy_percentage = df_dataset1_copy[[
            'percentage_daily_return']]
        df_y_test_withdailyreturns = pd.merge(
            df_y_test_withdailyreturns, df_dataset1_copy_percentage, on='Date')

        # Add Bitcoin Price in USD
        df_y_test_withdailyreturns = pd.merge(
            df_y_test_withdailyreturns, df_dataset1_copy['Closed Price USD'], on='Date')

        df_y_test_withdailyreturns['y_pred_prob_bereinigt'] = np.where(
            df_y_test_withdailyreturns['y_pred_prob'] < 0.5, 1-df_y_test_withdailyreturns['y_pred_prob'], df_y_test_withdailyreturns['y_pred_prob'])

        df_y_test_withdailyreturns["Einsatz"] = einsatz  # * \
        # df_y_test_withdailyreturns["y_pred_prob_bereinigt"]  # **2

        # Add and Calculate Transaction Costs based on the Bitcoin Price in USD
        df_y_test_withdailyreturns['Transaction Costs in USD'] = df_y_test_withdailyreturns['Einsatz']*0.00027*2

        # Create List of Multiple Thresholds
        # A Threshold of 0.7 means that only predicted probabilites with more than 0.7 or less than 0.3 are considered further
        list_thresholdcalc = [0.5, 0.51, 0.52, 0.53]

        # Try and Except Block to Address Cases in Which a Data Frame does NOT Contain Any Values Because No predicted Probability Meets the Threshold Requirements
        try:

            # For Loop Over List
            for thresholdcalc in list_thresholdcalc:

                # Creation of a Data Frame Containing Only Probability Predictions Which Meet the Threshold Requirements
                df_y_test_probaall = df_y_test.loc[(df_y_test['y_pred_prob'] > thresholdcalc) | (
                    df_y_test['y_pred_prob'] < 1-thresholdcalc)]

                # Calculation Of Accuracy Only for the Data Subset Meeting Threshold Requirements
                acc_calc = len(df_y_test_probaall[df_y_test_probaall['Class']
                                                  == df_y_test_probaall['y_pred']]) / len(df_y_test_probaall)

                print("Threshold of:", thresholdcalc)
                print(classification_report(
                    df_y_test_probaall["Class"], df_y_test_probaall["y_pred"]))

                # Creation of Another Data Frame including Returns and Transaction Costs
                # Only For Probability Predicitons that Meet the Threshold Requirements
                df_y_test_probaall_withdailyreturns = df_y_test_withdailyreturns.loc[(
                    df_y_test_withdailyreturns['y_pred_prob'] > thresholdcalc) | (df_y_test_withdailyreturns['y_pred_prob'] < 1-thresholdcalc)]

                # Trading Assumption: Bet Amount = 1 Bitcoin for Each Prediction
                # Select Daily Returns Only From True Predictions Where Predicted Class equals Actual Class.
                # It does not matter whether Daily Returns are Positive or Negative as Short-Selling Option Allows To Profit From Falling Prices
                profit = df_y_test_probaall_withdailyreturns["percentage_daily_return"][
                    df_y_test_probaall_withdailyreturns['Class'] == df_y_test_probaall_withdailyreturns['y_pred']].abs().sum()

                # Select  Daily Returns Only From False Predictions Where Predicted Class equals Actual Class
                loss = df_y_test_probaall_withdailyreturns["percentage_daily_return"][df_y_test_probaall_withdailyreturns['Class']
                                                                                      != df_y_test_probaall_withdailyreturns['y_pred']].abs().sum()

                print("prestake profit-loss {:.2f} x".format(profit-loss))

                if thresholdcalc == 0.5:
                    prestakepercentlist.append(int((profit-loss)*100))
                if thresholdcalc == 0.51:
                    prestakepercentlist51.append(int((profit-loss)*100))
                if thresholdcalc == 0.52:
                    prestakepercentlist52.append(int((profit-loss)*100))
                if thresholdcalc == 0.53:
                    prestakepercentlist53.append(int((profit-loss)*100))

                loss = loss * einsatz
                profit = profit * einsatz

                # Calculation of Transaction Costs based On BTC Value in USD. Each Future Trade considers Transaction Costs twice, becauase of Buying and Selling the Future
                transaction_costs = df_y_test_probaall_withdailyreturns['Transaction Costs in USD'].sum(
                )

                # Calculation of Total Profit Considering Profit, Loss and Transaction Costs
                total = profit-loss-transaction_costs

                # In addition, the same setting with different time horizon
                currentpreds_all = df_y_test_probaall_withdailyreturns[
                    df_y_test_probaall_withdailyreturns.index >= spec_start_date]

                # the len
                currentpreds_all_count = len(currentpreds_all)

                # Analog to Profit Calculation Above
                currentpreds_winning_total = currentpreds_all["percentage_daily_return"][(
                    currentpreds_all['Class'] == currentpreds_all['y_pred'])].abs().sum()
                # Analog to Loss Calculation Above
                currentpreds_loosing_total = currentpreds_all["percentage_daily_return"][(
                    currentpreds_all['Class'] != currentpreds_all['y_pred'])].abs().sum()
                # Analog to Calculation of Transaction Costs Above
                currentpreds_transaction_costs = currentpreds_all['Transaction Costs in USD'][(
                    currentpreds_all['Class'] != currentpreds_all['y_pred'])].abs().sum()

                currentpreds_loosing_total = currentpreds_loosing_total * einsatz
                currentpreds_winning_total = currentpreds_winning_total * einsatz

                # Analog to Calculation of Total Profit
                currentpreds_total = currentpreds_winning_total - \
                    currentpreds_loosing_total-currentpreds_transaction_costs

                print("currentpreds_winning_total {} - currentpreds_loosing_total {} -currentpreds_transaction_costs {} ".format(
                    currentpreds_winning_total, currentpreds_loosing_total, currentpreds_transaction_costs))

                # Condition
                # In Case the Shorter Time Horizon Does Not Include any Sample -> Empyt Output
                if currentpreds_all_count == 0:
                    currentpreds_all_count = '-'
                    current_acc_calc = '-'

                # Otherwise Calculation of Accuracy analog to theeins Accuracy Calculation for the Entire Period
                else:
                    current_acc_calc = (len(currentpreds_all["Daily Return in USD"][(
                        currentpreds_all['Class'] == currentpreds_all['y_pred'])]))/currentpreds_all_count

                # Display of Threshold, Total Profit, Profit, Loss, Transaction Costs, Accuracy and Correct Number of Predictions Both for the Specified Time Horizon and the Entire Time
                print("SpecPeriod:")
                print("einsatz", int(currentpreds_all["Einsatz"].sum()), "Total Return in SpecPeriod:", int(currentpreds_total), "/ Profit: ", int(currentpreds_winning_total), "/ Loss: ", int(currentpreds_loosing_total), "/ TransCosts: ", int(currentpreds_transaction_costs),
                      "/ #CorrectPreds:", (len(currentpreds_all["Daily Return in USD"][(currentpreds_all['Class'] == currentpreds_all['y_pred'])])), "of", len(currentpreds_all))

                print("CalcAccuracy {:.2f} -- profit in %/bet: {:.2f}".format(current_acc_calc, 100*(int(
                    currentpreds_total) / len(currentpreds_all)) / (currentpreds_all["Einsatz"].sum() / len(currentpreds_all))))
                print("einsatz * 1+profit% ^ # wetten {:.2f}".format(einsatz*(1+(int(currentpreds_total) / len(currentpreds_all)) / (
                    currentpreds_all["Einsatz"].sum() / len(currentpreds_all))) ** len(currentpreds_all)))

                if thresholdcalc == 0.5:
                    current_profitpercent.append(100*(int(currentpreds_total) / len(
                        currentpreds_all)) / (currentpreds_all["Einsatz"].sum() / len(currentpreds_all)))
                if thresholdcalc == 0.51:
                    current_profitpercent51.append(100*(int(currentpreds_total) / len(
                        currentpreds_all)) / (currentpreds_all["Einsatz"].sum() / len(currentpreds_all)))
                if thresholdcalc == 0.52:
                    current_profitpercent52.append(100*(int(currentpreds_total) / len(
                        currentpreds_all)) / (currentpreds_all["Einsatz"].sum() / len(currentpreds_all)))
                if thresholdcalc == 0.53:
                    current_profitpercent53.append(100*(int(currentpreds_total) / len(
                        currentpreds_all)) / (currentpreds_all["Einsatz"].sum() / len(currentpreds_all)))

                print()
                print("Entire Period:")
                print("einsatz", int(df_y_test_probaall_withdailyreturns["Einsatz"].sum()), "TotReturn:", int(total), "/ Profit: ", int(profit), "/ Loss: ", int(loss), "/ TransCosts: ", int(
                    transaction_costs), "/ CalcAccuracy", round(acc_calc, 2), "/ #CorrectPreds:", len(df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]),  "Of", len(df_y_test_probaall))
                print("einsatz/bet", df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall), "TotReturn / bet", int(total) / len(
                    df_y_test_probaall), "/ TranCosts / bet ", int(transaction_costs) / len(df_y_test_probaall), "/ CalcuAcc", round(acc_calc, 2))
                print("profit in %/bet: {:.2f}".format(100*(int(total) / len(df_y_test_probaall)) / (
                    df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))))

                if thresholdcalc == 0.5:
                    profitpercent.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.51:
                    profitpercent51.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.52:
                    profitpercent52.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.53:
                    profitpercent53.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))

                print("einsatz * 1+profit% ^ # wetten {:.2f}".format(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                    df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall)))
                if thresholdcalc == 0.5:
                    zinseszins.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                    current_zinseszins.append(einsatz*(1+(int(currentpreds_total) / len(currentpreds_all)) / (
                        currentpreds_all["Einsatz"].sum() / len(currentpreds_all))) ** len(currentpreds_all))

                print()
                print()

                # Store All Calculated Values For Every Variable of Every Run
                # Specified Period
                # Threshold = 0.5
                if thresholdcalc == 0.5:
                    spec_list_05_total.append(currentpreds_total)
                    spec_list_05_profit.append(currentpreds_winning_total)
                    spec_list_05_loss.append(currentpreds_loosing_total)
                    spec_list_05_trans.append(currentpreds_transaction_costs)
                    spec_list_05_acc.append(current_acc_calc)
                    spec_list_05_preds_corr.append((len(df_y_test_probaall_withdailyreturns["Daily Return in USD"][(
                        df_y_test_probaall_withdailyreturns.index >= spec_start_date) & (df_y_test_probaall_withdailyreturns['Class'] == df_y_test_probaall_withdailyreturns['y_pred'])])))
                    spec_list_05_preds.append(currentpreds_all_count)
                    # Entire Period:
                    list_05_total.append(total)
                    list_05_profit.append(profit)
                    list_05_loss.append(loss)
                    list_05_trans.append(transaction_costs)
                    list_05_acc.append(acc_calc)
                    list_05_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_05_preds.append(len(df_y_test_probaall))

                # Threshold = 0.6
                if thresholdcalc == 0.51:
                    spec_list_06_total.append(currentpreds_total)
                    spec_list_06_profit.append(currentpreds_winning_total)
                    spec_list_06_loss.append(currentpreds_loosing_total)
                    spec_list_06_trans.append(currentpreds_transaction_costs)
                    spec_list_06_acc.append(current_acc_calc)
                    spec_list_06_preds_corr.append((len(df_y_test_probaall_withdailyreturns["Daily Return in USD"][(
                        df_y_test_probaall_withdailyreturns.index >= spec_start_date) & (df_y_test_probaall_withdailyreturns['Class'] == df_y_test_probaall_withdailyreturns['y_pred'])])))
                    spec_list_06_preds.append(currentpreds_all_count)
                    # Entire Period:
                    list_06_total.append(total)
                    list_06_profit.append(profit)
                    list_06_loss.append(loss)
                    list_06_trans.append(transaction_costs)
                    list_06_acc.append(acc_calc)
                    list_06_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_06_preds.append(len(df_y_test_probaall))

                    zinseszins51.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                    current_zinseszins51.append(einsatz*(1+(int(currentpreds_total) / len(currentpreds_all)) / (
                        currentpreds_all["Einsatz"].sum() / len(currentpreds_all))) ** len(currentpreds_all))

                # Threshold = 0.7
                if thresholdcalc == 0.52:
                    spec_list_07_total.append(currentpreds_total)
                    spec_list_07_profit.append(currentpreds_winning_total)
                    spec_list_07_loss.append(currentpreds_loosing_total)
                    spec_list_07_trans.append(currentpreds_transaction_costs)
                    spec_list_07_acc.append(current_acc_calc)
                    spec_list_07_preds_corr.append((len(df_y_test_probaall_withdailyreturns["Daily Return in USD"][(
                        df_y_test_probaall_withdailyreturns.index >= spec_start_date) & (df_y_test_probaall_withdailyreturns['Class'] == df_y_test_probaall_withdailyreturns['y_pred'])])))
                    spec_list_07_preds.append(currentpreds_all_count)
                    # Entire Period:
                    list_07_total.append(total)
                    list_07_profit.append(profit)
                    list_07_loss.append(loss)
                    list_07_trans.append(transaction_costs)
                    list_07_acc.append(acc_calc)
                    list_07_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_07_preds.append(len(df_y_test_probaall))
                    zinseszins52.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                    current_zinseszins52.append(einsatz*(1+(int(currentpreds_total) / len(currentpreds_all)) / (
                        currentpreds_all["Einsatz"].sum() / len(currentpreds_all))) ** len(currentpreds_all))

                # Threshold = 0.8
                if thresholdcalc == 0.53:
                    spec_list_08_total.append(currentpreds_total)
                    spec_list_08_profit.append(currentpreds_winning_total)
                    spec_list_08_loss.append(currentpreds_loosing_total)
                    spec_list_08_trans.append(currentpreds_transaction_costs)
                    spec_list_08_acc.append(current_acc_calc)
                    spec_list_08_preds_corr.append((len(df_y_test_probaall_withdailyreturns["Daily Return in USD"][(
                        df_y_test_probaall_withdailyreturns.index >= spec_start_date) & (df_y_test_probaall_withdailyreturns['Class'] == df_y_test_probaall_withdailyreturns['y_pred'])])))
                    spec_list_08_preds.append(currentpreds_all_count)
                    # Entire Period:
                    list_08_total.append(total)
                    list_08_profit.append(profit)
                    list_08_loss.append(loss)
                    list_08_trans.append(transaction_costs)
                    list_08_acc.append(acc_calc)
                    list_08_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_08_preds.append(len(df_y_test_probaall))
                    zinseszins53.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                    current_zinseszins53.append(einsatz*(1+(int(currentpreds_total) / len(currentpreds_all)) / (
                        currentpreds_all["Einsatz"].sum() / len(currentpreds_all))) ** len(currentpreds_all))

        except:
            print("error")
        print()
        print()

    # In[ ]:

    # In[ ]:

    # In[213]:

    # Test Accuracy
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    accuarylist = []

    for x in range(30):
        # Split Data into Training and Testing
        X_train, X_test, y_train, y_test = train_test_split(
            predictors, target, test_size=0.3, stratify=target)

        # Train Classifier
        lreg.fit(X_train, y_train)
        y_pred = lreg.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        accuarylist.append(accuracy * 100.0)

    mean_auc_10 = np.mean(accuarylist)
    mean_auc_10

    # In[214]:

    import statistics

    result_to_csv = []

    result_to_csv.append(("statistics.median(prestakepercentlist)",
                          statistics.median(prestakepercentlist)))
    result_to_csv.append(
        ("statistics.median(profitpercent)", statistics.median(profitpercent)))
    result_to_csv.append(("statistics.median(zinseszins)",
                          statistics.median(zinseszins)))
    result_to_csv.append(("statistics.median(current_profitpercent)",
                          statistics.median(current_profitpercent)))
    result_to_csv.append(
        ("statistics.median(current_zinseszins)", statistics.median(current_zinseszins)))
    result_to_csv.append(("statistics.median(prestakepercentlist51)",
                          statistics.median(prestakepercentlist51)))
    result_to_csv.append(
        ("statistics.median(profitpercent51)", statistics.median(profitpercent51)))
    result_to_csv.append(("statistics.median(zinseszins51)",
                          statistics.median(zinseszins51)))
    result_to_csv.append(("statistics.median(current_profitpercent51)",
                          statistics.median(current_profitpercent51)))
    result_to_csv.append(("statistics.median(current_zinseszins51)",
                          statistics.median(current_zinseszins51)))
    result_to_csv.append(("statistics.median(prestakepercentlist52)",
                          statistics.median(prestakepercentlist52)))
    result_to_csv.append(
        ("statistics.median(profitpercent52)", statistics.median(profitpercent52)))
    result_to_csv.append(("statistics.median(zinseszins52)",
                          statistics.median(zinseszins52)))
    result_to_csv.append(("statistics.median(current_profitpercent52)",
                          statistics.median(current_profitpercent52)))
    result_to_csv.append(("statistics.median(current_zinseszins52)",
                          statistics.median(current_zinseszins52)))
    result_to_csv.append(("statistics.median(prestakepercentlist53)",
                          statistics.median(prestakepercentlist53)))
    result_to_csv.append(
        ("statistics.median(profitpercent53)", statistics.median(profitpercent53)))
    result_to_csv.append(("statistics.median(zinseszins53)",
                          statistics.median(zinseszins53)))
    result_to_csv.append(("statistics.median(current_profitpercent53)",
                          statistics.median(current_profitpercent53)))
    result_to_csv.append(("statistics.median(current_zinseszins53)",
                          statistics.median(current_zinseszins53)))
    df_zusammen = pd.DataFrame(result_to_csv)

    # In[215]:

    X_train.shape

    # In[216]:

    import time

    df_zusammen["algo"] = pd.Series([lreg])
    df_zusammen["spec_start_date"] = pd.Series([spec_start_date])
    df_zusammen["X_train.shape"] = pd.Series([X_train.shape])
    df_zusammen["X_test.shape"] = pd.Series([X_test.shape])
    df_zusammen["mean_auc_10"] = pd.Series([mean_auc_10])
    # adapt column for class value
    # df_dataset1=df_dataset1.drop(['Class'])
    df_zusammen["df_dataset1.columns"] = pd.Series(
        [df_dataset1.columns.tolist()])

    # In[217]:

    print(statistics.median(prestakepercentlist))
    print(statistics.median(profitpercent))
    print(statistics.median(zinseszins))
    print(statistics.median(current_profitpercent))
    print(statistics.median(current_zinseszins))
    print()
    print("proba > 51")
    print(statistics.median(prestakepercentlist51))
    print(statistics.median(profitpercent51))
    print(statistics.median(zinseszins51))
    print(statistics.median(current_profitpercent51))
    print(statistics.median(current_zinseszins51))
    print()
    print("proba > 52")
    print(statistics.median(prestakepercentlist52))
    print(statistics.median(profitpercent52))
    print(statistics.median(zinseszins52))
    print(statistics.median(current_profitpercent52))
    print(statistics.median(current_zinseszins52))
    print()
    print("proba > 53")
    print(statistics.median(prestakepercentlist53))
    print(statistics.median(profitpercent53))
    print(statistics.median(zinseszins53))
    print(statistics.median(current_profitpercent53))
    print(statistics.median(current_zinseszins53))

    # # Compare with year 2017

    # In[218]:

    # get time of 01.01.2017 until 31.12.2017 as testing df
    # df_dataset1_copy_2017 df_dataset1_copy_allbut2017

    print(len(df_dataset1_allbut2017))
    print(len(df_dataset1_2017))

    try:
        # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
        X_trainpredictors_allbut2017 = df_dataset1_allbut2017.drop(
            ['Class'], axis=1).values.astype(np.float32)
    except:
        print("Class already deleted")
        X_trainpredictors_allbut2017 = df_dataset1_allbut2017.values.astype(
            np.float32)

    try:
        # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
        X_testpredictors_2017 = df_dataset1_2017.drop(
            ['Class'], axis=1).values.astype(np.float32)
    except:
        print("Class already deleted")
        X_testpredictors_2017 = df_dataset1_2017.values.astype(np.float32)

    # Split Data into Training and Testing
    X_train = X_trainpredictors_allbut2017
    X_test = X_testpredictors_2017
    y_train = y_traintarget_allbut2017
    y_test = y_testtarget_2017

    # In[219]:

    X_trainpredictors_allbut2017

    # In[220]:

    y_traintarget_allbut2017
    y_testtarget_2017.groupby(y_testtarget_2017).size()

    # In[221]:

    X_train.shape

    # In[222]:

    y_test.shape

    # In[223]:

    einsatz = 1000
    prestakepercentlist = []
    zinseszins = []
    profitpercent = []

    prestakepercentlist51 = []
    prestakepercentlist52 = []
    prestakepercentlist53 = []

    profitpercent51 = []
    profitpercent52 = []
    profitpercent53 = []

    zinseszins51 = []
    zinseszins52 = []
    zinseszins53 = []

    # For Loop to Simulate K-fold Cross Validation
    for x in range(1, 10):

        # Display Number of Runs
        print("Run:", x)

        # Amount of Columns
        n_nodes = predictors.shape[1]

        # Instantiation of MinMaxScaler
        norm_scaler = MinMaxScaler()

        # Train Classifier
        lreg.fit(X_train, y_train)

        # Prediction of Classes
        y_pred = lreg.predict(X_test)

        # Display Confusion Matrix and Classification Report
        print(confusion_matrix(y_test, y_pred))

        # Prediction of Probabilities
        y_pred_prob = lreg.predict_proba(X_test)[:, 1]

        # Create Data Frame of Actual Values - y_test
        df_y_test = pd.DataFrame(y_test)

        # Add Predicted Probabilities to Data Frame
        df_y_test["y_pred_prob"] = y_pred_prob

        # Add Predicted Classes (0 or 1) to Data Frame to Cross-Check Predicted Probabilities
        df_y_test["y_pred"] = y_pred

        # Select Actual Daily Returns in USD from the Data Set Copy
        df_dataset1_copy_dailyreturns = df_dataset1_copy[[
            'Daily Return in USD']]

        # Merge Predictions and Daily Returns Data Frame on the Column "Date"
        df_y_test_withdailyreturns = pd.merge(
            df_y_test, df_dataset1_copy_dailyreturns, on='Date')

        df_dataset1_copy_percentage = df_dataset1_copy[[
            'percentage_daily_return_bef_shift']]
        df_y_test_withdailyreturns = pd.merge(
            df_y_test_withdailyreturns, df_dataset1_copy_percentage, on='Date')

        # Add Bitcoin Price in USD
        df_y_test_withdailyreturns = pd.merge(
            df_y_test_withdailyreturns, df_dataset1_copy['Closed Price USD'], on='Date')

        df_y_test_withdailyreturns['y_pred_prob_bereinigt'] = np.where(
            df_y_test_withdailyreturns['y_pred_prob'] < 0.5, 1-df_y_test_withdailyreturns['y_pred_prob'], df_y_test_withdailyreturns['y_pred_prob'])

        df_y_test_withdailyreturns["Einsatz"] = einsatz  # * \
        # df_y_test_withdailyreturns["y_pred_prob_bereinigt"]  # **2

        # Add and Calculate Transaction Costs based on the Bitcoin Price in USD
        df_y_test_withdailyreturns['Transaction Costs in USD'] = df_y_test_withdailyreturns['Einsatz']*0.00027*2

        # Create List of Multiple Thresholds
        # A Threshold of 0.7 means that only predicted probabilites with more than 0.7 or less than 0.3 are considered further
        list_thresholdcalc = [0.5, 0.51, 0.52, 0.53]

        try:

            # For Loop Over List
            for thresholdcalc in list_thresholdcalc:

                # Creation of a Data Frame Containing Only Probability Predictions Which Meet the Threshold Requirements
                df_y_test_probaall = df_y_test.loc[(df_y_test['y_pred_prob'] > thresholdcalc) | (
                    df_y_test['y_pred_prob'] < 1-thresholdcalc)]

                # Calculation Of Accuracy Only for the Data Subset Meeting Threshold Requirements
                acc_calc = len(df_y_test_probaall[df_y_test_probaall['Class']
                                                  == df_y_test_probaall['y_pred']]) / len(df_y_test_probaall)

                print("Threshold of:", thresholdcalc)
                print(classification_report(
                    df_y_test_probaall["Class"], df_y_test_probaall["y_pred"]))

                # Creation of Another Data Frame including Returns and Transaction Costs
                # Only For Probability Predicitons that Meet the Threshold Requirements
                df_y_test_probaall_withdailyreturns = df_y_test_withdailyreturns.loc[(
                    df_y_test_withdailyreturns['y_pred_prob'] > thresholdcalc) | (df_y_test_withdailyreturns['y_pred_prob'] < 1-thresholdcalc)]

                # Trading Assumption: Bet Amount = 1 Bitcoin for Each Prediction
                # Select Daily Returns Only From True Predictions Where Predicted Class equals Actual Class.
                # It does not matter whether Daily Returns are Positive or Negative as Short-Selling Option Allows To Profit From Falling Prices
                profit = df_y_test_probaall_withdailyreturns["percentage_daily_return_bef_shift"][
                    df_y_test_probaall_withdailyreturns['Class'] == df_y_test_probaall_withdailyreturns['y_pred']].abs().sum()

                # Select  Daily Returns Only From False Predictions Where Predicted Class equals Actual Class
                loss = df_y_test_probaall_withdailyreturns["percentage_daily_return_bef_shift"][df_y_test_probaall_withdailyreturns['Class']
                                                                                                != df_y_test_probaall_withdailyreturns['y_pred']].abs().sum()

                print("prestake profit-loss {:.2f} x".format(profit-loss))

                if thresholdcalc == 0.5:
                    prestakepercentlist.append(int((profit-loss)*100))
                if thresholdcalc == 0.51:
                    prestakepercentlist51.append(int((profit-loss)*100))
                if thresholdcalc == 0.52:
                    prestakepercentlist52.append(int((profit-loss)*100))
                if thresholdcalc == 0.53:
                    prestakepercentlist53.append(int((profit-loss)*100))

                loss = loss * einsatz
                profit = profit * einsatz

                # Calculation of Transaction Costs based On BTC Value in USD. Each Future Trade considers Transaction Costs twice, becauase of Buying and Selling the Future
                transaction_costs = df_y_test_probaall_withdailyreturns['Transaction Costs in USD'].sum(
                )

                # Calculation of Total Profit Considering Profit, Loss and Transaction Costs
                total = profit-loss-transaction_costs

                print()
                print("Entire Period:")
                print("einsatz", int(df_y_test_probaall_withdailyreturns["Einsatz"].sum()), "TotReturn:", int(total), "/ Profit: ", int(profit), "/ Loss: ", int(loss), "/ TransCosts: ", int(
                    transaction_costs), "/ CalcAccuracy", round(acc_calc, 2), "/ #CorrectPreds:", len(df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]),  "Of", len(df_y_test_probaall))
                print("einsatz/bet", df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall), "TotReturn / bet", int(total) / len(
                    df_y_test_probaall), "/ TranCosts / bet ", int(transaction_costs) / len(df_y_test_probaall), "/ CalcuAcc", round(acc_calc, 2))
                print("profit in %/bet: {:.2f}".format(100*(int(total) / len(df_y_test_probaall)) / (
                    df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))))

                if thresholdcalc == 0.5:
                    profitpercent.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.51:
                    profitpercent51.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.52:
                    profitpercent52.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.53:
                    profitpercent53.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))

                print("einsatz * 1+profit% ^ # wetten {:.2f}".format(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                    df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall)))
                if thresholdcalc == 0.5:
                    zinseszins.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                print()
                print()

                # Store All Calculated Values For Every Variable of Every Run
                # Specified Period
                # Threshold = 0.5
                if thresholdcalc == 0.5:
                    # Entire Period:
                    list_05_total.append(total)
                    list_05_profit.append(profit)
                    list_05_loss.append(loss)
                    list_05_trans.append(transaction_costs)
                    list_05_acc.append(acc_calc)
                    list_05_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_05_preds.append(len(df_y_test_probaall))

                # Threshold = 0.6
                if thresholdcalc == 0.51:
                    # Entire Period:
                    list_06_total.append(total)
                    list_06_profit.append(profit)
                    list_06_loss.append(loss)
                    list_06_trans.append(transaction_costs)
                    list_06_acc.append(acc_calc)
                    list_06_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_06_preds.append(len(df_y_test_probaall))

                    zinseszins51.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                # Threshold = 0.7
                if thresholdcalc == 0.52:
                    # Entire Period:
                    list_07_total.append(total)
                    list_07_profit.append(profit)
                    list_07_loss.append(loss)
                    list_07_trans.append(transaction_costs)
                    list_07_acc.append(acc_calc)
                    list_07_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_07_preds.append(len(df_y_test_probaall))
                    zinseszins52.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                # Threshold = 0.8
                if thresholdcalc == 0.53:
                    # Entire Period:
                    list_08_total.append(total)
                    list_08_profit.append(profit)
                    list_08_loss.append(loss)
                    list_08_trans.append(transaction_costs)
                    list_08_acc.append(acc_calc)
                    list_08_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_08_preds.append(len(df_y_test_probaall))
                    zinseszins53.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

        except:
            print("problem")

        print()
        print()

    # In[224]:

    print(statistics.median(prestakepercentlist))
    print(statistics.median(profitpercent))
    print(statistics.median(zinseszins))
    print()
    print("proba > 51")
    print(statistics.median(prestakepercentlist51))
    print(statistics.median(profitpercent51))
    print(statistics.median(zinseszins51))
    print()
    print("proba > 52")
    print(statistics.median(prestakepercentlist52))
    print(statistics.median(profitpercent52))
    print(statistics.median(zinseszins52))
    print()
    print("proba > 53")
    print(statistics.median(prestakepercentlist53))
    print(statistics.median(profitpercent53))
    print(statistics.median(zinseszins53))

    # In[225]:

    import statistics

    results_of_2017 = []

    results_of_2017.append(
        ("2017_statistics.median(prestakepercentlist)", statistics.median(prestakepercentlist)))
    results_of_2017.append(
        ("2017_statistics.median(profitpercent)", statistics.median(profitpercent)))
    results_of_2017.append(
        ("2017_statistics.median(zinseszins)", statistics.median(zinseszins)))
    results_of_2017.append(("", ""))
    results_of_2017.append(("", ""))
    results_of_2017.append(
        ("2017_statistics.median(prestakepercentlist51)", statistics.median(prestakepercentlist51)))
    results_of_2017.append(
        ("2017_statistics.median(profitpercent51)", statistics.median(profitpercent51)))
    results_of_2017.append(
        ("2017_statistics.median(zinseszins51)", statistics.median(zinseszins51)))
    results_of_2017.append(("", ""))
    results_of_2017.append(("", ""))
    results_of_2017.append(
        ("2017_statistics.median(prestakepercentlist52)", statistics.median(prestakepercentlist52)))
    results_of_2017.append(
        ("2017_statistics.median(profitpercent52)", statistics.median(profitpercent52)))
    results_of_2017.append(
        ("2017_statistics.median(zinseszins52)", statistics.median(zinseszins52)))
    results_of_2017.append(("", ""))
    results_of_2017.append(("", ""))
    results_of_2017.append(
        ("2017_statistics.median(prestakepercentlist53)", statistics.median(prestakepercentlist53)))
    results_of_2017.append(
        ("2017_statistics.median(profitpercent53)", statistics.median(profitpercent53)))
    results_of_2017.append(
        ("2017_statistics.median(zinseszins53)", statistics.median(zinseszins53)))
    results_of_2017.append(("", ""))
    results_of_2017.append(("", ""))

    df_zusammen["year to predict_2017"] = "2017"
    df_zusammen["keys_2017"] = pd.Series([i[0] for i in results_of_2017])
    df_zusammen["values_2017"] = pd.Series([i[1] for i in results_of_2017])
    df_zusammen["X_train.shape_2017"] = pd.Series([X_train.shape])
    df_zusammen["X_test.shape_2017"] = pd.Series([X_test.shape])

    # # Compare with year 2018

    # In[226]:

    # get time of 01.01.2018 until 31.12.2018 as testing df
    # df_dataset1_copy_2018 df_dataset1_copy_allbut2018

    print(len(df_dataset1_allbut2018))
    print(len(df_dataset1_2018))

    try:
        # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
        X_trainpredictors_allbut2018 = df_dataset1_allbut2018.drop(
            ['Class'], axis=1).values.astype(np.float32)
    except:
        print("Class already deleted")
        X_trainpredictors_allbut2018 = df_dataset1_allbut2018.values.astype(
            np.float32)

    try:
        # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
        X_testpredictors_2018 = df_dataset1_2018.drop(
            ['Class'], axis=1).values.astype(np.float32)
    except:
        print("Class already deleted")
        X_testpredictors_2018 = df_dataset1_2018.values.astype(np.float32)

    # Split Data into Training and Testing
    X_train = X_trainpredictors_allbut2018
    X_test = X_testpredictors_2018
    y_train = y_traintarget_allbut2018
    y_test = y_testtarget_2018

    # In[227]:

    y_traintarget_allbut2018
    y_testtarget_2018.groupby(y_testtarget_2018).size()

    # In[228]:

    X_train.shape

    # In[229]:

    y_test.shape

    # In[230]:

    einsatz = 1000
    prestakepercentlist = []
    zinseszins = []
    profitpercent = []

    prestakepercentlist51 = []
    prestakepercentlist52 = []
    prestakepercentlist53 = []

    profitpercent51 = []
    profitpercent52 = []
    profitpercent53 = []

    zinseszins51 = []
    zinseszins52 = []
    zinseszins53 = []

    # For Loop to Simulate K-fold Cross Validation
    for x in range(1, 10):

        # Display Number of Runs
        print("Run:", x)

        # Amount of Columns
        n_nodes = predictors.shape[1]

        # Instantiation of MinMaxScaler
        norm_scaler = MinMaxScaler()

        # Train Classifier
        lreg.fit(X_train, y_train)

        # Prediction of Classes
        y_pred = lreg.predict(X_test)

        # Display Confusion Matrix and Classification Report
        print(confusion_matrix(y_test, y_pred))

        # Prediction of Probabilities
        y_pred_prob = lreg.predict_proba(X_test)[:, 1]

        # Create Data Frame of Actual Values - y_test
        df_y_test = pd.DataFrame(y_test)

        # Add Predicted Probabilities to Data Frame
        df_y_test["y_pred_prob"] = y_pred_prob

        # Add Predicted Classes (0 or 1) to Data Frame to Cross-Check Predicted Probabilities
        df_y_test["y_pred"] = y_pred

        # Select Actual Daily Returns in USD from the Data Set Copy
        df_dataset1_copy_dailyreturns = df_dataset1_copy[[
            'Daily Return in USD']]

        # Merge Predictions and Daily Returns Data Frame on the Column "Date"
        df_y_test_withdailyreturns = pd.merge(
            df_y_test, df_dataset1_copy_dailyreturns, on='Date')

        df_dataset1_copy_percentage = df_dataset1_copy[[
            'percentage_daily_return_bef_shift']]
        df_y_test_withdailyreturns = pd.merge(
            df_y_test_withdailyreturns, df_dataset1_copy_percentage, on='Date')

        # Add Bitcoin Price in USD
        df_y_test_withdailyreturns = pd.merge(
            df_y_test_withdailyreturns, df_dataset1_copy['Closed Price USD'], on='Date')

        df_y_test_withdailyreturns['y_pred_prob_bereinigt'] = np.where(
            df_y_test_withdailyreturns['y_pred_prob'] < 0.5, 1-df_y_test_withdailyreturns['y_pred_prob'], df_y_test_withdailyreturns['y_pred_prob'])

        df_y_test_withdailyreturns["Einsatz"] = einsatz  # * \
        # df_y_test_withdailyreturns["y_pred_prob_bereinigt"]  # **2

        # Add and Calculate Transaction Costs based on the Bitcoin Price in USD
        df_y_test_withdailyreturns['Transaction Costs in USD'] = df_y_test_withdailyreturns['Einsatz']*0.00027*2

        # Create List of Multiple Thresholds
        # A Threshold of 0.7 means that only predicted probabilites with more than 0.7 or less than 0.3 are considered further
        list_thresholdcalc = [0.5, 0.51, 0.52, 0.53]

        try:

            # For Loop Over List
            for thresholdcalc in list_thresholdcalc:

                # Creation of a Data Frame Containing Only Probability Predictions Which Meet the Threshold Requirements
                df_y_test_probaall = df_y_test.loc[(df_y_test['y_pred_prob'] > thresholdcalc) | (
                    df_y_test['y_pred_prob'] < 1-thresholdcalc)]

                # Calculation Of Accuracy Only for the Data Subset Meeting Threshold Requirements
                acc_calc = len(df_y_test_probaall[df_y_test_probaall['Class']
                                                  == df_y_test_probaall['y_pred']]) / len(df_y_test_probaall)

                print("Threshold of:", thresholdcalc)
                print(classification_report(
                    df_y_test_probaall["Class"], df_y_test_probaall["y_pred"]))

                # Creation of Another Data Frame including Returns and Transaction Costs
                # Only For Probability Predicitons that Meet the Threshold Requirements
                df_y_test_probaall_withdailyreturns = df_y_test_withdailyreturns.loc[(
                    df_y_test_withdailyreturns['y_pred_prob'] > thresholdcalc) | (df_y_test_withdailyreturns['y_pred_prob'] < 1-thresholdcalc)]

                # Trading Assumption: Bet Amount = 1 Bitcoin for Each Prediction
                # Select Daily Returns Only From True Predictions Where Predicted Class equals Actual Class.
                # It does not matter whether Daily Returns are Positive or Negative as Short-Selling Option Allows To Profit From Falling Prices
                profit = df_y_test_probaall_withdailyreturns["percentage_daily_return_bef_shift"][
                    df_y_test_probaall_withdailyreturns['Class'] == df_y_test_probaall_withdailyreturns['y_pred']].abs().sum()

                # Select  Daily Returns Only From False Predictions Where Predicted Class equals Actual Class
                loss = df_y_test_probaall_withdailyreturns["percentage_daily_return_bef_shift"][df_y_test_probaall_withdailyreturns['Class']
                                                                                                != df_y_test_probaall_withdailyreturns['y_pred']].abs().sum()

                print("prestake profit-loss {:.2f} x".format(profit-loss))

                if thresholdcalc == 0.5:
                    prestakepercentlist.append(int((profit-loss)*100))
                if thresholdcalc == 0.51:
                    prestakepercentlist51.append(int((profit-loss)*100))
                if thresholdcalc == 0.52:
                    prestakepercentlist52.append(int((profit-loss)*100))
                if thresholdcalc == 0.53:
                    prestakepercentlist53.append(int((profit-loss)*100))

                loss = loss * einsatz
                profit = profit * einsatz

                # Calculation of Transaction Costs based On BTC Value in USD. Each Future Trade considers Transaction Costs twice, becauase of Buying and Selling the Future
                transaction_costs = df_y_test_probaall_withdailyreturns['Transaction Costs in USD'].sum(
                )

                # Calculation of Total Profit Considering Profit, Loss and Transaction Costs
                total = profit-loss-transaction_costs

                print()
                print("Entire Period:")
                print("einsatz", int(df_y_test_probaall_withdailyreturns["Einsatz"].sum()), "TotReturn:", int(total), "/ Profit: ", int(profit), "/ Loss: ", int(loss), "/ TransCosts: ", int(
                    transaction_costs), "/ CalcAccuracy", round(acc_calc, 2), "/ #CorrectPreds:", len(df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]),  "Of", len(df_y_test_probaall))
                print("einsatz/bet", df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall), "TotReturn / bet", int(total) / len(
                    df_y_test_probaall), "/ TranCosts / bet ", int(transaction_costs) / len(df_y_test_probaall), "/ CalcuAcc", round(acc_calc, 2))
                print("profit in %/bet: {:.2f}".format(100*(int(total) / len(df_y_test_probaall)) / (
                    df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))))

                if thresholdcalc == 0.5:
                    profitpercent.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.51:
                    profitpercent51.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.52:
                    profitpercent52.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.53:
                    profitpercent53.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))

                print("einsatz * 1+profit% ^ # wetten {:.2f}".format(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                    df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall)))
                if thresholdcalc == 0.5:
                    zinseszins.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                print()
                print()

                # Store All Calculated Values For Every Variable of Every Run
                # Specified Period
                # Threshold = 0.5
                if thresholdcalc == 0.5:
                    # Entire Period:
                    list_05_total.append(total)
                    list_05_profit.append(profit)
                    list_05_loss.append(loss)
                    list_05_trans.append(transaction_costs)
                    list_05_acc.append(acc_calc)
                    list_05_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_05_preds.append(len(df_y_test_probaall))

                # Threshold = 0.6
                if thresholdcalc == 0.51:
                    # Entire Period:
                    list_06_total.append(total)
                    list_06_profit.append(profit)
                    list_06_loss.append(loss)
                    list_06_trans.append(transaction_costs)
                    list_06_acc.append(acc_calc)
                    list_06_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_06_preds.append(len(df_y_test_probaall))

                    zinseszins51.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                # Threshold = 0.7
                if thresholdcalc == 0.52:
                    # Entire Period:
                    list_07_total.append(total)
                    list_07_profit.append(profit)
                    list_07_loss.append(loss)
                    list_07_trans.append(transaction_costs)
                    list_07_acc.append(acc_calc)
                    list_07_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_07_preds.append(len(df_y_test_probaall))
                    zinseszins52.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                # Threshold = 0.8
                if thresholdcalc == 0.53:
                    # Entire Period:
                    list_08_total.append(total)
                    list_08_profit.append(profit)
                    list_08_loss.append(loss)
                    list_08_trans.append(transaction_costs)
                    list_08_acc.append(acc_calc)
                    list_08_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_08_preds.append(len(df_y_test_probaall))
                    zinseszins53.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

        except:
            print("problem")

        print()
        print()

    # In[231]:

    print(statistics.median(prestakepercentlist))
    print(statistics.median(profitpercent))
    print(statistics.median(zinseszins))
    print()
    print("proba > 51")
    print(statistics.median(prestakepercentlist51))
    print(statistics.median(profitpercent51))
    print(statistics.median(zinseszins51))
    print()
    print("proba > 52")
    print(statistics.median(prestakepercentlist52))
    print(statistics.median(profitpercent52))
    print(statistics.median(zinseszins52))
    print()
    print("proba > 53")
    print(statistics.median(prestakepercentlist53))
    print(statistics.median(profitpercent53))
    print(statistics.median(zinseszins53))

    # In[232]:

    import statistics

    results_of_2018 = []

    results_of_2018.append(
        ("2018_statistics.median(prestakepercentlist)", statistics.median(prestakepercentlist)))
    results_of_2018.append(
        ("2018_statistics.median(profitpercent)", statistics.median(profitpercent)))
    results_of_2018.append(
        ("2018_statistics.median(zinseszins)", statistics.median(zinseszins)))
    results_of_2018.append(("", ""))
    results_of_2018.append(("", ""))
    results_of_2018.append(
        ("2018_statistics.median(prestakepercentlist51)", statistics.median(prestakepercentlist51)))
    results_of_2018.append(
        ("2018_statistics.median(profitpercent51)", statistics.median(profitpercent51)))
    results_of_2018.append(
        ("2018_statistics.median(zinseszins51)", statistics.median(zinseszins51)))
    results_of_2018.append(("", ""))
    results_of_2018.append(("", ""))
    results_of_2018.append(
        ("2018_statistics.median(prestakepercentlist52)", statistics.median(prestakepercentlist52)))
    results_of_2018.append(
        ("2018_statistics.median(profitpercent52)", statistics.median(profitpercent52)))
    results_of_2018.append(
        ("2018_statistics.median(zinseszins52)", statistics.median(zinseszins52)))
    results_of_2018.append(("", ""))
    results_of_2018.append(("", ""))
    results_of_2018.append(
        ("2018_statistics.median(prestakepercentlist53)", statistics.median(prestakepercentlist53)))
    results_of_2018.append(
        ("2018_statistics.median(profitpercent53)", statistics.median(profitpercent53)))
    results_of_2018.append(
        ("2018_statistics.median(zinseszins53)", statistics.median(zinseszins53)))
    results_of_2018.append(("", ""))
    results_of_2018.append(("", ""))

    df_zusammen["year to predict_2018"] = "2018"
    df_zusammen["keys_2018"] = pd.Series([i[0] for i in results_of_2018])
    df_zusammen["values_2018"] = pd.Series([i[1] for i in results_of_2018])
    df_zusammen["X_train.shape_2018"] = pd.Series([X_train.shape])
    df_zusammen["X_test.shape_2018"] = pd.Series([X_test.shape])

    # In[ ]:

    # # Compare with year 2019

    # In[233]:

    # get time of 01.01.2019 until 31.12.2019 as testing df
    # df_dataset1_copy_2019 df_dataset1_copy_allbut2019

    print(len(df_dataset1_allbut2019))
    print(len(df_dataset1_2019))
    # Setup of Predictors (Indpendent Variables) by Excluding Target Variable

    try:
        # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
        X_trainpredictors_allbut2019 = df_dataset1_allbut2019.drop(
            ['Class'], axis=1).values.astype(np.float32)
    except:
        print("Class already deleted")
        X_trainpredictors_allbut2019 = df_dataset1_allbut2019.values.astype(
            np.float32)

    try:
        # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
        X_testpredictors_2019 = df_dataset1_2019.drop(
            ['Class'], axis=1).values.astype(np.float32)
    except:
        print("Class already deleted")
        X_testpredictors_2019 = df_dataset1_2019.values.astype(np.float32)

    # Split Data into Training and Testing
    X_train = X_trainpredictors_allbut2019
    X_test = X_testpredictors_2019
    y_train = y_traintarget_allbut2019
    y_test = y_testtarget_2019
    X_train_scaled = X_train
    X_test_scaled = X_test

    # In[234]:

    X_train.shape

    # In[235]:

    y_test.shape

    # In[236]:

    einsatz = 1000
    prestakepercentlist = []
    zinseszins = []
    profitpercent = []

    prestakepercentlist51 = []
    prestakepercentlist52 = []
    prestakepercentlist53 = []

    profitpercent51 = []
    profitpercent52 = []
    profitpercent53 = []

    zinseszins51 = []
    zinseszins52 = []
    zinseszins53 = []

    # For Loop to Simulate K-fold Cross Validation
    for x in range(1, 10):

        # Display Number of Runs
        print("Run:", x)

        # Amount of Columns
        n_nodes = predictors.shape[1]

        # Instantiation of MinMaxScaler
        norm_scaler = MinMaxScaler()

        # Train Classifier
        lreg.fit(X_train_scaled, y_train)

        # Prediction of Classes
        y_pred = lreg.predict(X_test_scaled)

        # Display Confusion Matrix and Classification Report
        print(confusion_matrix(y_test, y_pred))

        # Prediction of Probabilities
        y_pred_prob = lreg.predict_proba(X_test_scaled)[:, 1]

        # Create Data Frame of Actual Values - y_test
        df_y_test = pd.DataFrame(y_test)

        # Add Predicted Probabilities to Data Frame
        df_y_test["y_pred_prob"] = y_pred_prob

        # Add Predicted Classes (0 or 1) to Data Frame to Cross-Check Predicted Probabilities
        df_y_test["y_pred"] = y_pred

        # Select Actual Daily Returns in USD from the Data Set Copy
        df_dataset1_copy_dailyreturns = df_dataset1_copy[[
            'Daily Return in USD']]

        # Merge Predictions and Daily Returns Data Frame on the Column "Date"
        df_y_test_withdailyreturns = pd.merge(
            df_y_test, df_dataset1_copy_dailyreturns, on='Date')

        df_dataset1_copy_percentage = df_dataset1_copy[[
            'percentage_daily_return_bef_shift']]
        df_y_test_withdailyreturns = pd.merge(
            df_y_test_withdailyreturns, df_dataset1_copy_percentage, on='Date')

        # Add Bitcoin Price in USD
        df_y_test_withdailyreturns = pd.merge(
            df_y_test_withdailyreturns, df_dataset1_copy['Closed Price USD'], on='Date')

        df_y_test_withdailyreturns['y_pred_prob_bereinigt'] = np.where(
            df_y_test_withdailyreturns['y_pred_prob'] < 0.5, 1-df_y_test_withdailyreturns['y_pred_prob'], df_y_test_withdailyreturns['y_pred_prob'])

        df_y_test_withdailyreturns["Einsatz"] = einsatz  # * \
        # df_y_test_withdailyreturns["y_pred_prob_bereinigt"]  # **2

        # Add and Calculate Transaction Costs based on the Bitcoin Price in USD
        df_y_test_withdailyreturns['Transaction Costs in USD'] = df_y_test_withdailyreturns['Einsatz']*0.00027*2

        # Create List of Multiple Thresholds
        # A Threshold of 0.7 means that only predicted probabilites with more than 0.7 or less than 0.3 are considered further
        list_thresholdcalc = [0.5, 0.51, 0.52, 0.53]

        try:

            # For Loop Over List
            for thresholdcalc in list_thresholdcalc:

                # Creation of a Data Frame Containing Only Probability Predictions Which Meet the Threshold Requirements
                df_y_test_probaall = df_y_test.loc[(df_y_test['y_pred_prob'] > thresholdcalc) | (
                    df_y_test['y_pred_prob'] < 1-thresholdcalc)]

                # Calculation Of Accuracy Only for the Data Subset Meeting Threshold Requirements
                acc_calc = len(df_y_test_probaall[df_y_test_probaall['Class']
                                                  == df_y_test_probaall['y_pred']]) / len(df_y_test_probaall)

                print("Threshold of:", thresholdcalc)
                print(classification_report(
                    df_y_test_probaall["Class"], df_y_test_probaall["y_pred"]))

                # Creation of Another Data Frame including Returns and Transaction Costs
                # Only For Probability Predicitons that Meet the Threshold Requirements
                df_y_test_probaall_withdailyreturns = df_y_test_withdailyreturns.loc[(
                    df_y_test_withdailyreturns['y_pred_prob'] > thresholdcalc) | (df_y_test_withdailyreturns['y_pred_prob'] < 1-thresholdcalc)]

                # Trading Assumption: Bet Amount = 1 Bitcoin for Each Prediction
                # Select Daily Returns Only From True Predictions Where Predicted Class equals Actual Class.
                # It does not matter whether Daily Returns are Positive or Negative as Short-Selling Option Allows To Profit From Falling Prices
                profit = df_y_test_probaall_withdailyreturns["percentage_daily_return_bef_shift"][
                    df_y_test_probaall_withdailyreturns['Class'] == df_y_test_probaall_withdailyreturns['y_pred']].abs().sum()

                # Select  Daily Returns Only From False Predictions Where Predicted Class equals Actual Class
                loss = df_y_test_probaall_withdailyreturns["percentage_daily_return_bef_shift"][df_y_test_probaall_withdailyreturns['Class']
                                                                                                != df_y_test_probaall_withdailyreturns['y_pred']].abs().sum()

                print("prestake profit-loss {:.2f} x".format(profit-loss))

                if thresholdcalc == 0.5:
                    prestakepercentlist.append(int((profit-loss)*100))
                if thresholdcalc == 0.51:
                    prestakepercentlist51.append(int((profit-loss)*100))
                if thresholdcalc == 0.52:
                    prestakepercentlist52.append(int((profit-loss)*100))
                if thresholdcalc == 0.53:
                    prestakepercentlist53.append(int((profit-loss)*100))

                loss = loss * einsatz
                profit = profit * einsatz

                # Calculation of Transaction Costs based On BTC Value in USD. Each Future Trade considers Transaction Costs twice, becauase of Buying and Selling the Future
                transaction_costs = df_y_test_probaall_withdailyreturns['Transaction Costs in USD'].sum(
                )

                # Calculation of Total Profit Considering Profit, Loss and Transaction Costs
                total = profit-loss-transaction_costs

                print()
                print("Entire Period:")
                print("einsatz", int(df_y_test_probaall_withdailyreturns["Einsatz"].sum()), "TotReturn:", int(total), "/ Profit: ", int(profit), "/ Loss: ", int(loss), "/ TransCosts: ", int(
                    transaction_costs), "/ CalcAccuracy", round(acc_calc, 2), "/ #CorrectPreds:", len(df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]),  "Of", len(df_y_test_probaall))
                print("einsatz/bet", df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall), "TotReturn / bet", int(total) / len(
                    df_y_test_probaall), "/ TranCosts / bet ", int(transaction_costs) / len(df_y_test_probaall), "/ CalcuAcc", round(acc_calc, 2))
                print("profit in %/bet: {:.2f}".format(100*(int(total) / len(df_y_test_probaall)) / (
                    df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))))

                if thresholdcalc == 0.5:
                    profitpercent.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.51:
                    profitpercent51.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.52:
                    profitpercent52.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == 0.53:
                    profitpercent53.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))

                print("einsatz * 1+profit% ^ # wetten {:.2f}".format(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                    df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall)))
                if thresholdcalc == 0.5:
                    zinseszins.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                print()
                print()

                # Store All Calculated Values For Every Variable of Every Run
                # Specified Period
                # Threshold = 0.5
                if thresholdcalc == 0.5:
                    # Entire Period:
                    list_05_total.append(total)
                    list_05_profit.append(profit)
                    list_05_loss.append(loss)
                    list_05_trans.append(transaction_costs)
                    list_05_acc.append(acc_calc)
                    list_05_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_05_preds.append(len(df_y_test_probaall))

                # Threshold = 0.6
                if thresholdcalc == 0.51:
                    # Entire Period:
                    list_06_total.append(total)
                    list_06_profit.append(profit)
                    list_06_loss.append(loss)
                    list_06_trans.append(transaction_costs)
                    list_06_acc.append(acc_calc)
                    list_06_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_06_preds.append(len(df_y_test_probaall))

                    zinseszins51.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                # Threshold = 0.7
                if thresholdcalc == 0.52:
                    # Entire Period:
                    list_07_total.append(total)
                    list_07_profit.append(profit)
                    list_07_loss.append(loss)
                    list_07_trans.append(transaction_costs)
                    list_07_acc.append(acc_calc)
                    list_07_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_07_preds.append(len(df_y_test_probaall))
                    zinseszins52.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                # Threshold = 0.8
                if thresholdcalc == 0.53:
                    # Entire Period:
                    list_08_total.append(total)
                    list_08_profit.append(profit)
                    list_08_loss.append(loss)
                    list_08_trans.append(transaction_costs)
                    list_08_acc.append(acc_calc)
                    list_08_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_08_preds.append(len(df_y_test_probaall))
                    zinseszins53.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

        except:
            print("problem")

        print()
        print()

    # In[237]:

    print(statistics.median(prestakepercentlist))
    print(statistics.median(profitpercent))
    print(statistics.median(zinseszins))
    print()
    print("proba > 51")
    print(statistics.median(prestakepercentlist51))
    print(statistics.median(profitpercent51))
    print(statistics.median(zinseszins51))
    print()
    print("proba > 52")
    print(statistics.median(prestakepercentlist52))
    print(statistics.median(profitpercent52))
    print(statistics.median(zinseszins52))
    print()
    print("proba > 53")
    print(statistics.median(prestakepercentlist53))
    print(statistics.median(profitpercent53))
    print(statistics.median(zinseszins53))

    # In[238]:

    import statistics

    results_of_2019 = []

    results_of_2019.append(
        ("2019_statistics.median(prestakepercentlist)", statistics.median(prestakepercentlist)))
    results_of_2019.append(
        ("2019_statistics.median(profitpercent)", statistics.median(profitpercent)))
    results_of_2019.append(
        ("2019_statistics.median(zinseszins)", statistics.median(zinseszins)))
    results_of_2019.append(("", ""))
    results_of_2019.append(("", ""))
    results_of_2019.append(
        ("2019_statistics.median(prestakepercentlist51)", statistics.median(prestakepercentlist51)))
    results_of_2019.append(
        ("2019_statistics.median(profitpercent51)", statistics.median(profitpercent51)))
    results_of_2019.append(
        ("2019_statistics.median(zinseszins51)", statistics.median(zinseszins51)))
    results_of_2019.append(("", ""))
    results_of_2019.append(("", ""))
    results_of_2019.append(
        ("2019_statistics.median(prestakepercentlist52)", statistics.median(prestakepercentlist52)))
    results_of_2019.append(
        ("2019_statistics.median(profitpercent52)", statistics.median(profitpercent52)))
    results_of_2019.append(
        ("2019_statistics.median(zinseszins52)", statistics.median(zinseszins52)))
    results_of_2019.append(("", ""))
    results_of_2019.append(("", ""))
    results_of_2019.append(
        ("2019_statistics.median(prestakepercentlist53)", statistics.median(prestakepercentlist53)))
    results_of_2019.append(
        ("2019_statistics.median(profitpercent53)", statistics.median(profitpercent53)))
    results_of_2019.append(
        ("2019_statistics.median(zinseszins53)", statistics.median(zinseszins53)))
    results_of_2019.append(("", ""))
    results_of_2019.append(("", ""))

    df_zusammen["year to predict_2019"] = "2019"
    df_zusammen["keys_2019"] = pd.Series([i[0] for i in results_of_2019])
    df_zusammen["values_2019"] = pd.Series([i[1] for i in results_of_2019])
    df_zusammen["X_train.shape_2019"] = pd.Series([X_train.shape])
    df_zusammen["X_test.shape_2019"] = pd.Series([X_test.shape])

    # In[ ]:

    # # Compare with year 20

    # In[239]:

    # get time of 01.01.2020 until 31.12.2020 as testing df
    # df_dataset1_copy_2020 df_dataset1_copy_allbut2020
    print(len(df_dataset1_allbut2020))
    print(len(df_dataset1_2020))

    try:
        # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
        X_trainpredictors_allbut2020 = df_dataset1_allbut2020.drop(
            ['Class'], axis=1).values.astype(np.float32)
    except:
        print("Class already deleted")
        X_trainpredictors_allbut2020 = df_dataset1_allbut2020.values.astype(
            np.float32)

    try:
        # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
        X_testpredictors_2020 = df_dataset1_2020.drop(
            ['Class'], axis=1).values.astype(np.float32)
    except:
        print("Class already deleted")
        X_testpredictors_2020 = df_dataset1_2020.values.astype(np.float32)

    # Split Data into Training and Testing
    X_train = X_trainpredictors_allbut2020
    X_test = X_testpredictors_2020
    y_train = y_traintarget_allbut2020
    y_test = y_testtarget_2020
    X_train_scaled = X_train
    X_test_scaled = X_test

    # In[240]:

    X_train.shape

    # In[241]:

    y_test.shape

    # In[242]:

    df_rights_wrongs_all = []

    einsatz = 1000
    prestakepercentlist = []
    zinseszins = []
    profitpercent = []

    prestakepercentlist51 = []
    prestakepercentlist52 = []
    prestakepercentlist53 = []

    profitpercent51 = []
    profitpercent52 = []
    profitpercent53 = []

    zinseszins51 = []
    zinseszins52 = []
    zinseszins53 = []

    # For Loop to Simulate K-fold Cross Validation
    for x in range(1, 10):

        # Display Number of Runs
        print("Run:", x)

        # Amount of Columns
        n_nodes = predictors.shape[1]

        # Instantiation of MinMaxScaler
        norm_scaler = MinMaxScaler()

        # Train Classifier
        lreg.fit(X_train_scaled, y_train)

        # Prediction of Classes
        y_pred = lreg.predict(X_test_scaled)

        # Display Confusion Matrix and Classification Report
        print(confusion_matrix(y_test, y_pred))

        # Prediction of Probabilities
        y_pred_prob = lreg.predict_proba(X_test_scaled)[:, 1]

        # Create Data Frame of Actual Values - y_test
        df_y_test = pd.DataFrame(y_test)

        # Add Predicted Probabilities to Data Frame
        df_y_test["y_pred_prob"] = y_pred_prob

        # Add Predicted Classes (0 or 1) to Data Frame to Cross-Check Predicted Probabilities
        df_y_test["y_pred"] = y_pred

        # Select Actual Daily Returns in USD from the Data Set Copy
        df_dataset1_copy_dailyreturns = df_dataset1_copy[[
            'Daily Return in USD']]

        # Merge Predictions and Daily Returns Data Frame on the Column "Date"
        df_y_test_withdailyreturns = pd.merge(
            df_y_test, df_dataset1_copy_dailyreturns, on='Date')

        df_dataset1_copy_percentage = df_dataset1_copy[[
            'percentage_daily_return_bef_shift']]
        df_y_test_withdailyreturns = pd.merge(
            df_y_test_withdailyreturns, df_dataset1_copy_percentage, on='Date')

        # Add Bitcoin Price in USD
        df_y_test_withdailyreturns = pd.merge(
            df_y_test_withdailyreturns, df_dataset1_copy['Closed Price USD'], on='Date')

        df_y_test_withdailyreturns['y_pred_prob_bereinigt'] = np.where(
            df_y_test_withdailyreturns['y_pred_prob'] < 0.5, 1-df_y_test_withdailyreturns['y_pred_prob'], df_y_test_withdailyreturns['y_pred_prob'])

        df_y_test_withdailyreturns["Einsatz"] = einsatz  # * \
        # df_y_test_withdailyreturns["y_pred_prob_bereinigt"]  # **2

        # Add and Calculate Transaction Costs based on the Bitcoin Price in USD
        df_y_test_withdailyreturns['Transaction Costs in USD'] = df_y_test_withdailyreturns['Einsatz']*0.00027*2

        # Create List of Multiple Thresholds
        # A Threshold of 0.7 means that only predicted probabilites with more than 0.7 or less than 0.3 are considered further
        list_thresholdcalc = [0.5, 0.51, 0.55, 0.6]

        df_rights_wrongs_all.append(df_y_test_withdailyreturns)

        try:

            # For Loop Over List
            for thresholdcalc in list_thresholdcalc:

                # Creation of a Data Frame Containing Only Probability Predictions Which Meet the Threshold Requirements
                df_y_test_probaall = df_y_test.loc[(df_y_test['y_pred_prob'] > thresholdcalc) | (
                    df_y_test['y_pred_prob'] < 1-thresholdcalc)]

                # Calculation Of Accuracy Only for the Data Subset Meeting Threshold Requirements
                acc_calc = len(df_y_test_probaall[df_y_test_probaall['Class']
                                                  == df_y_test_probaall['y_pred']]) / len(df_y_test_probaall)

                print("Threshold of:", thresholdcalc)
                print(classification_report(
                    df_y_test_probaall["Class"], df_y_test_probaall["y_pred"]))

                # Creation of Another Data Frame including Returns and Transaction Costs
                # Only For Probability Predicitons that Meet the Threshold Requirements
                df_y_test_probaall_withdailyreturns = df_y_test_withdailyreturns.loc[(
                    df_y_test_withdailyreturns['y_pred_prob'] > thresholdcalc) | (df_y_test_withdailyreturns['y_pred_prob'] < 1-thresholdcalc)]

                # Trading Assumption: Bet Amount = 1 Bitcoin for Each Prediction
                # Select Daily Returns Only From True Predictions Where Predicted Class equals Actual Class.
                # It does not matter whether Daily Returns are Positive or Negative as Short-Selling Option Allows To Profit From Falling Prices
                profit = df_y_test_probaall_withdailyreturns["percentage_daily_return_bef_shift"][
                    df_y_test_probaall_withdailyreturns['Class'] == df_y_test_probaall_withdailyreturns['y_pred']].abs().sum()

                # Select  Daily Returns Only From False Predictions Where Predicted Class equals Actual Class
                loss = df_y_test_probaall_withdailyreturns["percentage_daily_return_bef_shift"][df_y_test_probaall_withdailyreturns['Class']
                                                                                                != df_y_test_probaall_withdailyreturns['y_pred']].abs().sum()

                print("prestake profit-loss {:.2f} x".format(profit-loss))

                if thresholdcalc == list_thresholdcalc[0]:
                    prestakepercentlist.append(int((profit-loss)*100))
                if thresholdcalc == list_thresholdcalc[1]:
                    prestakepercentlist51.append(int((profit-loss)*100))
                if thresholdcalc == list_thresholdcalc[2]:
                    prestakepercentlist52.append(int((profit-loss)*100))
                if thresholdcalc == list_thresholdcalc[3]:
                    prestakepercentlist53.append(int((profit-loss)*100))

                loss = loss * einsatz
                profit = profit * einsatz

                # Calculation of Transaction Costs based On BTC Value in USD. Each Future Trade considers Transaction Costs twice, becauase of Buying and Selling the Future
                transaction_costs = df_y_test_probaall_withdailyreturns['Transaction Costs in USD'].sum(
                )

                # Calculation of Total Profit Considering Profit, Loss and Transaction Costs
                total = profit-loss-transaction_costs

                print()
                print("Entire Period:")
                print("einsatz", int(df_y_test_probaall_withdailyreturns["Einsatz"].sum()), "TotReturn:", int(total), "/ Profit: ", int(profit), "/ Loss: ", int(loss), "/ TransCosts: ", int(
                    transaction_costs), "/ CalcAccuracy", round(acc_calc, 2), "/ #CorrectPreds:", len(df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]),  "Of", len(df_y_test_probaall))
                print("einsatz/bet", df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall), "TotReturn / bet", int(total) / len(
                    df_y_test_probaall), "/ TranCosts / bet ", int(transaction_costs) / len(df_y_test_probaall), "/ CalcuAcc", round(acc_calc, 2))
                print("profit in %/bet: {:.2f}".format(100*(int(total) / len(df_y_test_probaall)) / (
                    df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))))

                if thresholdcalc == list_thresholdcalc[0]:
                    profitpercent.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == list_thresholdcalc[1]:
                    profitpercent51.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == list_thresholdcalc[2]:
                    profitpercent52.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))
                if thresholdcalc == list_thresholdcalc[3]:
                    profitpercent53.append(100*(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall)))

                print("einsatz * 1+profit% ^ # wetten {:.2f}".format(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                    df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall)))
                if thresholdcalc == list_thresholdcalc[0]:
                    zinseszins.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                print()
                print()

                # Store All Calculated Values For Every Variable of Every Run
                # Specified Period
                # Threshold = 0.5
                if thresholdcalc == list_thresholdcalc[0]:
                    # Entire Period:
                    list_05_total.append(total)
                    list_05_profit.append(profit)
                    list_05_loss.append(loss)
                    list_05_trans.append(transaction_costs)
                    list_05_acc.append(acc_calc)
                    list_05_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_05_preds.append(len(df_y_test_probaall))

                # Threshold = 0.6
                if thresholdcalc == list_thresholdcalc[1]:
                    # Entire Period:
                    list_06_total.append(total)
                    list_06_profit.append(profit)
                    list_06_loss.append(loss)
                    list_06_trans.append(transaction_costs)
                    list_06_acc.append(acc_calc)
                    list_06_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_06_preds.append(len(df_y_test_probaall))

                    zinseszins51.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                # Threshold = 0.7
                if thresholdcalc == list_thresholdcalc[2]:
                    # Entire Period:
                    list_07_total.append(total)
                    list_07_profit.append(profit)
                    list_07_loss.append(loss)
                    list_07_trans.append(transaction_costs)
                    list_07_acc.append(acc_calc)
                    list_07_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_07_preds.append(len(df_y_test_probaall))
                    zinseszins52.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

                # Threshold = 0.8
                if thresholdcalc == list_thresholdcalc[3]:
                    # Entire Period:
                    list_08_total.append(total)
                    list_08_profit.append(profit)
                    list_08_loss.append(loss)
                    list_08_trans.append(transaction_costs)
                    list_08_acc.append(acc_calc)
                    list_08_preds_corr.append(len(
                        df_y_test_probaall[df_y_test_probaall['Class'] == df_y_test_probaall['y_pred']]))
                    list_08_preds.append(len(df_y_test_probaall))
                    zinseszins53.append(einsatz*(1+(int(total) / len(df_y_test_probaall)) / (
                        df_y_test_probaall_withdailyreturns["Einsatz"].sum() / len(df_y_test_probaall))) ** len(df_y_test_probaall))

        except:
            print("problem")

        print()
        print()

    # In[243]:

    print(statistics.median(prestakepercentlist))
    print(statistics.median(profitpercent))
    print(statistics.median(zinseszins))
    print()
    print("proba > 51")
    print(statistics.median(prestakepercentlist51))
    print(statistics.median(profitpercent51))
    print(statistics.median(zinseszins51))
    print()
    print("proba > 52")
    print(statistics.median(prestakepercentlist52))
    print(statistics.median(profitpercent52))
    print(statistics.median(zinseszins52))
    print()
    print("proba > 53")
    print(statistics.median(prestakepercentlist53))
    print(statistics.median(profitpercent53))
    print(statistics.median(zinseszins53))

    # In[244]:

    list_08_preds_corr

    # In[245]:

    import statistics

    results_of_2020 = []

    results_of_2020.append(
        ("2020_statistics.median(prestakepercentlist)", statistics.median(prestakepercentlist)))
    results_of_2020.append(
        ("2020_statistics.median(profitpercent)", statistics.median(profitpercent)))
    results_of_2020.append(
        ("2020_statistics.median(zinseszins)", statistics.median(zinseszins)))
    results_of_2020.append(("", ""))
    results_of_2020.append(("", ""))
    results_of_2020.append(
        ("2020_statistics.median(prestakepercentlist51)", statistics.median(prestakepercentlist51)))
    results_of_2020.append(
        ("2020_statistics.median(profitpercent51)", statistics.median(profitpercent51)))
    results_of_2020.append(
        ("2020_statistics.median(zinseszins51)", statistics.median(zinseszins51)))
    results_of_2020.append(("", ""))
    results_of_2020.append(("", ""))
    results_of_2020.append(
        ("2020_statistics.median(prestakepercentlist52)", statistics.median(prestakepercentlist52)))
    results_of_2020.append(
        ("2020_statistics.median(profitpercent52)", statistics.median(profitpercent52)))
    results_of_2020.append(
        ("2020_statistics.median(zinseszins52)", statistics.median(zinseszins52)))
    results_of_2020.append(("", ""))
    results_of_2020.append(("", ""))
    results_of_2020.append(
        ("2020_statistics.median(prestakepercentlist53)", statistics.median(prestakepercentlist53)))
    results_of_2020.append(
        ("2020_statistics.median(profitpercent53)", statistics.median(profitpercent53)))
    results_of_2020.append(
        ("2020_statistics.median(zinseszins53)", statistics.median(zinseszins53)))
    results_of_2020.append(("", ""))
    results_of_2020.append(("", ""))

    df_zusammen["year to predict 2020"] = "2020"
    df_zusammen["keys2020"] = pd.Series([i[0] for i in results_of_2020])
    df_zusammen["values2020"] = pd.Series([i[1] for i in results_of_2020])
    df_zusammen["X_train.shape2020"] = pd.Series([X_train.shape])
    df_zusammen["X_test.shape2020"] = pd.Series([X_test.shape])

    # # Save Results as csv

    # In[246]:

    out_path = "{}_{}_ppd{}_f{}.xlsx".format(int(time.time()), str(lreg)[:10], str(
        round(statistics.median(current_profitpercent), 2)), anzahlfeats)

    writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
    df_zusammen.to_excel(writer, sheet_name='Tabelle1')
    writer.save()
