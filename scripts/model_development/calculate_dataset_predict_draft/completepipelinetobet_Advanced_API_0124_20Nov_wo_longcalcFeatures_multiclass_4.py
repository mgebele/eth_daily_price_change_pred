#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data-Preprocessing" data-toc-modified-id="Data-Preprocessing-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Data-Import-via-Application-Programming-Interfaces" data-toc-modified-id="Data-Import-via-Application-Programming-Interfaces-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Data Import via Application Programming Interfaces</a></span><ul class="toc-item"><li><span><a href="#Blockchain-Data-from-Glassnode" data-toc-modified-id="Blockchain-Data-from-Glassnode-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Blockchain Data from Glassnode</a></span><ul class="toc-item"><li><span><a href="#Market-Metrics-from-Glassnode" data-toc-modified-id="Market-Metrics-from-Glassnode-1.1.1.1"><span class="toc-item-num">1.1.1.1&nbsp;&nbsp;</span>Market Metrics from Glassnode</a></span></li><li><span><a href="#Mining-Metrics-from-Glassnode" data-toc-modified-id="Mining-Metrics-from-Glassnode-1.1.1.2"><span class="toc-item-num">1.1.1.2&nbsp;&nbsp;</span>Mining Metrics from Glassnode</a></span></li><li><span><a href="#Blocks-Metrics-from-Glassnode" data-toc-modified-id="Blocks-Metrics-from-Glassnode-1.1.1.3"><span class="toc-item-num">1.1.1.3&nbsp;&nbsp;</span>Blocks Metrics from Glassnode</a></span></li><li><span><a href="#Distribution-Metrics-from-Glassnode" data-toc-modified-id="Distribution-Metrics-from-Glassnode-1.1.1.4"><span class="toc-item-num">1.1.1.4&nbsp;&nbsp;</span>Distribution Metrics from Glassnode</a></span></li><li><span><a href="#Fee-Metrics-from-Glassnode" data-toc-modified-id="Fee-Metrics-from-Glassnode-1.1.1.5"><span class="toc-item-num">1.1.1.5&nbsp;&nbsp;</span>Fee Metrics from Glassnode</a></span></li><li><span><a href="#UTXO-Metrics-from-Glassnode" data-toc-modified-id="UTXO-Metrics-from-Glassnode-1.1.1.6"><span class="toc-item-num">1.1.1.6&nbsp;&nbsp;</span>UTXO Metrics from Glassnode</a></span></li><li><span><a href="#Supply-Metrics-from-Glassnode" data-toc-modified-id="Supply-Metrics-from-Glassnode-1.1.1.7"><span class="toc-item-num">1.1.1.7&nbsp;&nbsp;</span>Supply Metrics from Glassnode</a></span></li><li><span><a href="#Transaction-Metrics-from-Glassnode" data-toc-modified-id="Transaction-Metrics-from-Glassnode-1.1.1.8"><span class="toc-item-num">1.1.1.8&nbsp;&nbsp;</span>Transaction Metrics from Glassnode</a></span></li><li><span><a href="#Exchange-Metrics-from-Glassnode---1-Month-Lag" data-toc-modified-id="Exchange-Metrics-from-Glassnode---1-Month-Lag-1.1.1.9"><span class="toc-item-num">1.1.1.9&nbsp;&nbsp;</span>Exchange Metrics from Glassnode - 1 Month Lag</a></span></li><li><span><a href="#Indicator-I-Metrics-from-Glassnode" data-toc-modified-id="Indicator-I-Metrics-from-Glassnode-1.1.1.10"><span class="toc-item-num">1.1.1.10&nbsp;&nbsp;</span>Indicator-I Metrics from Glassnode</a></span></li><li><span><a href="#Indicator-2-Metrics-from-Glassnode" data-toc-modified-id="Indicator-2-Metrics-from-Glassnode-1.1.1.11"><span class="toc-item-num">1.1.1.11&nbsp;&nbsp;</span>Indicator-2 Metrics from Glassnode</a></span></li><li><span><a href="#Indicator-3-Metrics-from-Glassnode" data-toc-modified-id="Indicator-3-Metrics-from-Glassnode-1.1.1.12"><span class="toc-item-num">1.1.1.12&nbsp;&nbsp;</span>Indicator-3 Metrics from Glassnode</a></span></li><li><span><a href="#Addresses-Metrics-from-Glassnode" data-toc-modified-id="Addresses-Metrics-from-Glassnode-1.1.1.13"><span class="toc-item-num">1.1.1.13&nbsp;&nbsp;</span>Addresses Metrics from Glassnode</a></span></li><li><span><a href="#Futures-Metrics-from-Glassnode---Excluded-as-Data-Available-Only-Since-Feb-2020" data-toc-modified-id="Futures-Metrics-from-Glassnode---Excluded-as-Data-Available-Only-Since-Feb-2020-1.1.1.14"><span class="toc-item-num">1.1.1.14&nbsp;&nbsp;</span>Futures Metrics from Glassnode - Excluded as Data Available Only Since Feb 2020</a></span><ul class="toc-item"><li><span><a href="#List-of-Futures-Features" data-toc-modified-id="List-of-Futures-Features-1.1.1.14.1"><span class="toc-item-num">1.1.1.14.1&nbsp;&nbsp;</span>List of Futures Features</a></span></li><li><span><a href="#Data-Frame-of-Future-features" data-toc-modified-id="Data-Frame-of-Future-features-1.1.1.14.2"><span class="toc-item-num">1.1.1.14.2&nbsp;&nbsp;</span>Data Frame of Future features</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#File-Import-and-Data-Tidying" data-toc-modified-id="File-Import-and-Data-Tidying-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>File Import and Data Tidying</a></span><ul class="toc-item"><li><span><a href="#Import-Blockchain-CSV-Files" data-toc-modified-id="Import-Blockchain-CSV-Files-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Import Blockchain CSV-Files</a></span><ul class="toc-item"><li><span><a href="#Market-Metric-CSV-Files" data-toc-modified-id="Market-Metric-CSV-Files-1.2.1.1"><span class="toc-item-num">1.2.1.1&nbsp;&nbsp;</span>Market Metric CSV-Files</a></span></li><li><span><a href="#Mining-Metric-CSV-Files" data-toc-modified-id="Mining-Metric-CSV-Files-1.2.1.2"><span class="toc-item-num">1.2.1.2&nbsp;&nbsp;</span>Mining Metric CSV-Files</a></span></li><li><span><a href="#Blocks-Metric-CSV-Files" data-toc-modified-id="Blocks-Metric-CSV-Files-1.2.1.3"><span class="toc-item-num">1.2.1.3&nbsp;&nbsp;</span>Blocks Metric CSV-Files</a></span></li><li><span><a href="#Distribution-Metric-CSV-Files" data-toc-modified-id="Distribution-Metric-CSV-Files-1.2.1.4"><span class="toc-item-num">1.2.1.4&nbsp;&nbsp;</span>Distribution Metric CSV-Files</a></span></li><li><span><a href="#Fee-Metric-CSV-Files" data-toc-modified-id="Fee-Metric-CSV-Files-1.2.1.5"><span class="toc-item-num">1.2.1.5&nbsp;&nbsp;</span>Fee Metric CSV-Files</a></span></li><li><span><a href="#UTXO-Metric-CSV-Files" data-toc-modified-id="UTXO-Metric-CSV-Files-1.2.1.6"><span class="toc-item-num">1.2.1.6&nbsp;&nbsp;</span>UTXO Metric CSV-Files</a></span></li><li><span><a href="#Supply-Metric-CSV-Files" data-toc-modified-id="Supply-Metric-CSV-Files-1.2.1.7"><span class="toc-item-num">1.2.1.7&nbsp;&nbsp;</span>Supply Metric CSV-Files</a></span></li><li><span><a href="#Transaction-Metric-CSV-Files" data-toc-modified-id="Transaction-Metric-CSV-Files-1.2.1.8"><span class="toc-item-num">1.2.1.8&nbsp;&nbsp;</span>Transaction Metric CSV-Files</a></span></li><li><span><a href="#Exchange-Metrics-CSV-Files" data-toc-modified-id="Exchange-Metrics-CSV-Files-1.2.1.9"><span class="toc-item-num">1.2.1.9&nbsp;&nbsp;</span>Exchange Metrics CSV-Files</a></span></li><li><span><a href="#Indicator-1-Metric-CSV-Files" data-toc-modified-id="Indicator-1-Metric-CSV-Files-1.2.1.10"><span class="toc-item-num">1.2.1.10&nbsp;&nbsp;</span>Indicator-1 Metric CSV-Files</a></span></li><li><span><a href="#Indicator-2-Metrics-CSV-Files" data-toc-modified-id="Indicator-2-Metrics-CSV-Files-1.2.1.11"><span class="toc-item-num">1.2.1.11&nbsp;&nbsp;</span>Indicator-2 Metrics CSV-Files</a></span></li><li><span><a href="#Indicator-3-Metrics-CSV-Files" data-toc-modified-id="Indicator-3-Metrics-CSV-Files-1.2.1.12"><span class="toc-item-num">1.2.1.12&nbsp;&nbsp;</span>Indicator-3 Metrics CSV-Files</a></span></li><li><span><a href="#Addresses-Metric-CSV-Files" data-toc-modified-id="Addresses-Metric-CSV-Files-1.2.1.13"><span class="toc-item-num">1.2.1.13&nbsp;&nbsp;</span>Addresses Metric CSV-Files</a></span></li><li><span><a href="#Futures-Metrics-CSV-Files---Excluded-as-Data-Available-Only-Since-Feb-2020" data-toc-modified-id="Futures-Metrics-CSV-Files---Excluded-as-Data-Available-Only-Since-Feb-2020-1.2.1.14"><span class="toc-item-num">1.2.1.14&nbsp;&nbsp;</span>Futures Metrics CSV-Files - Excluded as Data Available Only Since Feb 2020</a></span><ul class="toc-item"><li><span><a href="#List-of-Future-Features" data-toc-modified-id="List-of-Future-Features-1.2.1.14.1"><span class="toc-item-num">1.2.1.14.1&nbsp;&nbsp;</span>List of Future Features</a></span></li><li><span><a href="#Counter-to-Differentiate-between-Data-Frame-Setup-and-Data-Frame-Extension" data-toc-modified-id="Counter-to-Differentiate-between-Data-Frame-Setup-and-Data-Frame-Extension-1.2.1.14.2"><span class="toc-item-num">1.2.1.14.2&nbsp;&nbsp;</span>Counter to Differentiate between Data Frame Setup and Data Frame Extension</a></span></li><li><span><a href="#For-Loop-through-all-Features-in-the-List" data-toc-modified-id="For-Loop-through-all-Features-in-the-List-1.2.1.14.3"><span class="toc-item-num">1.2.1.14.3&nbsp;&nbsp;</span>For Loop through all Features in the List</a></span></li><li><span><a href="#Store-Data-Frame-Copy-in-Market-Data-Frame" data-toc-modified-id="Store-Data-Frame-Copy-in-Market-Data-Frame-1.2.1.14.4"><span class="toc-item-num">1.2.1.14.4&nbsp;&nbsp;</span>Store Data Frame Copy in Market Data Frame</a></span></li><li><span><a href="#Delete-Original-Data-Frame" data-toc-modified-id="Delete-Original-Data-Frame-1.2.1.14.5"><span class="toc-item-num">1.2.1.14.5&nbsp;&nbsp;</span>Delete Original Data Frame</a></span></li><li><span><a href="#Change-Data-Type-of-Columns-to-Float" data-toc-modified-id="Change-Data-Type-of-Columns-to-Float-1.2.1.14.6"><span class="toc-item-num">1.2.1.14.6&nbsp;&nbsp;</span>Change Data Type of Columns to Float</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#Target-Variables" data-toc-modified-id="Target-Variables-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Target Variables</a></span></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#Volatility" data-toc-modified-id="Volatility-1.4.1"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>Volatility</a></span></li><li><span><a href="#Log-Price" data-toc-modified-id="Log-Price-1.4.2"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Log Price</a></span></li></ul></li></ul></li><li><span><a href="#Data-Sets" data-toc-modified-id="Data-Sets-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Sets</a></span><ul class="toc-item"><li><span><a href="#Data-Set-I.---Blockchain-Metrics-Only" data-toc-modified-id="Data-Set-I.---Blockchain-Metrics-Only-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Data Set I. - Blockchain Metrics Only</a></span></li></ul></li><li><span><a href="#Classification-Models---Data-Set-Variations" data-toc-modified-id="Classification-Models---Data-Set-Variations-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Classification Models - Data Set Variations</a></span></li><li><span><a href="#Import-Data-Sets" data-toc-modified-id="Import-Data-Sets-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Import Data Sets</a></span><ul class="toc-item"><li><span><a href="#Data-Set-I." data-toc-modified-id="Data-Set-I.-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Data Set I.</a></span></li><li><span><a href="#Prediction-Shift" data-toc-modified-id="Prediction-Shift-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Prediction Shift</a></span><ul class="toc-item"><li><span><a href="#Data-Set-I." data-toc-modified-id="Data-Set-I.-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>Data Set I.</a></span></li></ul></li></ul></li><li><span><a href="#Classification-Setup" data-toc-modified-id="Classification-Setup-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Classification Setup</a></span><ul class="toc-item"><li><span><a href="#Create-Data-Set-Copies-for-Trading-Strategy" data-toc-modified-id="Create-Data-Set-Copies-for-Trading-Strategy-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Create Data Set Copies for Trading Strategy</a></span></li><li><span><a href="#Categorization---Data-Set-I." data-toc-modified-id="Categorization---Data-Set-I.-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Categorization - Data Set I.</a></span></li></ul></li><li><span><a href="#Model-Preparation---Data-Set-I." data-toc-modified-id="Model-Preparation---Data-Set-I.-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Model Preparation - Data Set I.</a></span><ul class="toc-item"><li><span><a href="#Train-Test-Split---Data-Set-I." data-toc-modified-id="Train-Test-Split---Data-Set-I.-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Train Test Split - Data Set I.</a></span><ul class="toc-item"><li><span><a href="#Check-of-Class-Balance" data-toc-modified-id="Check-of-Class-Balance-6.1.1"><span class="toc-item-num">6.1.1&nbsp;&nbsp;</span>Check of Class Balance</a></span></li></ul></li></ul></li><li><span><a href="#Visualize-pred-vs-reality" data-toc-modified-id="Visualize-pred-vs-reality-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Visualize pred vs reality</a></span><ul class="toc-item"><li><span><a href="#Feature-Scaling---Data-Set-I." data-toc-modified-id="Feature-Scaling---Data-Set-I.-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Feature Scaling - Data Set I.</a></span></li></ul></li><li><span><a href="#Classification-Models---Data-Set-I." data-toc-modified-id="Classification-Models---Data-Set-I.-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Classification Models - Data Set I.</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression---Data-Set-I." data-toc-modified-id="Logistic-Regression---Data-Set-I.-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Logistic Regression - Data Set I.</a></span></li><li><span><a href="#Random-Forest-Classifier---Data-Set-I." data-toc-modified-id="Random-Forest-Classifier---Data-Set-I.-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Random Forest Classifier - Data Set I.</a></span></li></ul></li><li><span><a href="#Trading-Strategy---Data-Set-I" data-toc-modified-id="Trading-Strategy---Data-Set-I-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Trading Strategy - Data Set I</a></span><ul class="toc-item"><li><span><a href="#Trading-Strategy-in-Combination-with-Logistic-Regression" data-toc-modified-id="Trading-Strategy-in-Combination-with-Logistic-Regression-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Trading Strategy in Combination with Logistic Regression</a></span></li></ul></li><li><span><a href="#Deepdive-in-prediction" data-toc-modified-id="Deepdive-in-prediction-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Deepdive in prediction</a></span></li><li><span><a href="#Bet-on-tomorrow" data-toc-modified-id="Bet-on-tomorrow-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Bet on tomorrow</a></span></li><li><span><a href="#get-x-train-data-of-today-to-predict-tomorrow-y" data-toc-modified-id="get-x-train-data-of-today-to-predict-tomorrow-y-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>get x train data of today to predict tomorrow y</a></span></li><li><span><a href="#Binance-Trading-Bot-execution!" data-toc-modified-id="Binance-Trading-Bot-execution!-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Binance Trading Bot execution!</a></span></li><li><span><a href="#Store-DF-of-the-one-prediction-in-database" data-toc-modified-id="Store-DF-of-the-one-prediction-in-database-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Store DF of the one prediction in database</a></span></li></ul></div>

# # Data Preprocessing

# ## Data Import via Application Programming Interfaces

# In[1]:


# Import of Modules and Packages
from sqlalchemy import create_engine
import plotly.express as px
import statistics
import plotly.graph_objects as go
import pycaret
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras
from pycaret.classification import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
import seaborn as sns
import os
import requests
import json
import datetime
import yfinance as yf
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytrends.request import TrendReq
from pytrends import dailydata


# In[2]:


# Adjustment of Decimal Places
pd.options.display.float_format = '{:.2f}'.format


# In[3]:


# Glassnode API Key
API_Key = ''


# In[4]:


#os.chdir('\\github_repo\\JupyterLabDir\\Rest\\MA BTC\\Markus_Code_MA_final\\20201010\\Glassnode')


# In[5]:


os.getcwd()


# In[6]:


#os.chdir('C:\\Users\\markus.gebele\\github_repo\\JupyterLabDir\\Rest\\MA BTC\\Markus_Code_MA_final\\20201010\\Glassnode')


# ### Blockchain Data from Glassnode

# In[7]:


# Import Function for a Single Feature Variable from Glassnode
def import_glassnode(url, feature_name):
    data = requests.get(url).json()
    df = pd.json_normalize(data)

    # Rename columns
    df.columns = ['Date', feature_name]

    # Convert Object to Datetime Object
    df.Date = pd.to_datetime(df.Date, unit='s')
    df.Date = df.Date.map(lambda x: x.strftime('%Y-%m-%d'))

    # Set Date Column as Index
    df.set_index('Date', inplace=True)

    # Drop Missing Values
    df = df.dropna()

    # Change Data Type to Float
    df = df.astype(float)

    # Creation of CSV-File, Part of Feature Name as CSV File Name
    df.to_csv("{}.csv".format(feature_name.rsplit('/')[-1]), decimal=',')

    # Plot of DataFrame
    #df.plot(figsize=(4.5, 1.75))
    # plt.xticks(rotation=45)
    return df


# In[8]:


# Import Function for Several Feature Variables from Glassnode
def import_glassnode_many(feature_list):
    # Counter to Differentiate between Data Frame Setup and Data Frame Join
    counter = 0

    # For Loop through all Features in the List
    for feature in feature_list:

        # URL
        url = 'https://api.glassnode.com/v1/metrics/{}?a=btc&f=json&api_key={}'.format(
            feature, API_Key)

        # Execute Import Function for a Single Feature Variable
        df = import_glassnode(url, feature)

        # Display Data Frame Tail
        print(df.tail())

        # Condition - Data Frame Set up or Data Frame Join
        if counter > 0:

            # Data Frame Join
            df_new = df_new.join(df, how='inner')

        else:

            # Data Frame Setup
            df_new = df

        # Increase Counter
        counter += 1

    return df_new


# #### Market Metrics from Glassnode

# In[9]:


# List of Market Features
market_feature_list = ['market/price_usd_close',
                       'market/price_drawdown_relative', 'market/price_realized_usd', 'market/mvrv']


# In[10]:


try:
    # Data Frame of Market Features
    df_market = import_glassnode_many(market_feature_list)
except:
    time.sleep(60)
    df_market = import_glassnode_many(market_feature_list)


# #### Mining Metrics from Glassnode

# In[11]:


# List of Mining Features    #'mining/revenue_sum',
mining_feature_list = ['mining/difficulty_latest',
                       'mining/hash_rate_mean', 'mining/marketcap_thermocap_ratio']


# In[12]:


# Data Frame of Mining Features
df_mining = import_glassnode_many(mining_feature_list)


# #### Blocks Metrics from Glassnode

# In[13]:


# List of Blockchain Features
blockchain_feature_list = ['blockchain/block_count',
                           'blockchain/block_interval_mean', 'blockchain/block_size_sum']


# In[14]:


# Data Frame of Blockchain Features
df_blockchainfeatures = import_glassnode_many(blockchain_feature_list)


# #### Distribution Metrics from Glassnode

# In[15]:


# List of Distribution Features
#distribution_feature_list = ['distribution/balance_1pct_holders', 'distribution/gini', 'distribution/herfindahl'] #


# In[16]:


# Data Frame of Distribution Features
#df_distribution = import_glassnode_many(distribution_feature_list)


# #### Fee Metrics from Glassnode

# In[17]:


# List of Fee Features
fees_feature_list = ['fees/volume_sum',
                     'fees/volume_mean', 'fees/fee_ratio_multiple']


# In[18]:


# Data Frame of Fee Features
df_fees = import_glassnode_many(fees_feature_list)


# #### UTXO Metrics from Glassnode

# In[19]:


# List of UTXO Features
utxo_feature_list = ['blockchain/utxo_created_count', 'blockchain/utxo_spent_count', 'blockchain/utxo_created_value_sum',
                     'blockchain/utxo_spent_value_sum', 'blockchain/utxo_profit_relative', 'blockchain/utxo_profit_count', 'blockchain/utxo_loss_count']


# In[20]:


# Data Frame of UTXO Features
df_utxo = import_glassnode_many(utxo_feature_list)


# In[ ]:


# #### Supply Metrics from Glassnode

# In[21]:


# List of Supply Features
supply_feature_list = ['supply/current', 'supply/profit_relative', 'supply/profit_sum', 'supply/loss_sum', 'supply/active_more_1y_percent', 'supply/active_more_2y_percent', 'supply/active_more_3y_percent', 'supply/active_more_5y_percent', 'supply/active_24h', 'supply/active_1d_1w',
                       'supply/active_1w_1m', 'supply/active_1m_3m', 'supply/active_3m_6m', 'supply/active_6m_12m', 'supply/active_1y_2y', 'supply/active_2y_3y', 'supply/active_3y_5y', 'supply/active_5y_7y', 'supply/active_7y_10y', 'supply/active_more_10y', 'supply/issued', 'supply/inflation_rate']


# In[22]:


# Data Frame of Supply Features
df_supply = import_glassnode_many(supply_feature_list)


# #### Transaction Metrics from Glassnode

# In[23]:


# List of Transaction Features
transaction_feature_list = ['transactions/count', 'transactions/size_sum',
                            'transactions/transfers_volume_sum', 'transactions/transfers_volume_adjusted_sum']


# In[24]:


# Data Frame of Transaction Features
df_transaction = import_glassnode_many(transaction_feature_list)


# In[ ]:


# #### Exchange Metrics from Glassnode - 1 Month Lag

# In[25]:


# List of Exchange Features
#exchange_feature_list = ['transactions/transfers_volume_to_exchanges_sum', 'transactions/transfers_volume_from_exchanges_sum', 'transactions/transfers_to_exchanges_count', 'transactions/transfers_from_exchanges_count', 'distribution/balance_exchanges', 'transactions/transfers_volume_exchanges_net']


# In[26]:


# Data Frame of Exchange Features
#df_exchange = import_glassnode_many(exchange_feature_list)


# #### Indicator-I Metrics from Glassnode

# In[27]:


# List of Indicator-1 Features
indicator1_feature_list = ['indicators/sopr_adjusted', 'indicators/nvt', 'indicators/velocity',
                           'indicators/cdd', 'indicators/reserve_risk', 'indicators/average_dormancy', 'indicators/liveliness']


# In[28]:


# Data Frame of Indicator-1 Features
df_indicator1 = import_glassnode_many(indicator1_feature_list)


# #### Indicator-2 Metrics from Glassnode

# In[29]:


# List of Indicator-2 Features
indicator2_feature_list = ['indicators/asol', 'indicators/sol_1h', 'indicators/sol_1h_24h', 'indicators/sol_1d_1w', 'indicators/sol_1w_1m', 'indicators/sol_1m_3m',
                           'indicators/sol_3m_6m', 'indicators/sol_6m_12m', 'indicators/sol_1y_2y', 'indicators/sol_2y_3y', 'indicators/sol_3y_5y', 'indicators/sol_5y_7y', 'indicators/sol_7y_10y']


# In[30]:


# Data Frame of Indicator-2 Features
df_indicator2 = import_glassnode_many(indicator2_feature_list)


# In[ ]:


# #### Indicator-3 Metrics from Glassnode

# In[31]:


# List of Indicator-3 Features
indicator3_feature_list = ['indicators/net_unrealized_profit_loss', 'indicators/unrealized_profit', 'indicators/unrealized_loss', 'indicators/net_realized_profit_loss', 'indicators/realized_profit',
                           'indicators/realized_loss', 'indicators/nupl_more_155', 'indicators/nupl_less_155', 'indicators/puell_multiple', 'indicators/stock_to_flow_deflection', 'indicators/difficulty_ribbon_compression']


# In[32]:


# Data Frame of Indicator-3 Features
df_indicator3 = import_glassnode_many(indicator3_feature_list)


# #### Addresses Metrics from Glassnode

# In[33]:


# List of Addresses Features
addresses_feature_list = ['addresses/active_count', 'addresses/sending_count', 'addresses/receiving_count', 'addresses/new_non_zero_count', 'addresses/non_zero_count',
                          'addresses/min_point_zero_1_count', 'addresses/min_point_1_count', 'addresses/min_1_count', 'addresses/min_10_count', 'addresses/min_100_count', 'addresses/min_1k_count', 'addresses/min_10k_count']


# In[34]:


# Data Frame of Addresses Features
df_addresses = import_glassnode_many(addresses_feature_list)


# #### Futures Metrics from Glassnode - Excluded as Data Available Only Since Feb 2020

# ##### List of Futures Features
# futures_feature_list = ['derivatives/futures_volume_daily_all_sum', 'derivatives/futures_volume_daily_perpetual_sum', 'derivatives/futures_open_interest_all_sum', 'derivatives/futures_open_interest_perpetual_sum', 'derivatives/futures_funding_rate_perpetual']

# ##### Data Frame of Future features
# df_futures = import_glassnode_many(futures_feature_list)

# In[ ]:


# ## File Import and Data Tidying

# In[35]:


# Import of Modules and Packages


# In[36]:


# Adjustment of Decimal Places
pd.options.display.float_format = '{:.6f}'.format


# In[37]:


os.chdir('C:\\Users\\markus.gebele\\github_repo\\JupyterLabDir\\Rest\\MA BTC\\Markus_Code_MA_final\\20201010')


# ### Import Blockchain CSV-Files

# #### Market Metric CSV-Files

# In[38]:


# List of Market Features
market_feature_list = ['price_usd_close',
                       'price_drawdown_relative', 'price_realized_usd', 'mvrv']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in market_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x),
                     index_col='Date', )  # decimal="."

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in Market Data Frame
df_market = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_market.columns:
    try:
        df_market["{}".format(y)] = df_market["{}".format(y)
                                              ].str.replace(',', '.')
        df_market["{}".format(y)] = df_market["{}".format(y)].astype(float)
    except:
        pass


# In[39]:


# Latest Market Metric Values
df_market.tail()


# #### Mining Metric CSV-Files

# In[40]:


# # List of Mining Features
# mining_feature_list = ['difficulty_latest'] #, 'hash_rate_mean', 'revenue_sum','marketcap_thermocap_ratio'

# # For Loop through all Features in the List
# for x in mining_feature_list:

#     # Import Feature CSV-Files
#     df_mining_difficulty_latest = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date')

# # Change Data Type of Columns to Float
# for y in df_mining_difficulty_latest.columns:
#     try:
#         df_mining_difficulty_latest["{}".format(y)] = df_mining_difficulty_latest["{}".format(y)].str.replace(',','.')
#         df_mining_difficulty_latest["{}".format(y)] = df_mining_difficulty_latest["{}".format(y)].astype(float)
#     except:
#         pass

# df_mining_difficulty_latest


# In[41]:


# # List of Mining Features
# mining_feature_list = ['hash_rate_mean'] # 'revenue_sum','marketcap_thermocap_ratio'

# # For Loop through all Features in the List
# for x in mining_feature_list:

#     # Import Feature CSV-Files
#     df_mining_hash_rate_mean = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date')

# # Change Data Type of Columns to Float
# for y in df_mining_hash_rate_mean.columns:
#     try:
#         df_mining_hash_rate_mean["{}".format(y)] = df_mining_hash_rate_mean["{}".format(y)].str.replace(',','.')
#         df_mining_hash_rate_mean["{}".format(y)] = df_mining_hash_rate_mean["{}".format(y)].astype(float)
#     except:
#         pass

# df_mining_hash_rate_mean


# In[42]:


# # List of Mining Features
# mining_feature_list = ['marketcap_thermocap_ratio'] # 'revenue_sum','marketcap_thermocap_ratio'

# # For Loop through all Features in the List
# for x in mining_feature_list:

#     # Import Feature CSV-Files
#     df_mining_marketcap_thermocap_ratio = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date')

# # Change Data Type of Columns to Float
# for y in df_mining_marketcap_thermocap_ratio.columns:
#     try:
#         df_mining_marketcap_thermocap_ratio["{}".format(y)] = df_mining_marketcap_thermocap_ratio["{}".format(y)].str.replace(',','.')
#         df_mining_marketcap_thermocap_ratio["{}".format(y)] = df_mining_marketcap_thermocap_ratio["{}".format(y)].astype(float)
#     except:
#         pass

# df_mining_revenue_sum


# In[43]:


# List of Mining Features
# , 'revenue_sum','marketcap_thermocap_ratio'
mining_feature_list = ['difficulty_latest', 'hash_rate_mean']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in mining_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date')

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in Market Data Frame
df_mining = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_mining.columns:
    try:
        df_mining["{}".format(y)] = df_mining["{}".format(y)
                                              ].str.replace(',', '.')
        df_mining["{}".format(y)] = df_mining["{}".format(y)].astype(float)
    except:
        pass


# In[44]:


# Latest Mining Metric Values
df_mining.tail()


# #### Blocks Metric CSV-Files

# In[45]:


# List of Blockchain Features
block_feature_list = ['block_count', 'block_interval_mean', 'block_size_sum']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in block_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date')

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in Market Data Frame
df_blockfeatures = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_blockfeatures.columns:
    try:
        df_blockfeatures["{}".format(
            y)] = df_blockfeatures["{}".format(y)].str.replace(',', '.')
        df_blockfeatures["{}".format(
            y)] = df_blockfeatures["{}".format(y)].astype(float)
    except:
        pass


# In[46]:


# Show Latest Feature Values
df_blockfeatures.tail()


# #### Distribution Metric CSV-Files

# In[47]:


# # List of Distribution Features
# distribution_feature_list = ['balance_1pct_holders', 'gini', 'herfindahl']

# # Counter to Differentiate between Data Frame Setup and Data Frame Extension
# counter=0

# # For Loop through all Features in the List
# for x in distribution_feature_list:

#     # Import Feature CSV-Files
#     df = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date')

#     # Condition - Data Frame Set up or Data Frame Extension
#     if counter > 0:

#         # Data Frame Extension - Inner Join
#         df_new = df_new.join(df, how='inner')

#     else:
#         # Data Frame Setup
#         df_new = df

#     # Increase Counter
#     counter +=1

# # Store Data Frame Copy in Distribution Data Frame
# df_distribution = df_new.copy()

# # Delete Original Data Frame
# del df_new

# # Change Data Type of Columns to Float
# for y in df_distribution.columns:
#     try:
#         df_distribution["{}".format(y)] = df_distribution["{}".format(y)].str.replace(',','.')
#         df_distribution["{}".format(y)] = df_distribution["{}".format(y)].astype(float)
#     except:
#         pass


# In[48]:


# # Latest Distribution Metric Values
# df_distribution.tail()


# #### Fee Metric CSV-Files

# In[49]:


# List of Fee Features
fees_feature_list = ['volume_sum', 'volume_mean', 'fee_ratio_multiple']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in fees_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x),
                     index_col='Date')  # decimal="."

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in Fees Data Frame
df_fees = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_fees.columns:
    try:
        df_fees["{}".format(y)] = df_fees["{}".format(y)].str.replace(',', '.')
        df_fees["{}".format(y)] = df_fees["{}".format(y)].astype(float)
    except:
        pass


# In[50]:


# Latest Fee Metric Values
df_fees.tail()


# #### UTXO Metric CSV-Files

# In[51]:


# List of UTXO Features
utxo_feature_list = ['utxo_created_count', 'utxo_spent_count', 'utxo_created_value_sum',
                     'utxo_spent_value_sum', 'utxo_profit_relative', 'utxo_profit_count', 'utxo_loss_count']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in utxo_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x),
                     index_col='Date', )  # decimal="."

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in UTXO Data Frame
df_utxo = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_utxo.columns:
    try:
        df_utxo["{}".format(y)] = df_utxo["{}".format(y)].str.replace(',', '.')
        df_utxo["{}".format(y)] = df_utxo["{}".format(y)].astype(float)
    except:
        pass


# In[52]:


# Latest UTXO Metric Values
df_utxo.tail()


# #### Supply Metric CSV-Files

# In[53]:


# List of Supply Features
supply_feature_list = ['current', 'profit_relative', 'profit_sum', 'loss_sum', 'active_more_1y_percent', 'active_more_2y_percent', 'active_more_3y_percent', 'active_more_5y_percent', 'active_24h',
                       'active_1d_1w', 'active_1w_1m', 'active_1m_3m', 'active_3m_6m', 'active_6m_12m', 'active_1y_2y', 'active_2y_3y', 'active_3y_5y', 'active_5y_7y', 'active_7y_10y', 'active_more_10y', 'issued', 'inflation_rate']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in supply_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x),
                     index_col='Date', )  # decimal="."

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in Supply Data Frame
df_supply = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_supply.columns:
    try:
        df_supply["{}".format(y)] = df_supply["{}".format(y)
                                              ].str.replace(',', '.')
        df_supply["{}".format(y)] = df_supply["{}".format(y)].astype(float)
    except:
        pass


# In[54]:


# Latest Supply Metric Values
df_supply.tail()


# #### Transaction Metric CSV-Files

# In[55]:


# List of Transaction features
transaction_feature_list = ['count', 'size_sum',
                            'transfers_volume_sum', 'transfers_volume_adjusted_sum']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in transaction_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x),
                     index_col='Date', )  # decimal="."

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in Transaction Data Frame
df_transaction = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_transaction.columns:
    try:
        df_transaction["{}".format(y)] = df_transaction["{}".format(
            y)].str.replace(',', '.')
        df_transaction["{}".format(
            y)] = df_transaction["{}".format(y)].astype(float)
    except:
        pass


# In[56]:


# Latest Transaction Metric Values
df_transaction.tail()


# #### Exchange Metrics CSV-Files

# In[57]:


# # List of Exchange Features
# exchange_feature_list = ['transfers_volume_to_exchanges_sum', 'transfers_volume_from_exchanges_sum', 'transfers_to_exchanges_count',
#                          'transfers_from_exchanges_count', 'balance_exchanges', 'transfers_volume_exchanges_net']

# # Counter to Differentiate between Data Frame Setup and Data Frame Extension
# counter=0

# # For Loop through all Features in the List
# for x in exchange_feature_list:

#     # Import Feature CSV-Files
#     df = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date', ) #decimal="."

#     # Condition - Data Frame Set up or Data Frame Extension
#     if counter > 0:

#         # Data Frame Extension - Inner Join
#         df_new = df_new.join(df, how='inner')

#     else:

#         # Data Frame Setup
#         df_new = df

#     # Increase Counter
#     counter +=1

# # Store Data Frame Copy in Exchange Data Frame
# df_exchange = df_new.copy()

# # Delete Original Data Frame
# del df_new

# # Change Data Type of Columns to Float
# for y in df_exchange.columns:
#     try:
#         df_exchange["{}".format(y)] = df_exchange["{}".format(y)].str.replace(',','.')
#         df_exchange["{}".format(y)] = df_exchange["{}".format(y)].astype(float)
#     except:
#         pass


# In[58]:


# # Latest Exchange Metric Values
# df_exchange.tail()


# #### Indicator-1 Metric CSV-Files

# In[59]:


# List of Indicator-1 features
indicator1_feature_list = ['sopr_adjusted', 'nvt', 'velocity',
                           'cdd', 'reserve_risk', 'average_dormancy', 'liveliness']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in indicator1_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x),
                     index_col='Date', )  # decimal="."

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in Indicator-1 Data Frame
df_indicator1 = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_indicator1.columns:
    try:
        df_indicator1["{}".format(y)] = df_indicator1["{}".format(
            y)].str.replace(',', '.')
        df_indicator1["{}".format(
            y)] = df_indicator1["{}".format(y)].astype(float)
    except:
        pass


# In[60]:


# Latest Indicator-1 Metric Values
df_indicator1.tail()


# #### Indicator-2 Metrics CSV-Files

# In[61]:


# List of Indicator-2 Features
indicator2_feature_list = ['asol', 'sol_1h', 'sol_1h_24h', 'sol_1d_1w', 'sol_1w_1m', 'sol_1m_3m',
                           'sol_3m_6m', 'sol_6m_12m', 'sol_1y_2y', 'sol_2y_3y', 'sol_3y_5y', 'sol_5y_7y', 'sol_7y_10y']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in indicator2_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date')

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in Indicator-2 Data Frame
df_indicator2 = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_indicator2.columns:
    try:
        df_indicator2["{}".format(y)] = df_indicator2["{}".format(
            y)].str.replace(',', '.')
        df_indicator2["{}".format(
            y)] = df_indicator2["{}".format(y)].astype(float)
    except:
        pass


# In[62]:


# Latest Indicator-2 Metric Values
df_indicator2.tail()


# #### Indicator-3 Metrics CSV-Files

# In[63]:


# List of Indicator-3 Features
indicator3_feature_list = ['net_unrealized_profit_loss', 'unrealized_profit', 'unrealized_loss', 'net_realized_profit_loss', 'realized_profit',
                           'realized_loss', 'nupl_more_155', 'nupl_less_155', 'puell_multiple', 'stock_to_flow_deflection', 'difficulty_ribbon_compression']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in indicator3_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date')

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in Indicator-3 Data Frame
df_indicator3 = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_indicator3.columns:
    try:
        df_indicator3["{}".format(y)] = df_indicator3["{}".format(
            y)].str.replace(',', '.')
        df_indicator3["{}".format(
            y)] = df_indicator3["{}".format(y)].astype(float)
    except:
        pass


# In[64]:


# Latest Indicator-3 Metric Values
df_indicator3.tail()


# #### Addresses Metric CSV-Files

# In[65]:


# List of Addresses Features
addresses_feature_list = ['active_count', 'sending_count', 'receiving_count', 'new_non_zero_count', 'non_zero_count',
                          'min_point_zero_1_count', 'min_point_1_count', 'min_1_count', 'min_10_count', 'min_100_count', 'min_1k_count', 'min_10k_count']

# Counter to Differentiate between Data Frame Setup and Data Frame Extension
counter = 0

# For Loop through all Features in the List
for x in addresses_feature_list:

    # Import Feature CSV-Files
    df = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date')

    # Condition - Data Frame Set up or Data Frame Extension
    if counter > 0:

        # Data Frame Extension - Inner Join
        df_new = df_new.join(df, how='inner')

    else:

        # Data Frame Setup
        df_new = df

    # Increase Counter
    counter += 1

# Store Data Frame Copy in Addresses Data Frame
df_addresses = df_new.copy()

# Delete Original Data Frame
del df_new

# Change Data Type of Columns to Float
for y in df_addresses.columns:
    try:
        df_addresses["{}".format(y)] = df_addresses["{}".format(
            y)].str.replace(',', '.')
        df_addresses["{}".format(
            y)] = df_addresses["{}".format(y)].astype(float)
    except:
        pass


# In[66]:


# Latest Addresses Metric Values
df_addresses.tail()


# #### Futures Metrics CSV-Files - Excluded as Data Available Only Since Feb 2020

# ##### List of Future Features
# futures_feature_list = ['futures_volume_daily_all_sum', 'futures_volume_daily_perpetual_sum', 'futures_open_interest_all_sum', 'futures_open_interest_perpetual_sum', 'futures_funding_rate_perpetual']
#
# ##### Counter to Differentiate between Data Frame Setup and Data Frame Extension
# counter=0
#
# ##### For Loop through all Features in the List
# for x in futures_feature_list:
#
#     ##### Import Feature CSV-Files
#     df = pd.read_csv("Glassnode/{}.csv".format(x), index_col='Date')
#
#     ##### Condition - Data Frame Set up or Data Frame Extension
#     if counter > 0:
#
#         ##### Data Frame Extension - Inner Join
#         df_new = df_new.join(df, how='inner')
#
#     else:
#
#         ##### Data Frame Setup
#         df_new = df
#
#     ##### Increase Counter
#     counter +=1
#
# ##### Store Data Frame Copy in Market Data Frame
# df_futures = df_new.copy()
#
# ##### Delete Original Data Frame
# del df_new
#
# ##### Change Data Type of Columns to Float
# for y in df_futures.columns:
#     try:
#         df_futures["{}".format(y)] = df_futures["{}".format(y)].str.replace(',','.')
#         df_futures["{}".format(y)] = df_futures["{}".format(y)].astype(float)
#     except:
#         pass

# ## Target Variables

# In[67]:


df_return = pd.DataFrame(df_market['market/price_usd_close'])


# In[68]:


df_return.columns = ['Closed Price USD']


# In[69]:


# Add Column with BTC Daily Returns in USD
df_return['Daily Return in USD'] = df_return['Closed Price USD'].diff()

# Add Column with BTC Daily Returns in Percent
df_return['Daily Returns in Percent'] = (df_return['Daily Returns USD']) / (
    df_return['Closed Price USD'].shift(1))*100  # Add Column with BTC Daily Log Returns in Percent
df_return['Daily Log Returns in Percent'] = pd.DataFrame(np.log(
    df_return['Closed Price USD']/df_return['Closed Price USD'].shift(1))*100)
# In[70]:


df_return.tail()


# ## Feature Engineering

# ### Volatility

# In[71]:


# Selecting BTC Close Price in USD
df_volatility = pd.DataFrame(df_market['market/price_usd_close'])


# In[72]:


# Rename Price Column
df_volatility.columns = ['Closed Price USD']


# In[73]:


# Add Column with BTC Daily Returns in USD
df_volatility['Daily Return in USD'] = df_volatility['Closed Price USD'].diff()


# In[74]:


# Add Column with BTC Daily Returns in Percent
df_volatility['Daily Return in Percent'] = (
    df_volatility['Daily Return in USD']) / (df_volatility['Closed Price USD'].shift(1))*100


# In[75]:


# Add Column with BTC Daily Log Returns in Percent
df_volatility['Daily Log Return in Percent'] = pd.DataFrame(np.log(
    df_volatility['Closed Price USD']/df_volatility['Closed Price USD'].shift(1))*100)


# In[76]:


df_volatility.tail()


# In[77]:


# Calculation of Rolling Volatility of BTC Daily Log Returns for 10, 20 and 30 days period
df_volatility['Volatility Daily Log Return in Percent 3D'] = pd.DataFrame(
    df_volatility['Daily Log Return in Percent'].rolling(window=3, min_periods=1).std())
df_volatility['Volatility Daily Log Return in Percent 5D'] = pd.DataFrame(
    df_volatility['Daily Log Return in Percent'].rolling(window=5, min_periods=1).std())
df_volatility['Volatility Daily Log Return in Percent 10D'] = pd.DataFrame(
    df_volatility['Daily Log Return in Percent'].rolling(window=10, min_periods=1).std())


# In[78]:


df_volatility.tail()


# In[79]:


df_volatility_reduced = df_volatility['Volatility Daily Log Return in Percent 5D']
df_volatility_reduced['Daily Log Return in Percent'] = df_volatility['Daily Log Return in Percent']


# ### Log Price

# In[80]:


# Selecting BTC Close Price in USD
df_logprice = pd.DataFrame(np.log(df_market['market/price_usd_close']))


# In[81]:


df_logprice.columns = ['Log Price in USD']


# In[82]:


df_logprice.tail()


# # Data Sets

# ## Data Set I. - Blockchain Metrics Only

# In[83]:


# Merging all Blockchain DataFrame Metrics
df_blockchain = pd.concat([df_return, df_logprice, df_volatility_reduced, df_market, df_mining, df_blockfeatures, df_fees, df_utxo, df_supply,
                           df_transaction, df_indicator1, df_indicator2, df_indicator3, df_addresses], axis=1, join='inner')


# In[84]:


df_blockchain = df_blockchain.dropna()


# In[85]:


df_blockchain = df_blockchain.drop(['Closed Price USD'], axis=1)


# In[86]:


df_blockchain.index = df_blockchain.index.astype('datetime64[ns]')


# In[87]:


df_blockchain.info()


# In[88]:


df_blockchain.tail()


# In[89]:


# Summary of each variable
df_blockchain.describe()


# In[90]:


# Creating CSV-File of Blockchain Data
df_blockchain.to_csv("df_dataset1.csv")


# In[ ]:


# # Classification Models - Data Set Variations

# In[91]:


# Import of Modules and Packages


# Logistic Regression

# Deep Neural Networks

# PyCaret


# In[92]:


# Adjustment of Decimal Places
pd.options.display.float_format = '{:.2f}'.format


# # Import Data Sets

# ## Data Set I.

# In[93]:


os.getcwd()


# In[94]:


# CSV Import of Data Set I. - Blockchain Data
df_dataset1 = pd.read_csv(
    'df_dataset1.csv', parse_dates=True, index_col='Date')


# In[95]:


# Daily Return in USD in row 2011-07-16 means the change in "btc dollar price" from 2011-07-15 23:59 to 2011-07-16 23:59
# in row 2011-07-17 means the change in "btc dollar price" from 2011-07-16 23:59 to 2011-07-17 23:59

df_dataset1


# In[96]:


# Data Set Info
df_dataset1.info()


# In[97]:


# List of Data Types
l_dtypes = df_dataset1.dtypes


# In[98]:


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


# In[99]:


# Data Types
df_dataset1.dtypes


# In[100]:


# Summary Statistics
df_dataset1.describe()


# In[101]:


# Dataset Tail
df_dataset1.tail()


# ## Prediction Shift

# ### Data Set I.

# In[102]:


# Shift of Target Column by 1 Day
df_dataset1['Daily Return in USD'] = df_dataset1['Daily Return in USD'].shift(
    1)
df_dataset1['Daily Log Return in Percent'] = df_dataset1['Daily Log Return in Percent'].shift(
    1)

print(df_dataset1['Daily Log Return in Percent'].describe())
# In[103]:
asd

# Check Shift
df_dataset1.tail()


# In[104]:


# Check Empty Cells
df_dataset1.isnull().sum()


# In[105]:


# daily return: von gestern auf heute die preisveränderung deswegen shift später
df_dataset1.isnull()


# In[106]:


# Delete Empty Row
df_dataset1 = df_dataset1.dropna()


# In[107]:


# Re-Check Empty Cells
df_dataset1.isnull().sum()


# # Classification Setup

# ## Create Data Set Copies for Trading Strategy

# In[108]:


# Data Set Copy for Trading Extension
df_dataset1_copy = df_dataset1.copy()


# In[112]:


df_dataset1['Daily Return in USD'].describe()


# ## Categorization - Data Set I.

# In[113]:


# Categorization

df_dataset1.loc[df_dataset1['Daily Return in USD'] >= 5, 'Class'] = 3

df_dataset1.loc[(df_dataset1['Daily Return in USD'] >= 0) & (
    df_dataset1['Daily Return in USD'] <= 5), 'Class'] = 2

df_dataset1.loc[(df_dataset1['Daily Return in USD'] < 0) & (
    df_dataset1['Daily Return in USD'] >= -5), 'Class'] = 1

df_dataset1.loc[df_dataset1['Daily Return in USD'] < -5, 'Class'] = 0


# In[119]:


df_dataset1['Class'].value_counts()


# In[120]:


# Data Set Head
df_dataset1.head()


# In[121]:


for x in range(len(df_dataset1.columns)):
    print(x, df_dataset1.columns[x])


# In[122]:


# Delete Original Target Column
df_dataset1 = df_dataset1.drop(['Daily Return in USD'], axis=1)


# In[123]:


# Check Class Balance
df_dataset1['Class'].value_counts()


# In[124]:


# Data Set Head
df_dataset1.tail()


# # Model Preparation - Data Set I.

# ## Train Test Split - Data Set I.

# In[125]:


# Setup of Predictors (Indpendent Variables) by Excluding Target Variable
predictors = df_dataset1.drop(['Class'], axis=1).values.astype(np.float32)


# In[126]:


# Setup of Categorized Target Variable
target = df_dataset1['Class'].astype(np.float32)


# In[127]:


# Predictor Dimensions
predictors.shape


# In[128]:


for x in range(len(predictors[0])):
    print(x, predictors[0][x])


# In[129]:


# Target Dimensions
target.shape


# In[130]:


# Train-Test-Split Function to Create Training and Test Data Set
X_train, X_test, y_train, y_test = train_test_split(
    predictors, target, test_size=0.3, random_state=41, stratify=target)


# In[131]:


# Dimensions of Predictors for Training
X_train.shape


# In[132]:


# Dimensions of Target for Training
y_train.shape


# In[133]:


# Number of Predictors
n_nodes = predictors.shape[1]
n_nodes


# ### Check of Class Balance

# In[134]:


# Check Training Target of Class 0
y_train[y_train == 0].count()


# In[135]:


# Check Training Target of Class 1
y_train[y_train == 1].count()


# In[136]:


# Check Test Target of Class 0
y_test[y_test == 0].count()


# In[137]:


# Check Test Target of Class 1
y_test[y_test == 1].count()


# # Visualize pred vs reality

# In[138]:


# Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    predictors, target, test_size=0.3, stratify=target)

# Amount of Columns
n_nodes = predictors.shape[1]

# Instantiation of MinMaxScaler
norm_scaler = MinMaxScaler()

# Training and Application of Feature Scaler on Training Data of Predictors
X_train_scaled = pd.DataFrame(norm_scaler.fit_transform(X_train))

# Application of Feature Scaler on Test Data of Predictors
X_test_scaled = norm_scaler.transform(X_test)

# Instantiate Classifier Logistic Regression
lreg = LogisticRegression(max_iter=12000)

# Train Classifier
lreg.fit(X_train_scaled, y_train)

# Prediction of Classes
y_pred = lreg.predict(X_test_scaled)

# Display Confusion Matrix and Classification Report
print(confusion_matrix(y_test, y_pred))
#     print(classification_report(y_test, y_pred))

# Prediction of Probabilities
y_pred_prob = lreg.predict_proba(X_test_scaled)[:, 1]

# Create Data Frame of Actual Values - y_test
df_y_test = pd.DataFrame(y_test)

# Add Predicted Probabilities to Data Frame
df_y_test["y_pred_prob"] = y_pred_prob

# Add Predicted Classes (0 or 1) to Data Frame to Cross-Check Predicted Probabilities
df_y_test["y_pred"] = y_pred


# In[139]:


df_y_test = df_y_test.sort_index()
df_y_test


# In[140]:


df_y_test[df_y_test['Class'] != df_y_test['y_pred']]


# In[141]:


df_y_test[df_y_test['Class'] == df_y_test['y_pred']]


# In[142]:


# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_y_test.index, y=df_y_test['Class'],
                         mode='lines+markers',
                         name='Class - reality'))
fig.add_trace(go.Scatter(x=df_y_test.index, y=df_y_test['y_pred_prob'],
                         mode='lines+markers',
                         name='y_pred'))
fig.show()


# In[150]:


# # Select Actual Daily Returns in USD from the Data Set Copy
# df_dataset1_copy_dailyreturns = df_dataset1_copy[['Daily Return in USD']]

# # Merge Predictions and Daily Returns Data Frame on the Column "Date"
# df_y_test_withdailyreturns = pd.merge(
#     df_y_test, df_dataset1_copy_dailyreturns, on='Date')

# df_dataset1_copy_percentage = df_dataset1_copy['percentage_daily_return']
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


# In[151]:


sdf


# In[ ]:


# ## Feature Scaling - Data Set I.

# In[152]:


# Instantiation of MinMaxScaler and StandardSacler
norm_scaler = MinMaxScaler()
#std_scaler = StandardScaler()


# In[153]:


# Scaler Training and Transformation of Training Set
X_train_scaled = pd.DataFrame(norm_scaler.fit_transform(X_train))


# In[154]:


# Scaler Transformation of Test Set
X_test_scaled = norm_scaler.transform(X_test)


# In[155]:


# Scaled Test Features
X_test_scaled


# In[156]:


# Scaler Transformation of Entire Training Set - Input for Cross-Validation
predictors_scaled = norm_scaler.transform(predictors)


# # Classification Models - Data Set I.

# In[157]:


# Function To Show Accuracy Mean and Accuracy Standard Deviation
def show_scores(scores):
    # Scores Series
    print('Scores:', scores)
    # Score Mean
    print('Mean:', np.mean(scores))
    # Score Standard Deviation
    print('Standard Deviation:', np.std(scores))


# ## Logistic Regression - Data Set I.

# In[158]:


# Instantiate Classifier - LogisticRegression
lreg = LogisticRegression(max_iter=1200)


# In[159]:


# Train Classifier
lreg.fit(X_train_scaled, y_train)


# In[160]:


# Prediction of Classes
y_pred = lreg.predict(X_test_scaled)


# In[161]:


# Prediction of Probabilities
y_pred_prob = lreg.predict_proba(X_test_scaled)[:, 1]


# In[162]:


# Select Predicted Probabilities
y_pred_prob[1:11]


# In[163]:


# Select Actual Classes
np.array(y_test[1:11])


# In[164]:


# Train Accuracy
lreg.score(X_train_scaled, y_train)


# In[165]:


# Test Accuracy
lreg.score(X_test_scaled, y_test)


# In[166]:


# Output of Confusion Matrix
print(confusion_matrix(y_test, y_pred))


# In[167]:


# Output of Classification Report
print(classification_report(y_test, y_pred))


# In[168]:


# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


# In[169]:


# 10-fold Cross Validation including Stratification
skf = StratifiedKFold(shuffle=True, n_splits=10)


# In[170]:


# Compute and print AUC Score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))


# In[171]:


# Cross-Validation for AUC scores
cv_auc = cross_val_score(lreg, predictors_scaled,
                         target, cv=skf, scoring='roc_auc')


# In[172]:


# Print List of AUC scores
print("AUC scores computed using 10-fold cross-validation: {}".format(cv_auc))


# In[173]:


# Cross-Validation for Accuracy Scores
cv_acc = cross_val_score(lreg, predictors_scaled,
                         target, cv=skf, scoring='accuracy')


# In[174]:


# Print List of Accuracy Scores
print("Accuracy scores computed using 10-fold cross-validation: {}".format(cv_acc))


# In[175]:


# Accuracy Results - Mean and Standard Deviation
show_scores(cv_acc)


# ## Random Forest Classifier - Data Set I.

# In[176]:


# Instantiate Classifier - Random Forest
rfc = RandomForestClassifier(
    n_estimators=100, max_leaf_nodes=None, criterion='gini', max_depth=3, random_state=41)


# In[177]:


# Train Classifier
rfc.fit(X_train_scaled, y_train)


# In[178]:


# Prediction of Classes
y_pred = rfc.predict(X_test_scaled)


# In[179]:


# Prediction of Probabilities
y_pred_prob = rfc.predict_proba(X_test_scaled)[:, 1]


# In[180]:


# Select Predicted Probabilities
y_pred_prob[1:11]


# In[181]:


# Train Accuracy
rfc.score(X_train_scaled, y_train)


# In[182]:


# Test Accuracy
rfc.score(X_test_scaled, y_test)


# In[183]:


# Output of Confusion Matrix
print(confusion_matrix(y_test, y_pred))


# In[184]:


# Output of Classification Report
print(classification_report(y_test, y_pred))


# In[185]:


# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


# In[186]:


# 10-fold Cross Validation including Stratification
skf = StratifiedKFold(shuffle=True, n_splits=10)


# In[187]:


# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))


# In[188]:


# Cross-Validation for AUC scores
cv_auc = cross_val_score(rfc, predictors_scaled,
                         target, cv=skf, scoring='roc_auc')


# In[ ]:


# Print List of AUC scores
print("AUC scores computed using 10-fold cross-validation: {}".format(cv_auc))


# In[189]:


# Cross-Validation for Accuracy Scores
cv_acc = cross_val_score(rfc, predictors_scaled,
                         target, cv=skf, scoring='accuracy')


# In[190]:


# Print List of Accuracy Scores
print("Accuracy scores computed using 10-fold cross-validation: {}".format(cv_acc))


# In[191]:


# Accuracy Results - Mean and Standard Deviation
show_scores(cv_acc)


# # Trading Strategy - Data Set I

# ## Trading Strategy in Combination with Logistic Regression

# In[192]:


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


# In[193]:


target


# In[194]:


predictors


# In[195]:


for x in df_dataset1.columns:
    print(x)


# In[196]:


df_dataset1_copy[['Daily Return in USD']]


# In[197]:


df_dataset1_copy["percentage_daily_return"] = df_dataset1_copy["Daily Return in USD"] / \
    df_dataset1_copy["market/price_usd_close"]
df_dataset1_copy["percentage_daily_return"]


# In[198]:


predictors


# In[206]:


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
for x in range(1, 50):

    # Display Number of Runs
    print("Run:", x)

    # Split Data into Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(
        predictors, target, test_size=0.3, stratify=target)

    # Amount of Columns
    n_nodes = predictors.shape[1]

    # Instantiation of MinMaxScaler
    norm_scaler = MinMaxScaler()

    # Training and Application of Feature Scaler on Training Data of Predictors
    X_train_scaled = pd.DataFrame(norm_scaler.fit_transform(X_train))

    # Application of Feature Scaler on Test Data of Predictors
    X_test_scaled = norm_scaler.transform(X_test)

    # Instantiate Classifier Logistic Regression
    lreg = LogisticRegression(max_iter=12000)

    # Train Classifier
    lreg.fit(X_train_scaled, y_train)

    # Prediction of Classes
    y_pred = lreg.predict(X_test_scaled)

    # Display Confusion Matrix and Classification Report
    print(confusion_matrix(y_test, y_pred))
#     print(classification_report(y_test, y_pred))

    # Prediction of Probabilities
    y_pred_prob = lreg.predict_proba(X_test_scaled)[:, 1]

    # Create Data Frame of Actual Values - y_test
    df_y_test = pd.DataFrame(y_test)

    # Add Predicted Probabilities to Data Frame
    df_y_test["y_pred_prob"] = y_pred_prob

    # Add Predicted Classes (0 or 1) to Data Frame to Cross-Check Predicted Probabilities
    df_y_test["y_pred"] = y_pred

    # Select Actual Daily Returns in USD from the Data Set Copy
    df_dataset1_copy_dailyreturns = df_dataset1_copy[['Daily Return in USD']]

    # Merge Predictions and Daily Returns Data Frame on the Column "Date"
    df_y_test_withdailyreturns = pd.merge(
        df_y_test, df_dataset1_copy_dailyreturns, on='Date')

    df_dataset1_copy_percentage = df_dataset1_copy[['percentage_daily_return']]
    df_y_test_withdailyreturns = pd.merge(
        df_y_test_withdailyreturns, df_dataset1_copy_percentage, on='Date')

    # Add Bitcoin Price in USD
    df_y_test_withdailyreturns = pd.merge(
        df_y_test_withdailyreturns, df_dataset1['market/price_usd_close'], on='Date')

    df_y_test_withdailyreturns['y_pred_prob_bereinigt'] = np.where(
        df_y_test_withdailyreturns['y_pred_prob'] < 0.5, 1-df_y_test_withdailyreturns['y_pred_prob'], df_y_test_withdailyreturns['y_pred_prob'])

    df_y_test_withdailyreturns["Einsatz"] = einsatz * \
        df_y_test_withdailyreturns["y_pred_prob_bereinigt"]  # **2

    # Add and Calculate Transaction Costs based on the Bitcoin Price in USD
    df_y_test_withdailyreturns['Transaction Costs in USD'] = df_y_test_withdailyreturns['Einsatz']*0.0003*2

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


# In[207]:


# In[208]:


print(statistics.median(prestakepercentlist))
print(statistics.median(profitpercent))
print(statistics.median(zinseszins))
print(statistics.median(current_profitpercent))
print(statistics.median(current_zinseszins))


# In[209]:


print(statistics.median(prestakepercentlist51))
print(statistics.median(profitpercent51))
print(statistics.median(zinseszins51))
print(statistics.median(current_profitpercent51))
print(statistics.median(current_zinseszins51))


# In[210]:


print(statistics.median(prestakepercentlist52))
print(statistics.median(profitpercent52))
print(statistics.median(zinseszins52))
print(statistics.median(current_profitpercent52))
print(statistics.median(current_zinseszins52))


# In[211]:


print(statistics.median(prestakepercentlist53))
print(statistics.median(profitpercent53))
print(statistics.median(zinseszins53))
print(statistics.median(current_profitpercent53))
print(statistics.median(current_zinseszins53))


# In[205]:


stop


# In[ ]:


current_profitpercent


# In[ ]:


currentpreds_all["Daily Return in USD"][(
    currentpreds_all['Class'] == currentpreds_all['y_pred'])].sort_values()


# In[ ]:


currentpreds_all["Daily Return in USD"][(
    currentpreds_all['Class'] != currentpreds_all['y_pred'])].sort_values()


# In[ ]:


# # Deepdive in prediction

# In[ ]:


current_profitpercent


# In[ ]:


profitpercent


# In[ ]:


df_y_test_withdailyreturns.sort_values("Daily Return in USD")


# In[ ]:


df_y_test_withdailyreturns[df_y_test_withdailyreturns["Class"]
                           != df_y_test_withdailyreturns["y_pred"]].abs().sum()


# In[ ]:


df_y_test_withdailyreturns[df_y_test_withdailyreturns["Class"]
                           == df_y_test_withdailyreturns["y_pred"]].abs().sum()


# In[ ]:


dfrights = df_y_test_withdailyreturns[df_y_test_withdailyreturns["Class"]
                                      == df_y_test_withdailyreturns["y_pred"]]
dfwrongs = df_y_test_withdailyreturns[df_y_test_withdailyreturns["Class"]
                                      != df_y_test_withdailyreturns["y_pred"]]


# In[ ]:


df_y_test_withdailyreturns = df_y_test_withdailyreturns.sort_index()
df_y_test_withdailyreturns["dfrights"] = np.where(
    df_y_test_withdailyreturns["Class"] == df_y_test_withdailyreturns["y_pred"], 1.0, 0.0)
df_y_test_withdailyreturns.head()


# In[ ]:


df_y_test_withdailyreturns['gain'] = np.where(df_y_test_withdailyreturns["dfrights"] == 1.0, df_y_test_withdailyreturns["percentage_daily_return"].abs(
), -1*df_y_test_withdailyreturns["percentage_daily_return"].abs())
df_y_test_withdailyreturns['gain_cumsum'] = df_y_test_withdailyreturns['gain'].cumsum(
)
df_y_test_withdailyreturns['gain_cumsum']


# In[ ]:


fig = px.line(df_y_test_withdailyreturns,
              y=df_y_test_withdailyreturns['gain_cumsum'], title='Algo gains over time')
fig.show()


# In[ ]:


# In[ ]:


# In[ ]:


dfwrongs.abs().index.values


# In[ ]:


dfwrongs.abs().percentage_daily_return.values


# In[ ]:


dfwrongs = dfwrongs.sort_index()
dfwrongs


# In[ ]:


# show perecentages of wrong predictions over time
fig = go.Figure(data=go.Scatter(x=dfwrongs.abs().index,
                                y=dfwrongs.abs().percentage_daily_return.values))
fig.show()


# In[ ]:


# show perecentages of wrong predictions histogramm

dfwrongs.abs().sort_values("percentage_daily_return")
fig = px.histogram(dfwrongs.abs(), x="percentage_daily_return")
fig.show()


# In[ ]:


dfrights.abs().sort_values("percentage_daily_return")
fig = px.histogram(dfrights.abs(), x="percentage_daily_return")
fig.show()


# In[ ]:


dfwrongs["year"] = dfwrongs.index.year
dfwrongs.abs().groupby(['year']).agg(['sum', 'count'])


# In[ ]:


# HERE DIFF LOOK AT percentage_daily_return

dfrights["year"] = dfrights.index.year
dfrights.abs().groupby(['year']).agg(['sum', 'count']) - \
    dfwrongs.abs().groupby(['year']).agg(['sum', 'count'])


# In[ ]:


stop


# # Bet on tomorrow

# In[ ]:


# Split Data into Training and Testing
#X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, stratify=target)

# Amount of Columns
n_nodes = predictors.shape[1]
print("Amount of Columns", n_nodes)

# Instantiation of MinMaxScaler
norm_scaler = MinMaxScaler()

# Training and Application of Feature Scaler on Training Data of Predictors
predictors_scaled = pd.DataFrame(norm_scaler.fit_transform(predictors))

# Instantiate Classifier Logistic Regression
lreg = LogisticRegression(max_iter=12000)

# Train Classifier
lreg.fit(predictors_scaled, target)


# # get x train data of today to predict tomorrow y

# In[ ]:


most_current_day = df_dataset1.tail(1)
dateofprediction = most_current_day.index
most_current_day


# In[ ]:


# Setup of Predictors (Indpendent Variables) by Excluding Target Variable
most_current_day_predictors = most_current_day.drop(
    ['Class'], axis=1).values.astype(np.float32)


# In[ ]:


# Scaler Transformation of Entire Training Set - Input for Cross-Validation
most_current_day_predictors_scaled = norm_scaler.transform(
    most_current_day_predictors)
most_current_day_predictors_scaled


# In[ ]:


# Prediction of Probabilities
most_current_day_pred_prob = lreg.predict_proba(
    most_current_day_predictors_scaled)[:, 1]
most_current_day_pred_prob[0]


# In[ ]:


# Prediction of Classes
most_current_day_pred = lreg.predict(most_current_day_predictors_scaled)
algoprediction = int(most_current_day_pred[0])
algoprediction


# In[ ]:


dateofprediction[0]


# # Binance Trading Bot execution!

# In[ ]:


asd


# In[ ]:


# execute notebook at 1:03 via task scheduler?


# In[ ]:


def future_order(symbol, positionSide, quantity, side):
    try:
        order = client.futures_create_order(symbol=symbol,
                                            positionSide=positionSide,
                                            quantity=quantity,
                                            side=side,
                                            type=Client.ORDER_TYPE_MARKET,
                                            )
    except Exception as e:
        print("an exception occurred - {}".format(e))
        return False
    return True


# In[ ]:


# if we got the data from last day!
if dateofprediction[0] + datetime.timedelta(hours=27) > datetime.datetime.now():
    api_key = ""
    api_secret = ""
    from binance.client import Client
    import time
    client = Client(api_key, api_secret)

    # Hier der Preis der oben bei binance angeeigt wird
    money_longshort_btc_list = []

    active_futures_list = client.futures_position_information(symbol='BTCUSDT')

    for x in range(len(active_futures_list)):
        if active_futures_list[x]['symbol'] == 'BTCUSDT' and float(active_futures_list[x]["positionAmt"]) >= 0.000001:
            print(float(active_futures_list[x]["positionAmt"]))
            money_longshort_btc_list.append(
                float(active_futures_list[x]["positionAmt"]))

    sum_money_longshort_btc = sum(money_longshort_btc_list)

    client.futures_change_leverage(symbol='BTCUSDT', leverage=1)

    # Wenn hier 0 als input kommt vom ml modell dann müssen
    if algoprediction == 0:
        # alle LONG BTC orders verkauft werden (Check ob alle verkauft)
        future_order('BTCUSDT', "LONG",
                     sum_money_longshort_btc, Client.SIDE_SELL)
        time.sleep(5)
        # und dann alle usdt auf SHORT BTC gesetzt werden
        future_order('BTCUSDT', "SHORT",
                     sum_money_longshort_btc, Client.SIDE_BUY)
        timeofbuy_longshort_order = datetime.datetime.now()

    # Wenn hier 1 als input kommt vom ml modell dann müssen
    elif algoprediction == 1:
        # alle SHORT BTC orders verkauft werden (Check ob alle verkauft)
        future_order('BTCUSDT', "SHORT",
                     sum_money_longshort_btc, Client.SIDE_SELL)
        time.sleep(5)
        # und dann alle usdt auf LONG BTC gesetzt werden
        future_order('BTCUSDT', "LONG",
                     sum_money_longshort_btc, Client.SIDE_BUY)
        timeofbuy_longshort_order = datetime.datetime.now()


# In[ ]:


# Hier der Preis der oben bei binance angeeigt wird
entryprice_btc_list = []

active_futures_list = client.futures_position_information(symbol='BTCUSDT')

for x in range(len(active_futures_list)):
    if active_futures_list[x]['symbol'] == 'BTCUSDT' and float(active_futures_list[x]["positionAmt"]) >= 0.000001:
        print(float(active_futures_list[x]["entryPrice"]))
        entryprice_btc_list.append(float(active_futures_list[x]["entryPrice"]))

sum_entryprice_btc_list = sum(entryprice_btc_list)
sum_entryprice_btc_list


# In[ ]:


# bring together with binance bot and add binance btc price into column!
timenow = datetime.datetime.now()
current_btc_price_in_dollar = most_current_day["market/price_usd_close"].iloc[0]

predicted_df = pd.DataFrame([[dateofprediction[0], most_current_day_pred_prob[0], most_current_day_pred[0], timenow, current_btc_price_in_dollar, timeofbuy_longshort_order, sum_entryprice_btc_list]], index=dateofprediction, columns=[
                            "dateofprediction", "most_current_day_pred_prob", "most_current_day_pred", "exact_time_of_prediction", "current_btc_price_in_dollar", "timeofbuy_longshort_order", "binance_sum_entryprice_btc"])
predicted_df


# # Store DF of the one prediction in database

# In[ ]:


engine = create_engine("mysql://root:root@127.0.0.1/btc_predictions")

predicted_df.to_sql(name="btc_preds",
                    con=engine, index=False,
                    if_exists='append',
                    )


# In[ ]:


# In[ ]:


os.getcwd()+"\\Glassnode"


# In[ ]:
