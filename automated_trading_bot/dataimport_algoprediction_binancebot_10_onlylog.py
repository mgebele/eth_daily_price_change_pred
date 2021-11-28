#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data" data-toc-modified-id="Data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data</a></span><ul class="toc-item"><li><span><a href="#Data-Import-via-Application-Programming-Interfaces" data-toc-modified-id="Data-Import-via-Application-Programming-Interfaces-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Data Import via Application Programming Interfaces</a></span><ul class="toc-item"><li><span><a href="#Bitcoin-API---Daily-Basis" data-toc-modified-id="Bitcoin-API---Daily-Basis-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Bitcoin API - Daily Basis</a></span></li><li><span><a href="#Bitcoin-API---Hourly-Basis" data-toc-modified-id="Bitcoin-API---Hourly-Basis-1.1.2"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>Bitcoin API - Hourly Basis</a></span></li><li><span><a href="#Ethereum-API---Daily-Basis" data-toc-modified-id="Ethereum-API---Daily-Basis-1.1.3"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>Ethereum API - Daily Basis</a></span></li><li><span><a href="#Ethereum-API---Hourly-Basis" data-toc-modified-id="Ethereum-API---Hourly-Basis-1.1.4"><span class="toc-item-num">1.1.4&nbsp;&nbsp;</span>Ethereum API - Hourly Basis</a></span></li></ul></li><li><span><a href="#Bitcoin-Feature" data-toc-modified-id="Bitcoin-Feature-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Bitcoin Feature</a></span><ul class="toc-item"><li><span><a href="#Bitcoin-Feature-Daily---Tier-1" data-toc-modified-id="Bitcoin-Feature-Daily---Tier-1-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Bitcoin Feature Daily - Tier 1</a></span></li><li><span><a href="#Bitcoin-Feature-Hourly---Tier-1" data-toc-modified-id="Bitcoin-Feature-Hourly---Tier-1-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>Bitcoin Feature Hourly - Tier 1</a></span></li><li><span><a href="#Bitcoin-Feature-Daily---Tier-2" data-toc-modified-id="Bitcoin-Feature-Daily---Tier-2-1.2.3"><span class="toc-item-num">1.2.3&nbsp;&nbsp;</span>Bitcoin Feature Daily - Tier 2</a></span></li></ul></li><li><span><a href="#Bitcoin-Data" data-toc-modified-id="Bitcoin-Data-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Bitcoin Data</a></span><ul class="toc-item"><li><span><a href="#Bitcoin-Data-Daily-Tier-1" data-toc-modified-id="Bitcoin-Data-Daily-Tier-1-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Bitcoin Data Daily Tier 1</a></span></li><li><span><a href="#Bitcoin-Data-Hourly-Tier-1" data-toc-modified-id="Bitcoin-Data-Hourly-Tier-1-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Bitcoin Data Hourly Tier 1</a></span></li><li><span><a href="#Bitcoin-Data-Daily-Tier-2" data-toc-modified-id="Bitcoin-Data-Daily-Tier-2-1.3.3"><span class="toc-item-num">1.3.3&nbsp;&nbsp;</span>Bitcoin Data Daily Tier 2</a></span></li></ul></li><li><span><a href="#Ethereum-Feature" data-toc-modified-id="Ethereum-Feature-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Ethereum Feature</a></span><ul class="toc-item"><li><span><a href="#Ethereum-Feature-Daily---Tier-1" data-toc-modified-id="Ethereum-Feature-Daily---Tier-1-1.4.1"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>Ethereum Feature Daily - Tier 1</a></span></li><li><span><a href="#Ethereum-Feature-Hourly---Tier-1" data-toc-modified-id="Ethereum-Feature-Hourly---Tier-1-1.4.2"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Ethereum Feature Hourly - Tier 1</a></span></li><li><span><a href="#Ethereum-Feature-Daily---Tier-2" data-toc-modified-id="Ethereum-Feature-Daily---Tier-2-1.4.3"><span class="toc-item-num">1.4.3&nbsp;&nbsp;</span>Ethereum Feature Daily - Tier 2</a></span></li></ul></li><li><span><a href="#Ethereum-Data" data-toc-modified-id="Ethereum-Data-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Ethereum Data</a></span><ul class="toc-item"><li><span><a href="#Ethereum-Data-Daily---Tier-1" data-toc-modified-id="Ethereum-Data-Daily---Tier-1-1.5.1"><span class="toc-item-num">1.5.1&nbsp;&nbsp;</span>Ethereum Data Daily - Tier 1</a></span></li><li><span><a href="#Ethereum-Data-Hourly---Tier-1" data-toc-modified-id="Ethereum-Data-Hourly---Tier-1-1.5.2"><span class="toc-item-num">1.5.2&nbsp;&nbsp;</span>Ethereum Data Hourly - Tier 1</a></span></li><li><span><a href="#Ethereum-Data-Daily---Tier-2" data-toc-modified-id="Ethereum-Data-Daily---Tier-2-1.5.3"><span class="toc-item-num">1.5.3&nbsp;&nbsp;</span>Ethereum Data Daily - Tier 2</a></span></li></ul></li><li><span><a href="#Target-Variables" data-toc-modified-id="Target-Variables-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Target Variables</a></span></li></ul></li><li><span><a href="#Financial-features" data-toc-modified-id="Financial-features-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Financial features</a></span><ul class="toc-item"><li><span><a href="#Bond-Yield-CSV-Files" data-toc-modified-id="Bond-Yield-CSV-Files-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Bond Yield CSV-Files</a></span></li><li><span><a href="#Currency-CSV-Files" data-toc-modified-id="Currency-CSV-Files-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Currency CSV-Files</a></span></li></ul></li><li><span><a href="#Stocks" data-toc-modified-id="Stocks-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Stocks</a></span><ul class="toc-item"><li><span><a href="#Data-Set-II.---Blockchain-and-Economic-Metrics" data-toc-modified-id="Data-Set-II.---Blockchain-and-Economic-Metrics-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Data Set II. - Blockchain and Economic Metrics</a></span></li></ul></li></ul></div>

# # Data Preprocessing

# ## Data Import via Application Programming Interfaces

# In[1]:
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from numpy import inf
from xgboost import XGBClassifier
import xlsxwriter
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import MinMaxScaler
from binance.client import Client

# Import of Modules and Packages
from sqlalchemy import create_engine
import os
import requests
import json
import datetime
import time
import numpy as np
import pandas as pd


d = datetime.datetime(2020, 5, 17)
dateofprediction = [d]
hourd = 80

# Adjustment of Decimal Places
pd.options.display.float_format = '{:.2f}'.format

# Glassnode API Key
API_Key = ''

get_base_wd = os.getcwd()

# Import Function for a Single Feature Variable from Glassnode
algoprediction = 99

filecreationdate = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
filename = "{}_completepipeline.txt".format(filecreationdate)

dirname = os.chdir(get_base_wd+"\\results")

# if not os.path.exists(dirname):
#     os.makedirs(dirname)
# with open(filename, 'w'):
f = open(filename, "a")

os.chdir(get_base_wd+"\\Glassnode")

# doit for 30 minutes
t_end = time.time() + 60 * 30

# if we got the data from last day!
while dateofprediction[0] + datetime.timedelta(hours=hourd) < datetime.datetime.now():

    # ### Blockchain Data from Glassnode
    def import_glassnode(url, feature_name):
        while True:
            try:
                data = requests.get(url).json()
                df = pd.json_normalize(data)

                # Rename columns
                df.columns = ['Date', feature_name]

                # Convert Object to Datetime Object
                df.Date = pd.to_datetime(df.Date, unit='s')
                #df.Date = df.Date.map(lambda x: x.strftime('%Y-%m-%d'))

                # Set Date Column as Index
                df.set_index('Date', inplace=True)

                # Drop Missing Values
                df = df.dropna()

                # Change Data Type to Float
                df = df.astype(float)

                # Creation of CSV-File, Part of Feature Name as CSV File Name
                feature_name = feature_name.replace('/', '_')
                df.to_csv("{}.csv".format(feature_name))
            except:
                time.sleep(60)
                continue

            break

        # Plot of DataFrame
        return df

    # ### Ethereum API - Daily Basis

    # Import Function for Several Feature Variables from Glassnode
    def import_eth_daily(feature_list):
        # Counter to Differentiate between Data Frame Setup and Data Frame Join
        counter = 0

        # For Loop through all Features in the List
        for feature in feature_list:

            # URL
            url = 'https://api.glassnode.com/v1/metrics/{}?a=eth&i=24h&f=json&api_key={}'.format(
                feature, API_Key)

            # Execute Import Function for a Single Feature Variable
            df = import_glassnode(url, feature)

            print(feature, file=f)

            # Display Data Frame Tail
            print(df.head(), file=f)
            print(df.tail(), file=f)

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

    # ## Ethereum Feature

    # ### Ethereum Feature Daily - Tier 1

    eth_tier1_list_d = ['addresses/new_non_zero_count', 'addresses/active_count', 'addresses/sending_count', 'addresses/receiving_count', 'addresses/count',
                        'blockchain/block_height', 'blockchain/block_count', 'blockchain/block_interval_mean', 'blockchain/block_interval_median', 'blockchain/block_size_sum',
                        'blockchain/block_size_mean', 'fees/volume_sum', 'fees/volume_mean', 'fees/volume_median', 'fees/gas_used_sum', 'fees/gas_used_mean', 'fees/gas_used_median',
                        'fees/gas_price_mean', 'fees/gas_price_median', 'fees/gas_limit_tx_mean', 'fees/gas_limit_tx_median', 'indicators/sopr', 'market/price_usd_close', 'market/price_drawdown_relative', 'market/marketcap_usd',
                        'mining/difficulty_latest', 'mining/hash_rate_mean',
                        'supply/current', 'transactions/count', 'transactions/rate', 'transactions/transfers_count', 'transactions/transfers_rate', 'transactions/transfers_volume_sum',
                        'transactions/transfers_volume_mean', 'transactions/transfers_volume_median'
                        ]

    # 'protocols/uniswap_volume_sum', 'protocols/uniswap_transaction_count',

    # In[30]:

    # 'protocols/uniswap_liquidity_latest',

    # ### Ethereum Feature Daily - Tier 2

    # In[32]:

    eth_tier2_list_d = ['addresses/non_zero_count', 'addresses/min_point_zero_1_count', 'addresses/min_point_1_count', 'addresses/min_1_count', 'addresses/min_10_count',
                        'addresses/min_100_count', 'addresses/min_1k_count', 'addresses/min_10k_count', 'addresses/min_32_count', 'distribution/balance_1pct_holders',
                        'distribution/gini', 'distribution/herfindahl', 'distribution/supply_contracts',
                        'transactions/transfers_volume_to_exchanges_sum', 'transactions/transfers_volume_from_exchanges_sum', 'transactions/transfers_volume_exchanges_net',
                        'transactions/transfers_to_exchanges_count', 'transactions/transfers_from_exchanges_count', 'transactions/transfers_volume_to_exchanges_mean',
                        'transactions/transfers_volume_from_exchanges_mean', 'distribution/balance_exchanges', 'fees/fee_ratio_multiple',
                        'indicators/net_unrealized_profit_loss', 'indicators/unrealized_profit', 'indicators/unrealized_loss', 'indicators/cdd', 'indicators/liveliness',
                        'indicators/average_dormancy', 'indicators/asol', 'indicators/msol', 'indicators/nvt', 'indicators/nvts', 'indicators/velocity',
                        'market/marketcap_realized_usd', 'market/mvrv', 'mining/thermocap', 'mining/marketcap_thermocap_ratio', 'mining/revenue_sum', 'mining/revenue_from_fees',
                        'supply/profit_relative', 'supply/profit_sum', 'supply/loss_sum',

                        'supply/active_24h', 'supply/active_1d_1w', 'supply/active_1w_1m', 'supply/active_1m_3m', 'supply/active_3m_6m', 'supply/active_6m_12m', 'supply/active_1y_2y',
                        'supply/active_2y_3y', 'supply/active_3y_5y', 'supply/active_5y_7y', 'supply/active_7y_10y', 'supply/active_more_10y', 'supply/issued', 'supply/inflation_rate',
                        'transactions/contract_calls_internal_count']

    # ### Ethereum Data Daily - Tier 1

    # Set Working Directory to Tier 1 ETH Daily
    os.chdir(get_base_wd+'/Tier1/ETH_Daily')

    df_eth_tier1_d = import_eth_daily(eth_tier1_list_d)

    df_eth_tier1_d.tail()

    df_eth_tier1_d.shape

    # ### Ethereum Data Daily - Tier 2
    # Set Working Directory to Tier 2 ETH Daily
    os.chdir(get_base_wd+'/Tier2/ETH_Daily')

    df_eth_tier2_d = import_eth_daily(eth_tier2_list_d)

    df_eth_tier2_d.shape

    for i, word in enumerate(eth_tier1_list_d):
        eth_tier1_list_d[i] = eth_tier1_list_d[i].replace('/', '_')

    for i, word in enumerate(eth_tier2_list_d):
        eth_tier2_list_d[i] = eth_tier2_list_d[i].replace('/', '_')

    df_dataset2_eth = pd.merge(
        df_eth_tier1_d, df_eth_tier2_d, how='inner', left_on='Date', right_on='Date')

    # ## Target Variables
    df_dataset2_eth['Closed Price USD'] = df_dataset2_eth['market/price_usd_close']

    try:
        df_dataset2_eth = df_dataset2_eth.drop('market/price_usd_close')
    except:
        print("was already gone")

    df_dataset2_eth['Daily Return in USD'] = df_dataset2_eth['Closed Price USD'].diff()

    # Add Column with eth Daily Returns in Percent
    df_dataset2_eth['Daily Return in Percent'] = (
        df_dataset2_eth['Daily Return in USD']) / (df_dataset2_eth['Closed Price USD'].shift(1))*100

    # Add Column with eth Daily Log Returns in Percent
    df_dataset2_eth['Daily Log Return in Percent'] = pd.DataFrame(np.log(
        df_dataset2_eth['Closed Price USD']/df_dataset2_eth['Closed Price USD'].shift(1))*100)

    df_dataset2_eth['Log Price in USD'] = np.log(
        df_dataset2_eth['market/price_usd_close'])

    df_dataset2_eth = df_dataset2_eth.dropna()

    # Creating CSV-File of Blockchain and Economic Data
    df_dataset2_eth.to_csv('df_eth_tier1_tier2.csv')

    #!/usr/bin/env python
    # coding: utf-8

    # <h1>Table of Contents<span class="tocSkip"></span></h1>
    # <div class="toc"><ul class="toc-item"><li><span><a href="#START-Classification-Models---Data-Set-Variations" data-toc-modified-id="START-Classification-Models---Data-Set-Variations-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>START Classification Models - Data Set Variations</a></span></li><li><span><a href="#Import-Data-Sets" data-toc-modified-id="Import-Data-Sets-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import Data Sets</a></span><ul class="toc-item"><li><span><a href="#Data-Set-I." data-toc-modified-id="Data-Set-I.-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Data Set I.</a></span></li></ul></li><li><span><a href="#import-btc-data" data-toc-modified-id="import-btc-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>import btc data</a></span><ul class="toc-item"><li><span><a href="#Prediction-Shift" data-toc-modified-id="Prediction-Shift-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Prediction Shift</a></span><ul class="toc-item"><li><span><a href="#Data-Set-I." data-toc-modified-id="Data-Set-I.-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Data Set I.</a></span></li></ul></li></ul></li><li><span><a href="#get-yearly-btc-dollar-increase-for-traditional-investment" data-toc-modified-id="get-yearly-btc-dollar-increase-for-traditional-investment-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>get yearly btc dollar increase for traditional investment</a></span></li><li><span><a href="#Classification-Setup" data-toc-modified-id="Classification-Setup-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Classification Setup</a></span><ul class="toc-item"><li><span><a href="#Create-Data-Set-Copies-for-Trading-Strategy" data-toc-modified-id="Create-Data-Set-Copies-for-Trading-Strategy-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Create Data Set Copies for Trading Strategy</a></span></li><li><span><a href="#Categorization---Data-Set-I." data-toc-modified-id="Categorization---Data-Set-I.-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Categorization - Data Set I.</a></span></li></ul></li><li><span><a href="#Model-Preparation---Data-Set-I." data-toc-modified-id="Model-Preparation---Data-Set-I.-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Model Preparation - Data Set I.</a></span><ul class="toc-item"><li><span><a href="#Train-Test-Split---Data-Set-I." data-toc-modified-id="Train-Test-Split---Data-Set-I.-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Train Test Split - Data Set I.</a></span><ul class="toc-item"><li><span><a href="#Check-of-Class-Balance" data-toc-modified-id="Check-of-Class-Balance-6.1.1"><span class="toc-item-num">6.1.1&nbsp;&nbsp;</span>Check of Class Balance</a></span></li></ul></li></ul></li><li><span><a href="#Feature-Selection-random-forest" data-toc-modified-id="Feature-Selection-random-forest-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Feature Selection random forest</a></span></li><li><span><a href="#Classification-Models---Data-Set-I." data-toc-modified-id="Classification-Models---Data-Set-I.-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Classification Models - Data Set I.</a></span></li><li><span><a href="#Hyperparam-tuning-ranfo" data-toc-modified-id="Hyperparam-tuning-ranfo-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Hyperparam tuning ranfo</a></span><ul class="toc-item"><li><span><a href="#XGBoost---Data-Set-I." data-toc-modified-id="XGBoost---Data-Set-I.-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>XGBoost - Data Set I.</a></span></li><li><span><a href="#Logistic-Regression---Data-Set-I." data-toc-modified-id="Logistic-Regression---Data-Set-I.-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Logistic Regression - Data Set I.</a></span></li><li><span><a href="#XGBoost---Data-Set-I." data-toc-modified-id="XGBoost---Data-Set-I.-9.3"><span class="toc-item-num">9.3&nbsp;&nbsp;</span>XGBoost - Data Set I.</a></span></li><li><span><a href="#Random-Forest-Classifier---Data-Set-I." data-toc-modified-id="Random-Forest-Classifier---Data-Set-I.-9.4"><span class="toc-item-num">9.4&nbsp;&nbsp;</span>Random Forest Classifier - Data Set I.</a></span></li></ul></li><li><span><a href="#Trading-Strategy---Data-Set-I" data-toc-modified-id="Trading-Strategy---Data-Set-I-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Trading Strategy - Data Set I</a></span><ul class="toc-item"><li><span><a href="#Trading-Strategy-in-Combination-with-Logistic-Regression" data-toc-modified-id="Trading-Strategy-in-Combination-with-Logistic-Regression-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Trading Strategy in Combination with Logistic Regression</a></span></li></ul></li><li><span><a href="#Compare-with-year-2017" data-toc-modified-id="Compare-with-year-2017-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Compare with year 2017</a></span></li><li><span><a href="#Compare-with-year-2018" data-toc-modified-id="Compare-with-year-2018-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Compare with year 2018</a></span></li><li><span><a href="#Compare-with-year-2019" data-toc-modified-id="Compare-with-year-2019-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Compare with year 2019</a></span></li><li><span><a href="#Compare-with-year-20" data-toc-modified-id="Compare-with-year-20-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Compare with year 20</a></span></li><li><span><a href="#Save-Results-as-csv" data-toc-modified-id="Save-Results-as-csv-15"><span class="toc-item-num">15&nbsp;&nbsp;</span>Save Results as csv</a></span></li><li><span><a href="#Deepdive-in-prediction" data-toc-modified-id="Deepdive-in-prediction-16"><span class="toc-item-num">16&nbsp;&nbsp;</span>Deepdive in prediction</a></span></li><li><span><a href="#predict-on-tomorrow" data-toc-modified-id="predict-on-tomorrow-17"><span class="toc-item-num">17&nbsp;&nbsp;</span>predict on tomorrow</a></span></li><li><span><a href="#get-x-train-data-of-today-to-predict-tomorrow-y" data-toc-modified-id="get-x-train-data-of-today-to-predict-tomorrow-y-18"><span class="toc-item-num">18&nbsp;&nbsp;</span>get x train data of today to predict tomorrow y</a></span></li></ul></div>

    # # START Classification Models - Data Set Variations
    # Adjustment of Decimal Places
    pd.options.display.float_format = '{:.2f}'.format
    pd.set_option('display.max_columns', None)

    # In[93]:

    # CSV Import of Data Set I. - Blockchain Data
    df_dataset1 = pd.read_csv(
        'df_eth_tier1_tier2.csv', parse_dates=True, index_col='Date')

    # recalculate the correct "Daily Return in USD" for Friday-Monday problem
    df_dataset1["Daily Return in USD"] = df_dataset1["market/price_usd_close"].diff()

    print(df_dataset1[["Daily Return in USD", "market/price_usd_close"]])

    df_dataset1["Log Price in USD"] = pd.DataFrame(
        np.log(df_dataset1['market/price_usd_close']))
    df_dataset1["Volatility Daily Log Return in Percent 5D"] = 0

    # List of Data Types
    l_dtypes = df_dataset1.dtypes

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

    len(df_dataset1.columns)

    print(df_dataset1.columns, file=f)

    for x in df_dataset1.columns:
        print(x)

    print(df_dataset1[['Daily Return in USD', 'Log Price in USD',
                       'Volatility Daily Log Return in Percent 5D', 'market/price_usd_close']])

    # Shift of Target Column by 1 Day
    df_dataset1['Daily Return in USD_ohneshift'] = df_dataset1['Daily Return in USD']
    df_dataset1["percentage_daily_return_bef_shift"] = df_dataset1["Daily Return in USD_ohneshift"] / \
        df_dataset1["market/price_usd_close"]

    # important for shift!!!
    df_dataset1 = df_dataset1.sort_index(ascending=False)

    df_dataset1['Daily Return in USD'] = df_dataset1['Daily Return in USD'].shift(
        1)

    print(df_dataset1[['Daily Return in USD', 'addresses/active_count', 'Log Price in USD',
                       'Volatility Daily Log Return in Percent 5D', 'market/price_usd_close']])

    df_dataset1['market/price_usd_close_cummax'] = df_dataset1['market/price_usd_close'].cummax()
    df_dataset1['price_usd_close_percent_of_maxtilnow'] = df_dataset1['market/price_usd_close'] / \
        df_dataset1['market/price_usd_close_cummax']

    df_dataset1 = df_dataset1.drop(
        'Volatility Daily Log Return in Percent 5D', axis=1)

    # Delete Empty Row
    predict_change_for24h = df_dataset1.head(1)
    df_dataset1 = df_dataset1.dropna()

    # Data Set Copy for Trading Extension
    df_dataset1_copy = df_dataset1.copy()

    print(df_dataset1_copy[['Closed Price USD',
                            'Daily Return in USD', 'Daily Return in Percent',
                            'Daily Log Return in Percent', 'Log Price in USD',
                            'Daily Return in USD_ohneshift', 'percentage_daily_return_bef_shift',
                            'market/price_usd_close_cummax',
                            'price_usd_close_percent_of_maxtilnow']].head(10))

    # Categorization
    # Class 1 for all Returns >= 0 USD
    df_dataset1.loc[df_dataset1['Daily Return in USD'] >= 0, 'Class'] = 1
    # Class 0 for all Returns < 0 USD
    df_dataset1.loc[df_dataset1['Daily Return in USD'] < 0, 'Class'] = 0

    for x in range(len(df_dataset1.columns)):
        print(x, df_dataset1.columns[x])

    # Delete Original Target Column
    df_dataset1 = df_dataset1.drop(['Daily Return in USD'], axis=1)

    # # Model Preparation - Data Set I.
    # ## Train Test Split - Data Set I.
    # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
    predictors = df_dataset1.drop(['Class'], axis=1).values.astype(np.float32)

    # Setup of Categorized Target Variable
    target = df_dataset1['Class'].astype(np.float32)

    for x in range(len(predictors[0])):
        print(x, predictors[0][x])

    # Select Actual Daily Returns in USD from the Data Set Copy
    df_dataset1_copy_dailyreturns = df_dataset1_copy[['Daily Return in USD']]

    # # Classification Models - Data Set I.
    # # Trading Strategy - Data Set I
    # ## Trading Strategy in Combination with Logistic Regression

    for x in df_dataset1.columns:
        print(x)

    df_dataset1_copy["percentage_daily_return"] = df_dataset1_copy["Daily Return in USD"] / \
        df_dataset1_copy["market/price_usd_close"]

    # # predict on tomorrow
    # Split Data into Training and Testing
    # X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, stratify=target)

    # Amount of Columns
    n_nodes = predictors.shape[1]
    print("Amount of Columns", n_nodes)

    print("predict next24 h")

    most_current_day = predict_change_for24h
    dateofprediction = most_current_day.index
    # Delete Original Target Column
    most_current_day = most_current_day.drop(['Daily Return in USD'], axis=1)

    try:
        # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
        most_current_day_array = most_current_day.drop(
            ['Class'], axis=1).values.astype(np.float32)
    except:
        # Setup of Predictors (Indpendent Variables) by Excluding Target Variable
        most_current_day_array = most_current_day.values.astype(np.float32)

    # most_current_day = most_current_day.values.astype(np.float32)

    all_prob_results = []
    for x in range(30):

        # Instantiate Classifier Logistic Regression
        lreg = RandomForestClassifier(bootstrap=False, min_samples_leaf=4,
                                      min_samples_split=5, n_estimators=200)

        # Train Classifier
        lreg.fit(predictors, target)

        # Prediction of Probabilities
        most_current_day_pred_prob = lreg.predict_proba(
            most_current_day_array)[:, 1]
        all_prob_results.append(most_current_day_pred_prob[0])
        print("all_prob_results", all_prob_results)

    print("statistics.mean(all_prob_results)",
          statistics.mean(all_prob_results))

    predict_change_for24h["algoo"] = float(most_current_day_pred_prob[0])

    print(predict_change_for24h[['Closed Price USD', 'Daily Return in Percent',
                                 'Log Price in USD', 'Daily Return in USD_ohneshift',
                                 'percentage_daily_return_bef_shift',  'algoo']])  # 'Class',

    # Prediction of Classes
    if statistics.mean(all_prob_results) >= 0.5:
        algoprediction = 1
    else:
        algoprediction = 0

    print("algoprediction", algoprediction)

    # In[ ]:

    # Prediction of Classes
    print("most_current_day_pred_prob[0]",
          most_current_day_pred_prob[0], file=f)
    print("algoprediction", algoprediction, file=f)
    print("date one day before the prediction", dateofprediction[0], file=f)


f.close()
