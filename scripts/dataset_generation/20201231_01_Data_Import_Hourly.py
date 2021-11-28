#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Data" data-toc-modified-id="Data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Data</a></span><ul class="toc-item"><li><span><a href="#Data-Import-via-Application-Programming-Interfaces" data-toc-modified-id="Data-Import-via-Application-Programming-Interfaces-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Data Import via Application Programming Interfaces</a></span></li></ul></li><li><span><a href="#set-dir" data-toc-modified-id="set-dir-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>set dir</a></span></li><li><span><a href="#Btc-Eth" data-toc-modified-id="Btc-Eth-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Btc Eth</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Bitcoin-API---Daily-Basis" data-toc-modified-id="Bitcoin-API---Daily-Basis-3.0.1"><span class="toc-item-num">3.0.1&nbsp;&nbsp;</span>Bitcoin API - Daily Basis</a></span></li><li><span><a href="#Bitcoin-API---Hourly-Basis" data-toc-modified-id="Bitcoin-API---Hourly-Basis-3.0.2"><span class="toc-item-num">3.0.2&nbsp;&nbsp;</span>Bitcoin API - Hourly Basis</a></span></li><li><span><a href="#Ethereum-API---Daily-Basis" data-toc-modified-id="Ethereum-API---Daily-Basis-3.0.3"><span class="toc-item-num">3.0.3&nbsp;&nbsp;</span>Ethereum API - Daily Basis</a></span></li><li><span><a href="#Ethereum-API---Hourly-Basis" data-toc-modified-id="Ethereum-API---Hourly-Basis-3.0.4"><span class="toc-item-num">3.0.4&nbsp;&nbsp;</span>Ethereum API - Hourly Basis</a></span></li></ul></li><li><span><a href="#Bitcoin-Feature" data-toc-modified-id="Bitcoin-Feature-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Bitcoin Feature</a></span><ul class="toc-item"><li><span><a href="#Bitcoin-Feature-Daily---Tier-1" data-toc-modified-id="Bitcoin-Feature-Daily---Tier-1-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Bitcoin Feature Daily - Tier 1</a></span></li><li><span><a href="#Bitcoin-Feature-Hourly---Tier-1" data-toc-modified-id="Bitcoin-Feature-Hourly---Tier-1-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Bitcoin Feature Hourly - Tier 1</a></span></li><li><span><a href="#Bitcoin-Feature-Daily---Tier-2" data-toc-modified-id="Bitcoin-Feature-Daily---Tier-2-3.1.3"><span class="toc-item-num">3.1.3&nbsp;&nbsp;</span>Bitcoin Feature Daily - Tier 2</a></span></li></ul></li><li><span><a href="#Bitcoin-Data" data-toc-modified-id="Bitcoin-Data-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Bitcoin Data</a></span><ul class="toc-item"><li><span><a href="#Bitcoin-Data-Daily-Tier-1" data-toc-modified-id="Bitcoin-Data-Daily-Tier-1-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Bitcoin Data Daily Tier 1</a></span></li><li><span><a href="#Bitcoin-Data-Hourly-Tier-1" data-toc-modified-id="Bitcoin-Data-Hourly-Tier-1-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Bitcoin Data Hourly Tier 1</a></span></li><li><span><a href="#Bitcoin-Data-Daily-Tier-2" data-toc-modified-id="Bitcoin-Data-Daily-Tier-2-3.2.3"><span class="toc-item-num">3.2.3&nbsp;&nbsp;</span>Bitcoin Data Daily Tier 2</a></span></li></ul></li><li><span><a href="#Ethereum-Feature" data-toc-modified-id="Ethereum-Feature-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Ethereum Feature</a></span><ul class="toc-item"><li><span><a href="#Ethereum-Feature-Daily---Tier-1" data-toc-modified-id="Ethereum-Feature-Daily---Tier-1-3.3.1"><span class="toc-item-num">3.3.1&nbsp;&nbsp;</span>Ethereum Feature Daily - Tier 1</a></span></li><li><span><a href="#Ethereum-Feature-Hourly---Tier-1" data-toc-modified-id="Ethereum-Feature-Hourly---Tier-1-3.3.2"><span class="toc-item-num">3.3.2&nbsp;&nbsp;</span>Ethereum Feature Hourly - Tier 1</a></span></li><li><span><a href="#Ethereum-Feature-Daily---Tier-2" data-toc-modified-id="Ethereum-Feature-Daily---Tier-2-3.3.3"><span class="toc-item-num">3.3.3&nbsp;&nbsp;</span>Ethereum Feature Daily - Tier 2</a></span></li></ul></li><li><span><a href="#Ethereum-Data" data-toc-modified-id="Ethereum-Data-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Ethereum Data</a></span><ul class="toc-item"><li><span><a href="#Ethereum-Data-Daily---Tier-1" data-toc-modified-id="Ethereum-Data-Daily---Tier-1-3.4.1"><span class="toc-item-num">3.4.1&nbsp;&nbsp;</span>Ethereum Data Daily - Tier 1</a></span></li><li><span><a href="#Ethereum-Data-Hourly---Tier-1" data-toc-modified-id="Ethereum-Data-Hourly---Tier-1-3.4.2"><span class="toc-item-num">3.4.2&nbsp;&nbsp;</span>Ethereum Data Hourly - Tier 1</a></span></li></ul></li><li><span><a href="#Target-Variables-eth-tier-1-h" data-toc-modified-id="Target-Variables-eth-tier-1-h-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>Target Variables eth tier 1 h</a></span><ul class="toc-item"><li><span><a href="#Ethereum-Data-Daily---Tier-2" data-toc-modified-id="Ethereum-Data-Daily---Tier-2-3.5.1"><span class="toc-item-num">3.5.1&nbsp;&nbsp;</span>Ethereum Data Daily - Tier 2</a></span></li></ul></li><li><span><a href="#Target-Variables" data-toc-modified-id="Target-Variables-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>Target Variables</a></span></li></ul></li><li><span><a href="#Financial-features" data-toc-modified-id="Financial-features-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Financial features</a></span><ul class="toc-item"><li><span><a href="#Bond-Yield-CSV-Files" data-toc-modified-id="Bond-Yield-CSV-Files-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Bond Yield CSV-Files</a></span></li><li><span><a href="#Currency-CSV-Files" data-toc-modified-id="Currency-CSV-Files-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Currency CSV-Files</a></span></li><li><span><a href="#Stocks" data-toc-modified-id="Stocks-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Stocks</a></span></li><li><span><a href="#Data-Set-II.---Blockchain-and-Economic-Metrics" data-toc-modified-id="Data-Set-II.---Blockchain-and-Economic-Metrics-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Data Set II. - Blockchain and Economic Metrics</a></span></li></ul></li></ul></div>

# # Data

# ## Data Import via Application Programming Interfaces

# In[1]:


# Import of Modules and Packages
import os
import requests
import json
import datetime
import yfinance as yf
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Adjustment of Decimal Places
pd.options.display.float_format = '{:.2f}'.format


# # set dir

# In[3]:


os.getcwd()


# In[4]:


setdir = 'C:\\Users\\markus.gebele\\github_repo\\JupyterLabDir\\Rest\\MA BTC\\Markus_Code_MA_final\\20201010'
os.chdir(setdir)


# In[5]:


# Current Working Directory
os.getcwd()


# In[6]:


# Glassnode API Key
API_Key = ''


# # Btc Eth

# In[7]:


# Import Function for a Single Feature Variable from Glassnode
def import_glassnode(url, feature_name):
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

    # Plot of DataFrame
    df.plot(figsize=(4.5, 1.75))
    plt.xticks(rotation=45)
    return df


# ### Bitcoin API - Daily Basis

# In[8]:


# Import Function for Several Feature Variables from Glassnode
def import_btc_daily(feature_list):
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
        print(df.head())
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


# ### Bitcoin API - Hourly Basis

# In[9]:


# Import Function for Several Feature Variables from Glassnode
def import_btc_hourly(feature_list):
    # Counter to Differentiate between Data Frame Setup and Data Frame Join
    counter = 0

    # For Loop through all Features in the List
    for feature in feature_list:

        # URL
        url = 'https://api.glassnode.com/v1/metrics/{}?a=btc&i=1h&f=json&api_key={}'.format(
            feature, API_Key)

        # Execute Import Function for a Single Feature Variable
        df = import_glassnode(url, feature)

        # Display Data Frame Tail
        print(df.head())
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


# ### Ethereum API - Daily Basis

# In[10]:


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

        print(feature)

        # Display Data Frame Tail
        print(df.head())
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


# ### Ethereum API - Hourly Basis

# In[11]:


# Import Function for Several Feature Variables from Glassnode
def import_eth_hourly(feature_list):
    # Counter to Differentiate between Data Frame Setup and Data Frame Join
    counter = 0

    # For Loop through all Features in the List
    for feature in feature_list:

        # URL
        url = 'https://api.glassnode.com/v1/metrics/{}?a=eth&i=1h&f=json&api_key={}'.format(
            feature, API_Key)

        # Execute Import Function for a Single Feature Variable
        df = import_glassnode(url, feature)

        # Display Data Frame Tail
        print(df.head())
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


# In[ ]:


# In[ ]:


# ## Bitcoin Feature

# ### Bitcoin Feature Daily - Tier 1

# In[12]:


btc_tier1_list_d = ['addresses/new_non_zero_count', 'addresses/active_count', 'addresses/sending_count', 'addresses/receiving_count', 'addresses/count',
                    'blockchain/block_height', 'blockchain/block_count', 'blockchain/block_interval_mean', 'blockchain/block_interval_median', 'blockchain/block_size_sum',
                    'blockchain/block_size_mean', 'fees/volume_sum', 'fees/volume_mean', 'fees/volume_median', 'indicators/sopr', 'market/price_usd_close', 'market/price_drawdown_relative', 'market/marketcap_usd',
                    'mining/difficulty_latest', 'mining/hash_rate_mean', 'supply/current', 'transactions/count', 'transactions/rate', 'transactions/size_sum',
                    'transactions/size_mean', 'transactions/transfers_volume_sum', 'transactions/transfers_volume_mean', 'transactions/transfers_volume_median',
                    'blockchain/utxo_created_count', 'blockchain/utxo_spent_count', 'blockchain/utxo_count', 'blockchain/utxo_created_value_sum', 'blockchain/utxo_created_value_mean',
                    'blockchain/utxo_created_value_median', 'blockchain/utxo_spent_value_sum', 'blockchain/utxo_spent_value_mean', 'blockchain/utxo_spent_value_median']


# In[13]:


#'indicators/stock_to_flow_ratio', 'indicators/difficulty_ribbon'


# ### Bitcoin Feature Hourly - Tier 1

# In[14]:


btc_tier1_list_h = ['addresses/new_non_zero_count', 'addresses/active_count', 'addresses/sending_count', 'addresses/receiving_count', 'addresses/count',
                    'blockchain/block_height', 'blockchain/block_count', 'blockchain/block_interval_mean', 'blockchain/block_interval_median', 'blockchain/block_size_sum',
                    'blockchain/block_size_mean', 'fees/volume_sum', 'fees/volume_mean', 'fees/volume_median', 'indicators/sopr', 'market/price_usd_close', 'market/price_drawdown_relative', 'market/marketcap_usd',
                    'mining/difficulty_latest', 'mining/hash_rate_mean', 'supply/current', 'transactions/count', 'transactions/rate', 'transactions/size_sum',
                    'transactions/size_mean', 'transactions/transfers_volume_sum', 'transactions/transfers_volume_mean', 'transactions/transfers_volume_median',
                    'blockchain/utxo_created_count', 'blockchain/utxo_spent_count', 'blockchain/utxo_count', 'blockchain/utxo_created_value_sum', 'blockchain/utxo_created_value_mean',
                    'blockchain/utxo_created_value_median', 'blockchain/utxo_spent_value_sum', 'blockchain/utxo_spent_value_mean', 'blockchain/utxo_spent_value_median']


# ### Bitcoin Feature Daily - Tier 2

# In[15]:


btc_tier2_list_d = ['addresses/non_zero_count', 'addresses/min_point_zero_1_count', 'addresses/min_point_1_count', 'addresses/min_1_count', 'addresses/min_10_count',
                    'addresses/min_100_count', 'addresses/min_1k_count', 'addresses/min_10k_count', 'transactions/transfers_volume_to_exchanges_sum',
                    'transactions/transfers_volume_from_exchanges_sum', 'transactions/transfers_volume_exchanges_net', 'transactions/transfers_to_exchanges_count',
                    'transactions/transfers_from_exchanges_count', 'transactions/transfers_volume_to_exchanges_mean', 'transactions/transfers_volume_from_exchanges_mean',
                    'distribution/balance_exchanges', 'fees/fee_ratio_multiple', 'indicators/sopr_adjusted',
                    'indicators/net_unrealized_profit_loss', 'indicators/unrealized_profit', 'indicators/unrealized_loss', 'indicators/realized_profit',
                    'indicators/realized_loss', 'indicators/net_realized_profit_loss', 'indicators/puell_multiple', 'indicators/cdd',
                    'indicators/cdd_supply_adjusted', 'indicators/cdd_supply_adjusted_binary', 'indicators/reserve_risk', 'indicators/liveliness',
                    'indicators/average_dormancy', 'indicators/average_dormancy_supply_adjusted',  'indicators/asol', 'indicators/msol',
                    'indicators/sol_1h', 'indicators/sol_1h_24h', 'indicators/sol_1d_1w', 'indicators/sol_1w_1m', 'indicators/sol_1m_3m',
                    'indicators/sol_3m_6m', 'indicators/sol_6m_12m', 'indicators/sol_1y_2y', 'indicators/sol_2y_3y', 'indicators/sol_3y_5y', 'indicators/sol_5y_7y',
                    'indicators/sol_7y_10y', 'indicators/sol_more_10y', 'indicators/nvt', 'indicators/nvts', 'indicators/velocity', 'indicators/stock_to_flow_deflection',
                    'indicators/difficulty_ribbon_compression', 'indicators/cvdd', 'indicators/rhodl_ratio', 'market/price_realized_usd',
                    'market/marketcap_realized_usd', 'market/mvrv', 'market/mvrv_z_score', 'mining/thermocap', 'mining/marketcap_thermocap_ratio', 'mining/revenue_sum',
                    'mining/revenue_from_fees', 'mining/volume_mined_sum', 'supply/profit_relative', 'supply/profit_sum', 'supply/loss_sum',
                    'supply/active_more_1y_percent', 'supply/active_more_2y_percent', 'supply/active_more_3y_percent', 'supply/active_more_5y_percent',
                    'supply/active_24h', 'supply/active_1d_1w', 'supply/active_1w_1m', 'supply/active_1m_3m', 'supply/active_3m_6m', 'supply/active_6m_12m', 'supply/active_1y_2y',
                    'supply/active_2y_3y', 'supply/active_3y_5y', 'supply/active_5y_7y', 'supply/active_7y_10y', 'supply/active_more_10y', 'supply/issued', 'supply/inflation_rate',
                    'supply/current_adjusted', 'transactions/transfers_volume_adjusted_sum', 'transactions/transfers_volume_adjusted_mean', 'transactions/transfers_volume_adjusted_median',
                    'blockchain/utxo_profit_relative', 'blockchain/utxo_profit_count', 'blockchain/utxo_loss_count']


# In[16]:


# Several Value Outputs
# distribution/balance_exchanges_all, 'indicators/soab', 'indicators/hash_ribbon', 'supply/hodl_waves', 'supply/rcap_hodl_waves',

# Derivatives Several (Exchanges) Output Value + Historic Data: 1 Month

'derivatives/futures_volume_daily_latest', 'derivatives/futures_volume_daily_sum_all', 'derivatives/futures_volume_daily_perpetual_sum_all', 'derivatives/futures_open_interest_latest',
'derivatives/futures_open_interest_sum_all', 'derivatives/futures_open_interest_perpetual_sum_all', 'derivatives/futures_funding_rate_perpetual_all',

# Derivatives Historic Data: 1 Month
'derivatives/futures_volume_daily_all_sum', 'derivatives/futures_volume_daily_perpetual_sum', 'derivatives/futures_open_interest_all_sum',
'derivatives/futures_open_interest_perpetual_sum', 'derivatives/futures_funding_rate_perpetual', 'derivatives/futures_liquidated_volume_long_sum',
'derivatives/futures_liquidated_volume_long_mean', 'derivatives/futures_liquidated_volume_short_sum', 'derivatives/futures_liquidated_volume_short_mean',
# In[17]:


btc_tier1_list_d


# ## Bitcoin Data

# ### Bitcoin Data Daily Tier 1

# In[18]:


# Set Working Directory to Tier 1 BTC Daily
os.chdir('{}/Tier1/BTC_Daily'.format(setdir))


# In[19]:


df_btc_tier1_d = import_btc_daily(btc_tier1_list_d)


# In[20]:


df_btc_tier1_d.tail()


# In[21]:


df_btc_tier1_d.shape


# In[22]:


os.getcwd()


# ### Bitcoin Data Hourly Tier 1

# In[23]:


# Set Working Directory to Tier 1 BTC Hourly
os.chdir('{}/Tier1/BTC_Hourly'.format(setdir))


# In[24]:


df_btc_tier1_h = import_btc_hourly(btc_tier1_list_h)


# In[25]:


df_btc_tier1_h.tail()


# In[26]:


df_btc_tier1_h.shape


# ## Target Variables btc tier 1 h

# In[46]:


df_btc_tier1_h['Closed Price USD'] = df_btc_tier1_h['market/price_usd_close']


# In[47]:


df_btc_tier1_h['Daily Return in USD'] = df_btc_tier1_h['Closed Price USD'].diff()


# In[48]:


# Add Column with BTC Daily Returns in Percent
df_btc_tier1_h['Daily Return in Percent'] = (
    df_btc_tier1_h['Daily Return in USD']) / (df_btc_tier1_h['Closed Price USD'].shift(1))*100


# In[49]:


# Add Column with BTC Daily Log Returns in Percent
df_btc_tier1_h['Daily Log Return in Percent'] = pd.DataFrame(np.log(
    df_btc_tier1_h['Closed Price USD']/df_btc_tier1_h['Closed Price USD'].shift(1))*100)


# In[50]:


df_btc_tier1_h['market/price_usd_close'] = df_btc_tier1_h['Closed Price USD']


# In[51]:


try:
    df_btc_tier1_h = df_btc_tier1_h.drop('market/price_usd_close', axis=1)
except:
    print("was already gone")


# In[52]:


df_btc_tier1_h = df_btc_tier1_h.dropna()


# In[53]:


os.chdir('{}'.format(setdir))


# In[54]:


# Creating CSV-File of Blockchain and Economic Data
df_btc_tier1_h.to_csv('df_btc_tier1_h.csv')


# In[55]:


# Current Working Directory
os.getcwd()


# ### Bitcoin Data Daily Tier 2

# In[27]:


# Set Working Directory to Tier 2 BTC Daily
os.chdir('{}/Tier2/BTC_Daily'.format(setdir))


# In[28]:


df_btc_tier2_d = import_btc_daily(btc_tier2_list_d)


# In[29]:


df_btc_tier2_d.tail()


# In[30]:


df_btc_tier2_d.shape


# In[31]:


for i, word in enumerate(btc_tier1_list_d):
    btc_tier1_list_d[i] = btc_tier1_list_d[i].replace('/', '_')

for i, word in enumerate(btc_tier1_list_h):
    btc_tier1_list_h[i] = btc_tier1_list_h[i].replace('/', '_')

for i, word in enumerate(btc_tier2_list_d):
    btc_tier2_list_d[i] = btc_tier2_list_d[i].replace('/', '_')


# ## Ethereum Feature

# ### Ethereum Feature Daily - Tier 1

# In[32]:


eth_tier1_list_d = ['addresses/new_non_zero_count', 'addresses/active_count', 'addresses/sending_count', 'addresses/receiving_count', 'addresses/count',
                    'blockchain/block_height', 'blockchain/block_count', 'blockchain/block_interval_mean', 'blockchain/block_interval_median', 'blockchain/block_size_sum',
                    'blockchain/block_size_mean', 'fees/volume_sum', 'fees/volume_mean', 'fees/volume_median', 'fees/gas_used_sum', 'fees/gas_used_mean', 'fees/gas_used_median',
                    'fees/gas_price_mean', 'fees/gas_price_median', 'fees/gas_limit_tx_mean', 'fees/gas_limit_tx_median', 'indicators/sopr', 'market/price_usd_close', 'market/price_drawdown_relative', 'market/marketcap_usd',
                    'mining/difficulty_latest', 'mining/hash_rate_mean',
                    'supply/current', 'transactions/count', 'transactions/rate', 'transactions/transfers_count', 'transactions/transfers_rate', 'transactions/transfers_volume_sum',
                    'transactions/transfers_volume_mean', 'transactions/transfers_volume_median'
                    ]

# 'protocols/uniswap_volume_sum', 'protocols/uniswap_transaction_count',


# In[33]:


# 'protocols/uniswap_liquidity_latest',


# ### Ethereum Feature Hourly - Tier 1

# In[34]:


eth_tier1_list_h = ['addresses/new_non_zero_count', 'addresses/active_count', 'addresses/sending_count', 'addresses/receiving_count', 'addresses/count',
                    'blockchain/block_height', 'blockchain/block_count', 'blockchain/block_interval_mean', 'blockchain/block_interval_median', 'blockchain/block_size_sum',
                    'blockchain/block_size_mean', 'fees/volume_sum', 'fees/volume_mean', 'fees/volume_median', 'fees/gas_used_sum', 'fees/gas_used_mean', 'fees/gas_used_median',
                    'fees/gas_price_mean', 'fees/gas_price_median', 'fees/gas_limit_tx_mean', 'fees/gas_limit_tx_median', 'indicators/sopr', 'market/price_usd_close', 'market/price_drawdown_relative', 'market/marketcap_usd',
                    'mining/difficulty_latest', 'mining/hash_rate_mean',
                    'supply/current', 'transactions/count', 'transactions/rate', 'transactions/transfers_count', 'transactions/transfers_rate', 'transactions/transfers_volume_sum',
                    'transactions/transfers_volume_mean', 'transactions/transfers_volume_median']


# ### Ethereum Feature Daily - Tier 2

# In[35]:


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


# In[36]:


# Several Value Outputs
# 'distribution/balance_exchanges_all', 'supply/hodl_waves', 'transactions/transfers_to_miners_count',

# Derivatives Several (Exchanges) Output Value + Historic Data: 1 Month

'derivatives/futures_volume_daily_latest', 'derivatives/futures_volume_daily_sum_all', 'derivatives/futures_volume_daily_perpetual_sum_all', 'derivatives/futures_open_interest_latest',
'derivatives/futures_open_interest_sum_all', 'derivatives/futures_open_interest_perpetual_sum_all', 'derivatives/futures_funding_rate_perpetual_all',

# Derivatives Historic Data: 1 Month
'derivatives/futures_volume_daily_all_sum', 'derivatives/futures_volume_daily_perpetual_sum', 'derivatives/futures_open_interest_all_sum',
'derivatives/futures_open_interest_perpetual_sum', 'derivatives/futures_funding_rate_perpetual', 'derivatives/futures_liquidated_volume_long_sum',
'derivatives/futures_liquidated_volume_long_mean', 'derivatives/futures_liquidated_volume_short_sum', 'derivatives/futures_liquidated_volume_short_mean',
# ## Ethereum Data

# ### Ethereum Data Daily - Tier 1

# In[37]:


# Set Working Directory to Tier 1 ETH Daily
os.chdir('{}/Tier1/ETH_Daily'.format(setdir))


# In[38]:


df_eth_tier1_d = import_eth_daily(eth_tier1_list_d)


# In[39]:


df_eth_tier1_d.tail()


# In[40]:


df_eth_tier1_d.shape


# ### Ethereum Data Hourly - Tier 1

# In[41]:


# Set Working Directory to Tier 1 ETH Hourly
os.chdir('{}/Tier1/ETH_Hourly'.format(setdir))


# In[42]:


df_eth_tier1_h = import_eth_hourly(eth_tier1_list_h)


# In[43]:


df_eth_tier1_h.tail()


# In[44]:


df_eth_tier1_h.shape


# In[45]:


df_eth_tier1_h.columns


# ## Target Variables eth tier 1 h

# In[46]:


df_eth_tier1_h['Closed Price USD'] = df_eth_tier1_h['market/price_usd_close']


# In[47]:


df_eth_tier1_h['Daily Return in USD'] = df_eth_tier1_h['Closed Price USD'].diff()


# In[48]:


# Add Column with BTC Daily Returns in Percent
df_eth_tier1_h['Daily Return in Percent'] = (
    df_eth_tier1_h['Daily Return in USD']) / (df_eth_tier1_h['Closed Price USD'].shift(1))*100


# In[49]:


# Add Column with BTC Daily Log Returns in Percent
df_eth_tier1_h['Daily Log Return in Percent'] = pd.DataFrame(np.log(
    df_eth_tier1_h['Closed Price USD']/df_eth_tier1_h['Closed Price USD'].shift(1))*100)


# In[50]:


df_eth_tier1_h['market/price_usd_close'] = df_eth_tier1_h['Closed Price USD']


# In[51]:


try:
    df_eth_tier1_h = df_eth_tier1_h.drop('market/price_usd_close', axis=1)
except:
    print("was already gone")


# In[52]:


df_eth_tier1_h = df_eth_tier1_h.dropna()


# In[53]:


os.chdir('{}'.format(setdir))


# In[54]:


# Creating CSV-File of Blockchain and Economic Data
df_eth_tier1_h.to_csv('df_eth_tier1_h.csv')


# In[55]:


# Current Working Directory
os.getcwd()


# ### Ethereum Data Daily - Tier 2

# In[56]:


# Set Working Directory to Tier 2 ETH Daily
os.chdir('{}/Tier2/ETH_Daily'.format(setdir))


# In[57]:


df_eth_tier2_d = import_eth_daily(eth_tier2_list_d)


# In[58]:


df_eth_tier2_d


# In[59]:


df_eth_tier2_d.shape


# In[60]:


for i, word in enumerate(eth_tier1_list_d):
    eth_tier1_list_d[i] = eth_tier1_list_d[i].replace('/', '_')

for i, word in enumerate(eth_tier1_list_h):
    eth_tier1_list_h[i] = eth_tier1_list_h[i].replace('/', '_')

for i, word in enumerate(eth_tier2_list_d):
    eth_tier2_list_d[i] = eth_tier2_list_d[i].replace('/', '_')


# In[61]:


df_eth_tier1_d


# In[62]:


df_dataset2_eth = pd.merge(
    df_eth_tier1_d, df_eth_tier2_d, how='inner', left_on='Date', right_on='Date')


# In[63]:


df_dataset2_btc = pd.merge(
    df_btc_tier1_d, df_btc_tier2_d, how='inner', left_on='Date', right_on='Date')


# In[64]:


df_dataset2_btc


# ## Target Variables

# In[65]:


df_dataset2_btc['Closed Price USD'] = df_dataset2_btc['market/price_usd_close']
df_dataset2_eth['Closed Price USD'] = df_dataset2_eth['market/price_usd_close']


# In[66]:


try:
    df_dataset2_btc = df_dataset2_btc.drop('market/price_usd_close', axis=1)
except:
    print("was already gone")

try:
    df_dataset2_eth = df_dataset2_eth.drop('market/price_usd_close', axis=1)
except:
    print("was already gone")


# In[67]:


df_dataset2_btc['Daily Return in USD'] = df_dataset2_btc['Closed Price USD'].diff()
df_dataset2_eth['Daily Return in USD'] = df_dataset2_eth['Closed Price USD'].diff()


# In[68]:


# Add Column with BTC Daily Returns in Percent
df_dataset2_btc['Daily Return in Percent'] = (
    df_dataset2_btc['Daily Return in USD']) / (df_dataset2_btc['Closed Price USD'].shift(1))*100
# Add Column with BTC Daily Returns in Percent
df_dataset2_eth['Daily Return in Percent'] = (
    df_dataset2_eth['Daily Return in USD']) / (df_dataset2_eth['Closed Price USD'].shift(1))*100


# In[69]:


# Add Column with BTC Daily Log Returns in Percent
df_dataset2_btc['Daily Log Return in Percent'] = pd.DataFrame(np.log(
    df_dataset2_btc['Closed Price USD']/df_dataset2_btc['Closed Price USD'].shift(1))*100)
# Add Column with BTC Daily Log Returns in Percent
df_dataset2_eth['Daily Log Return in Percent'] = pd.DataFrame(np.log(
    df_dataset2_eth['Closed Price USD']/df_dataset2_eth['Closed Price USD'].shift(1))*100)


# In[70]:


df_dataset2_btc['Log Price in USD'] = np.log(
    df_dataset2_btc['Closed Price USD'])
df_dataset2_eth['Log Price in USD'] = np.log(
    df_dataset2_eth['Closed Price USD'])


# In[71]:


df_dataset2_btc = df_dataset2_btc.dropna()


# In[72]:


df_dataset2_eth = df_dataset2_eth.dropna()


# In[73]:


os.chdir('{}'.format(setdir))


# In[74]:


# Creating CSV-File of Blockchain and Economic Data
df_dataset2_btc.to_csv('df_btc_tier1_tier2.csv')


# In[75]:


# Creating CSV-File of Blockchain and Economic Data
df_dataset2_eth.to_csv('df_eth_tier1_tier2.csv')


# In[76]:


# Current Working Directory
os.getcwd()


# # Financial features

# In[77]:


setdir


# In[78]:


# Economic Data from Yahoo Finance
# Change of Working Directory
os.chdir('{}/YahooFinance'.format(setdir))


# In[79]:


# Import Function for Feature Variables from Yahoo Finance
def import_economic_data(list):

    data_list = []

    for i in range(len(list)):

        stock = yf.Ticker(list[i])

        # get max. historical market data
        data_list.append(stock.history(period='max'))

    return data_list


# In[80]:


# List of Economic Feature Labels
input_list = ['GC=F', 'SI=F', 'CL=F', 'NG=F', '^FVX', '^TNX', '^TYX', '^DJI', '^RUT', '^GSPC', '^STOXX50E',
              '^IXIC', '^VIX', '^SKEW', 'DX-Y.NYB', 'EURUSD=X', 'JPYUSD=X', 'GBPUSD=X', 'CADUSD=X', 'SEKUSD=X', 'CHFUSD=X']


# In[81]:


# List of Economic Feature Labels
for i, val in enumerate(input_list):
    print(i, val)


# In[82]:


# Import of Financial Features Into a List
data_list = import_economic_data(input_list)


# In[83]:


# Select a Data Frame of List
data_list[0]


# In[84]:


# Creating CSV-Files of all Economic Features

# Counter
i = 0

# For Loop Through Feature List
for x in data_list:

    # Iterative Creation of CSV-Files, Input List as CSV File Name
    x.to_csv("{}.csv".format(input_list[i]), decimal=',')

    # Increase Counter
    i += 1


# In[85]:


# Import Economic CSV-Files


# In[86]:


# Change of Working Directory
os.chdir('{}'.format(setdir))


# In[87]:


# List of Economic Feature Labels
input_list = ['GC=F', 'SI=F', 'CL=F', 'NG=F', '^FVX', '^TNX', '^TYX', '^DJI', '^RUT', '^GSPC', '^STOXX50E',
              '^IXIC', '^VIX', '^SKEW', 'DX-Y.NYB', 'EURUSD=X', 'JPYUSD=X', 'GBPUSD=X', 'CADUSD=X', 'SEKUSD=X', 'CHFUSD=X']


# In[88]:


# Counter
counter = 0

# Create List
data_list = []

# For Loop through Input List
for x in input_list:

    # Import CSV-Files
    df = pd.read_csv("YahooFinance/{}.csv".format(x), index_col='Date')

    # Change Data Type of Columns to Float
    for y in df.columns:
        try:
            df["{}".format(y)] = df["{}".format(y)].str.replace(',', '.')
            df["{}".format(y)] = df["{}".format(y)].astype(float)
        except:
            pass

    # Append Data Frame to List
    data_list.append(df)


# In[89]:


data_list[0].info()


# In[90]:


# Commodity CSV-Files


# In[91]:


# Merge of Commodity CSV-Files
commodities_1 = pd.merge(data_list[0], data_list[1], on=[
                         'Date'], how='inner', suffixes=[' Gold', ' Silver'])


# In[92]:


# Selection of Closed Price Columns
commodities_1 = commodities_1[['Close Gold', 'Close Silver']]


# In[93]:


# Merge of Commodity CSV-Files
commodities_2 = pd.merge(data_list[2], data_list[3], on=[
                         'Date'], how='inner', suffixes=[' WTI Crude Oil', ' Natural Gas'])


# In[94]:


# Selection of Closed Price Columns
commodities_2 = commodities_2[['Close WTI Crude Oil', 'Close Natural Gas']]


# In[95]:


# Concatenation of all Commodity Features
df_commodities = pd.concat(
    [commodities_1, commodities_2], axis=1, join='inner')


# In[96]:


df_commodities.info()


# ## Bond Yield CSV-Files

# In[97]:


# Merge of Bond Yield CSV-Files
bonds_1 = pd.merge(data_list[4], data_list[5], on=['Date'], how='inner', suffixes=[
                   ' Treasury Yield 5 Years', ' Treasury Yield 10 Years'])


# In[98]:


# Selection of Closed Price Columns
bonds_1 = bonds_1[['Close Treasury Yield 5 Years',
                   'Close Treasury Yield 10 Years']]


# In[99]:


# DataFrame of Remaining Feature
bonds_2 = pd.DataFrame(data_list[6])


# In[100]:


# Selection of Closed Price Column
bonds_2 = pd.DataFrame(bonds_2['Close'])


# In[101]:


# Rename Column
bonds_2.columns = ['Treasury Yield 30 Years']


# In[102]:


# Concatentation of Bond Yield Features
df_bonds = pd.concat([bonds_1, bonds_2], axis=1, join='inner')


# In[103]:


df_bonds.info()


# In[104]:


df_bonds.tail()


# In[105]:


# Merge of Index CSV-Files
stocks_1 = pd.merge(data_list[7], data_list[8], on=['Date'], how='inner', suffixes=[
                    ' Dow Jones Industrial Average', ' Russell 2000'])


# In[106]:


# Selection of Closed Price Columns
stocks_1 = stocks_1[[
    'Close Dow Jones Industrial Average', 'Close Russell 2000']]


# In[107]:


# Merge of Index CSV-Files
stocks_2 = pd.merge(data_list[9], data_list[10], on=[
                    'Date'], how='inner', suffixes=[' S&P 500', ' EURO STOXX 50'])


# In[108]:


# Selection of Closed Price Columns
stocks_2 = stocks_2[['Close S&P 500', 'Close EURO STOXX 50']]


# In[109]:


# Merge of Index CSV-Files
stocks_3 = pd.merge(data_list[11], data_list[12], on=[
                    'Date'], how='inner', suffixes=[' NASDAQ', ' VIX'])


# In[110]:


# Selection of Closed Price Columns
stocks_3 = stocks_3[['Close NASDAQ', 'Close VIX']]


# In[111]:


# DataFrame of Remaining Feature
stocks_4 = pd.DataFrame(data_list[13])


# In[112]:


# Selection of Closed Price Column
stocks_4 = stocks_4[['Close']]


# In[113]:


# Rename Column
stocks_4.columns = ['Close CBOE Skew']


# In[114]:


# Concatenation of Index Features
df_stocks = pd.concat([stocks_1, stocks_2, stocks_3,
                       stocks_4], axis=1, join='inner')


# In[115]:


df_stocks.info()


# In[116]:


df_stocks.tail()


# ## Currency CSV-Files

# In[117]:


# Merge of Currency CSV-Files
currencies_1 = pd.merge(data_list[14], data_list[15], on='Date', how='inner', suffixes=[
                        ' USD Index', ' EUR/USD'])


# In[118]:


# Selection of Closed Currency Columns
currencies_1 = currencies_1[['Close USD Index', 'Close EUR/USD']]


# In[119]:


# Merge of Currency CSV-Files
currencies_2 = pd.merge(data_list[16], data_list[17], on='Date', how='inner', suffixes=[
                        ' JPY/USD', ' GBP/USD'])


# In[120]:


# Selection of Closed Currency Columns
currencies_2 = currencies_2[['Close JPY/USD', 'Close GBP/USD']]


# In[121]:


# Merge of Currency CSV-Files
currencies_3 = pd.merge(data_list[18], data_list[19], on='Date', how='inner', suffixes=[
                        ' CAD/USD', ' SEK/USD'])


# In[122]:


# Selection of Closed Currency Columns
currencies_3 = currencies_3[['Close CAD/USD', 'Close SEK/USD']]


# In[123]:


# DataFrame of remaining Exchange Currency
currencies_4 = pd.DataFrame(data_list[20].loc[:, 'Close'])


# In[124]:


# Rename Column
currencies_4.columns = ['Close CHF/USD']


# In[125]:


# Concatenation of Currencies
df_currencies = pd.concat(
    [currencies_1, currencies_2, currencies_3, currencies_4], axis=1, join='inner')


# In[126]:


df_currencies.info()


# In[127]:


df_currencies.tail()


# ## Stocks

# In[128]:


# Calcuation of Daily Changes for Stock Market Variables
df_stocks['Dow Jones Industrial Average Daily Change'] = df_stocks['Close Dow Jones Industrial Average'].diff()
df_stocks['Russell 2000 Daily Change'] = df_stocks['Close Russell 2000'].diff()
df_stocks['S&P 500 Daily Change'] = df_stocks['Close S&P 500'].diff()
df_stocks['EURO STOXX 50 Daily Change'] = df_stocks['Close EURO STOXX 50'].diff()
df_stocks['NASDAQ Daily Change'] = df_stocks['Close NASDAQ'].diff()
df_stocks['VIX Daily Change'] = df_stocks['Close VIX'].diff()
df_stocks['CBOE Skew Daily Change'] = df_stocks['Close CBOE Skew'].diff()


# In[129]:


df_stocks.iloc[-4:-1, :]


# In[130]:


df_stocks = df_stocks.dropna()


# In[131]:


df_stocks.tail()


# ## Data Set II. - Blockchain and Economic Metrics

# In[132]:


df_commodities.index = df_commodities.index.astype('datetime64[ns]')
df_bonds.index = df_bonds.index.astype('datetime64[ns]')
df_stocks.index = df_stocks.index.astype('datetime64[ns]')
df_currencies.index = df_currencies.index.astype('datetime64[ns]')


# In[133]:


# Concatenation of all Financial Featuers
df_economics = pd.concat(
    [df_commodities, df_bonds, df_stocks, df_currencies], axis=1, join='inner')


# In[134]:


df_economics = df_economics.dropna()


# In[135]:


df_economics.info()


# In[136]:


# Summary of each variable
df_economics.describe()


# In[137]:


df_economics


# In[138]:


df_economics = df_economics.dropna()


# In[139]:


os.chdir('{}'.format(setdir))


# In[140]:


# Creating CSV-File of Blockchain and Economic Data
df_economics.to_csv('df_dataset_financial.csv')


# In[ ]:


# In[ ]:


# In[ ]:
