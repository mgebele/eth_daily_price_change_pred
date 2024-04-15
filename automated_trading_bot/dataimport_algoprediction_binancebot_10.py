# Standard libraries
import os
import time
import datetime
import smtplib
import statistics
import pandas as pd
import numpy as np

# Machine learning
from sklearn.ensemble import RandomForestClassifier

# External APIs and Data storage
from binance.client import Client
from sqlalchemy import create_engine
import xlsxwriter

# Plotting
import plotly.graph_objects as go

# Email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration Variables 
API_KEY = "" # Glassnode API Key
EMAIL_SEND = 'gebele.markus@googlemail.com'
DATABASE_URI = "mysql://root:root@127.0.0.1/eth_predictions"

d = datetime.datetime(2020, 5, 17)
dateofprediction = [d]
# f.I.: date one day before the prediction 2021-01-08 00:00:00 ==
# executed on: 2021-01-09 01:28:05
hourd = 52
leverage_amount = 2

# Adjustment of Decimal Places
pd.options.display.float_format = '{:.2f}'.format


get_base_wd = os.getcwd()

# Import Function for a Single Feature Variable from Glassnode
algoprediction = 99

filecreationdate = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
filename = "{}_completepipeline_Bet.txt".format(filecreationdate)

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

    # try:
    #     df_dataset2_eth = df_dataset2_eth.drop('market/price_usd_close', axis = 1)
    # except:
    #     print("was already gone")

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
    #most_current_day_pred = lreg.predict(most_current_day_array)

    if statistics.mean(all_prob_results) >= 0.5:
        algoprediction = 1
    else:
        algoprediction = 0

    #algoprediction = int(most_current_day_pred[0])

    # In[ ]:

    # Prediction of Classes
    print("most_current_day_pred_prob[0]",
          most_current_day_pred_prob[0], file=f)
    print("algoprediction", algoprediction, file=f)
    print("date one day before the prediction", dateofprediction[0], file=f)

    def sendmail_result(pred_proba, algoprediction, current_short_or_long, money_active,
                        timenow, current_eth_price_in_dollar, mean_entryprice_eth_list, after_bet_short_or_long,
                        sum_after_bet_money_longshort_eth_list, sum_unRealizedProfit_list,
                        eths_to_set_1, eths_to_set_2, eths_to_set_3,
                        sum_money_longshort_eth_1, sum_money_longshort_eth_2, sum_money_longshort_eth_3):
        # create message object instance
        msg = MIMEMultipart()

        # please download the html?
        message = "on {} before_short_or_long {} _ before_act {} _ eth_price_glassnode {} _ \n\
                    eths_to_set_1_acquired {}_ eths_to_set_2_acquired {} _ eths_to_set_3_acquired {} _ \n\
                    eths_to_set_1_dissolved {} _eths_to_set_2_dissolved {} _eths_to_set_3_dissolved {} _".format(
            timenow, current_short_or_long, money_active, current_eth_price_in_dollar,
            eths_to_set_1, eths_to_set_2, eths_to_set_3,
            sum_money_longshort_eth_1, sum_money_longshort_eth_2, sum_money_longshort_eth_3)

        # setup the parameters of the message
        password = ""
        msg['From'] = ""
        msg['Subject'] = "Pred: {:.2f} => {} _ {}-Order executed _eth_entryprc_bin {} _ actB {} _currentPL {}".format(
            pred_proba, algoprediction, after_bet_short_or_long, mean_entryprice_eth_list,
            sum_after_bet_money_longshort_eth_list, sum_unRealizedProfit_list)

        # Type in the email recipients
        email_send = 'gebele.markus@googlemail.com'  # ,j.gebele@web.de

        # add in the message body
        msg.attach(MIMEText(message, 'plain'))

        # create server
        server = smtplib.SMTP('smtp.gmail.com: 587')
        server.starttls()

        # Login Credentials for sending the mail
        server.login(msg['From'], password)

        # send the message via the server to one recipient
        # server.sendmail(msg['From'], msg['To'], msg.as_string())

        # send the message via the server to multiple recipients
        server.sendmail(msg['From'], email_send.split(','), msg.as_string())

        server.quit()

        print("successfully sent email to %s" % email_send, file=f)

    # In[204]:

    # # Binance Trading Bot execution!

    # In[206]:

    def future_order(symbol, positionSide, quantity, side):
        try:
            order = client.futures_create_order(symbol=symbol,
                                                positionSide=positionSide,
                                                quantity=quantity,
                                                side=side,
                                                type=Client.ORDER_TYPE_MARKET,
                                                )
        except Exception as e:
            print("an exception occurred - {}".format(e), file=f)
            return False
        return True

    # In[211]:


# if we got the data from last day!
if dateofprediction[0] + datetime.timedelta(hours=hourd) > datetime.datetime.now():
    api_key = ""
    api_secret = ""
    from binance.client import Client
    import time

    client = Client(api_key, api_secret)

    # Hier der Preis der oben bei binance angeeigt wird
    money_longshort_eth_list = []

    active_futures_list = client.futures_position_information(symbol='ETHUSDT')

    for x in range(len(active_futures_list)):
        if active_futures_list[x]['symbol'] == 'ETHUSDT' and (float(active_futures_list[x]["positionAmt"]) >= 0.001 or float(active_futures_list[x]["positionAmt"]) <= -0.001):
            print("active_futures_list[x][positionAmt]", float(
                active_futures_list[x]["positionAmt"]), file=f)
            money_longshort_eth_list.append(
                float(active_futures_list[x]["positionAmt"]))

    sum_money_longshort_eth = abs(sum(money_longshort_eth_list))
    print("sum_money_longshort_eth {}".format(sum_money_longshort_eth), file=f)
    sum_money_longshort_eth_1 = round(sum_money_longshort_eth / 3, 3)
    print("sum_money_longshort_eth_1 {}".format(
        sum_money_longshort_eth_1), file=f)
    sum_money_longshort_eth_2 = round(sum_money_longshort_eth / 3, 3)
    print("sum_money_longshort_eth_2 {}".format(
        sum_money_longshort_eth_2), file=f)
    sum_money_longshort_eth_3 = round(sum_money_longshort_eth -
                                      sum_money_longshort_eth_1 - sum_money_longshort_eth_2, 3)
    print("sum_money_longshort_eth_3 {}".format(
        sum_money_longshort_eth_3), file=f)

    try:
        client.futures_change_leverage(
            symbol='ETHUSDT', leverage=leverage_amount)
    except:
        print("leverage setting to {} did not work?".format(leverage_amount))

    current_short_or_long = "noPosition"
    # get info if im currently short or long on eth
    unRealizedProfit_list = []

    active_futures_list = client.futures_position_information(symbol='ETHUSDT')
    for x in range(len(active_futures_list)):
        if active_futures_list[x]['symbol'] == 'ETHUSDT' and (float(active_futures_list[x]["positionAmt"]) > 0.001 or float(active_futures_list[x]["positionAmt"]) < -0.001):
            current_short_or_long = active_futures_list[x]["positionSide"]
            print("current_short_or_long", current_short_or_long, file=f)
            unRealizedProfit_list.append(
                float(active_futures_list[x]["unRealizedProfit"]))
    sum_unRealizedProfit_list = sum(unRealizedProfit_list)

    # get info either way
    timeofbuy_longshort_order = datetime.datetime.now()

    # try:
    if algoprediction == 0:
        # Wenn hier 0 als input kommt vom ml modell dann müssen

        if current_short_or_long == 'LONG':
            # alle LONG eth orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "LONG",
                         sum_money_longshort_eth_1, Client.SIDE_SELL)
            print("future_order1 LONG eth dissolved {}".format(
                sum_money_longshort_eth_1), file=f)
            time.sleep(5)
            # alle LONG ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "LONG",
                         sum_money_longshort_eth_2, Client.SIDE_SELL)
            print("future_order2 LONG eth dissolved {}".format(
                sum_money_longshort_eth_2), file=f)
            time.sleep(5)
            # alle LONG ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "LONG",
                         sum_money_longshort_eth_3, Client.SIDE_SELL)
            print("future_order3 LONG eth dissolved {}".format(
                sum_money_longshort_eth_3), file=f)

            # get money in tha bank
            # get my current account balance
            futures_acc_balance_list = client.futures_account_balance()
            for x in range(len(futures_acc_balance_list)):
                if futures_acc_balance_list[x]["asset"] == 'USDT':
                    usdt_acc_balance = futures_acc_balance_list[x]["balance"]
                    usdt_acc_balance = float(usdt_acc_balance)
                    print("usdt_acc_balance: ", usdt_acc_balance, file=f)
            # current eth futures price
            current_eth_price_info = client.futures_mark_price(
                symbol='ETHUSDT')
            current_eth_price = float(current_eth_price_info["markPrice"])
            print("current_eth_price: {}".format(
                current_eth_price), file=f)
            # get affordable eth to buy
            eths_to_set = round(usdt_acc_balance/current_eth_price, 3)
            print("eths_to_set: {}".format(eths_to_set), file=f)
            eths_to_set = eths_to_set * leverage_amount - 0.002
            print("eths_to_set {}er hebel: {}".format(
                leverage_amount, eths_to_set), file=f)

            eths_to_set_1 = round(eths_to_set / 3, 3)
            eths_to_set_2 = round(eths_to_set / 3, 3)
            eths_to_set_3 = round(
                eths_to_set - eths_to_set_1 - eths_to_set_2, 3)

            time.sleep(1)

            # und dann alle usdt auf SHORT ETH gesetzt werden
            future_order('ETHUSDT', "SHORT",
                         eths_to_set_1, Client.SIDE_SELL)
            print("future_order1 SHORT eth acquired {}".format(
                eths_to_set_1), file=f)
            time.sleep(5)
            # und dann alle usdt auf SHORT ETH gesetzt werden
            future_order('ETHUSDT', "SHORT",
                         eths_to_set_2, Client.SIDE_SELL)
            print("future_order2 SHORT eth acquired {}".format(
                eths_to_set_2), file=f)
            time.sleep(5)

            # did not execute before because price rose and
            # to less money in acc for calc eths to set
            # get rest money in tha bank
            futures_acc_balance_list = client.futures_account_balance()
            for x in range(len(futures_acc_balance_list)):
                if futures_acc_balance_list[x]["asset"] == 'USDT':
                    rest_to_bet = futures_acc_balance_list[x]["withdrawAvailable"]
                    rest_to_bet = float(rest_to_bet)
                    print("usdt_rest_to_bet: ", rest_to_bet, file=f)

            # manchmal gibt es usdt preisveränderungen oder eth preisveränderungen
            # weshalb manchmal die 3te order nicht ausgeführt wird
            # wegen insufficent margin - 3 % ist abzug sollte reichen
            rest_to_bet = rest_to_bet - rest_to_bet * 0.03
            # current eth futures price
            current_eth_price_info = client.futures_mark_price(
                symbol='ETHUSDT')
            current_eth_price = float(current_eth_price_info["markPrice"])
            print("current_eth_price: {}".format(
                current_eth_price), file=f)
            # get affordable eth to buy
            rest_eths_to_set = round(rest_to_bet/current_eth_price, 3)
            print("eths_to_set: {}".format(rest_eths_to_set), file=f)
            rest_eths_to_set = rest_eths_to_set * leverage_amount - 0.002
            rest_eths_to_set = round(rest_eths_to_set, 3)
            print("eths_to_set {}er hebel: {}".format(
                leverage_amount, rest_eths_to_set), file=f)

            # und dann alle usdt auf SHORT ETH gesetzt werden
            future_order('ETHUSDT', "SHORT",
                         rest_eths_to_set, Client.SIDE_SELL)
            print("future_order3 SHORT eth acquired {}".format(
                rest_eths_to_set), file=f)

        else:
            # alle SHORT ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "SHORT",
                         sum_money_longshort_eth_1, Client.SIDE_BUY)
            print("future_order1 SHORT eth dissolved {}".format(
                sum_money_longshort_eth_1), file=f)
            time.sleep(5)
            # alle SHORT ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "SHORT",
                         sum_money_longshort_eth_2, Client.SIDE_BUY)
            print("future_order2 SHORT eth dissolved {}".format(
                sum_money_longshort_eth_2), file=f)
            time.sleep(5)
            # alle SHORT ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "SHORT",
                         sum_money_longshort_eth_3, Client.SIDE_BUY)
            print("future_order3 SHORT eth dissolved {}".format(
                sum_money_longshort_eth_3), file=f)

            # get money in tha bank
            # get my current account balance
            futures_acc_balance_list = client.futures_account_balance()
            for x in range(len(futures_acc_balance_list)):
                if futures_acc_balance_list[x]["asset"] == 'USDT':
                    usdt_acc_balance = futures_acc_balance_list[x]["balance"]
                    usdt_acc_balance = float(usdt_acc_balance)
                    print("usdt_acc_balance: ", usdt_acc_balance, file=f)
            # current eth futures price
            current_eth_price_info = client.futures_mark_price(
                symbol='ETHUSDT')
            current_eth_price = float(current_eth_price_info["markPrice"])

            # get affordable eth to buy
            eths_to_set = round(usdt_acc_balance/current_eth_price, 3)
            print("eths_to_set: {}".format(eths_to_set), file=f)
            eths_to_set = eths_to_set * leverage_amount - 0.002
            print("eths_to_set {}er hebel: {}".format(
                leverage_amount, eths_to_set), file=f)

            eths_to_set_1 = round(eths_to_set / 3, 3)
            eths_to_set_2 = round(eths_to_set / 3, 3)
            eths_to_set_3 = round(
                eths_to_set - eths_to_set_1 - eths_to_set_2, 3)

            time.sleep(1)

            # und dann alle usdt auf SHORT ETH gesetzt werden
            future_order('ETHUSDT', "SHORT",
                         eths_to_set_1, Client.SIDE_SELL)
            print("future_order1 SHORT eth acquired {}".format(
                eths_to_set_1), file=f)
            time.sleep(5)
            # und dann alle usdt auf SHORT ETH gesetzt werden
            future_order('ETHUSDT', "SHORT",
                         eths_to_set_2, Client.SIDE_SELL)
            print("future_order2 SHORT eth acquired {}".format(
                eths_to_set_2), file=f)
            time.sleep(5)

            # did not execute before because price rose and
            # to less money in acc for calc eths to set
            # get rest money in tha bank
            futures_acc_balance_list = client.futures_account_balance()
            for x in range(len(futures_acc_balance_list)):
                if futures_acc_balance_list[x]["asset"] == 'USDT':
                    rest_to_bet = futures_acc_balance_list[x]["withdrawAvailable"]
                    rest_to_bet = float(rest_to_bet)
                    print("usdt_rest_to_bet: ", rest_to_bet, file=f)

            # manchmal gibt es usdt preisveränderungen oder eth preisveränderungen
            # weshalb manchmal die 3te order nicht ausgeführt wird
            # wegen insufficent margin
            rest_to_bet = rest_to_bet - rest_to_bet * 0.03
            # current eth futures price
            current_eth_price_info = client.futures_mark_price(
                symbol='ETHUSDT')
            current_eth_price = float(current_eth_price_info["markPrice"])
            print("current_eth_price: {}".format(
                current_eth_price), file=f)
            # get affordable eth to buy
            rest_eths_to_set = round(rest_to_bet/current_eth_price, 3)
            print("eths_to_set: {}".format(rest_eths_to_set), file=f)
            rest_eths_to_set = rest_eths_to_set * leverage_amount - 0.002
            rest_eths_to_set = round(rest_eths_to_set, 3)
            print("eths_to_set {}er hebel: {}".format(
                leverage_amount, rest_eths_to_set), file=f)

            # und dann alle usdt auf SHORT ETH gesetzt werden
            future_order('ETHUSDT', "SHORT",
                         rest_eths_to_set, Client.SIDE_SELL)
            print("future_order3 SHORT eth acquired {}".format(
                rest_eths_to_set), file=f)

    # Wenn hier 1 als input kommt vom ml modell dann müssen
    elif algoprediction == 1:
        # check ob wir SHORT ETH orders haben??
        if current_short_or_long == 'SHORT':
            # alle SHORT ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "SHORT",
                         sum_money_longshort_eth_1, Client.SIDE_BUY)
            print("future_order1 SHORT eth dissolved {}".format(
                sum_money_longshort_eth_1), file=f)
            time.sleep(5)
            # alle SHORT ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "SHORT",
                         sum_money_longshort_eth_2, Client.SIDE_BUY)
            print("future_order2 SHORT eth dissolved {}".format(
                sum_money_longshort_eth_2), file=f)
            time.sleep(5)
            # alle SHORT ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "SHORT",
                         sum_money_longshort_eth_3, Client.SIDE_BUY)
            print("future_order3 SHORT eth dissolved {}".format(
                sum_money_longshort_eth_3), file=f)

            # get money in tha bank
            # get my current account balance
            futures_acc_balance_list = client.futures_account_balance()
            for x in range(len(futures_acc_balance_list)):
                if futures_acc_balance_list[x]["asset"] == 'USDT':
                    usdt_acc_balance = futures_acc_balance_list[x]["balance"]
                    usdt_acc_balance = float(usdt_acc_balance)
                    print("usdt_acc_balance: ", usdt_acc_balance, file=f)
            # current eth futures price
            current_eth_price_info = client.futures_mark_price(
                symbol='ETHUSDT')
            current_eth_price = float(current_eth_price_info["markPrice"])
            # get affordable eth to buy
            eths_to_set = round(usdt_acc_balance/current_eth_price, 3)
            print("eths_to_set: {}".format(eths_to_set), file=f)
            eths_to_set = eths_to_set * leverage_amount - 0.002
            print("eths_to_set {}er hebel: {}".format(
                leverage_amount, eths_to_set), file=f)

            eths_to_set_1 = round(eths_to_set / 3, 3)
            eths_to_set_2 = round(eths_to_set / 3, 3)
            eths_to_set_3 = round(
                eths_to_set - eths_to_set_1 - eths_to_set_2, 3)

            time.sleep(1)

            # und dann alle usdt auf LONG ETH gesetzt werden
            future_order('ETHUSDT', "LONG",
                         eths_to_set_1, Client.SIDE_BUY)
            print("future_order1 LONG eth acquired {}".format(
                eths_to_set_1), file=f)
            time.sleep(5)
            # und dann alle usdt auf LONG ETH gesetzt werden
            future_order('ETHUSDT', "LONG",
                         eths_to_set_2, Client.SIDE_BUY)
            print("future_order2 LONG eth acquired {}".format(
                eths_to_set_2), file=f)
            time.sleep(5)

            # did not execute before because price rose and
            # to less money in acc for calc eths to set
            # get rest money in tha bank
            futures_acc_balance_list = client.futures_account_balance()
            for x in range(len(futures_acc_balance_list)):
                if futures_acc_balance_list[x]["asset"] == 'USDT':
                    rest_to_bet = futures_acc_balance_list[x]["withdrawAvailable"]
                    rest_to_bet = float(rest_to_bet)
                    print("usdt_rest_to_bet: ", rest_to_bet, file=f)

            # manchmal gibt es usdt preisveränderungen oder eth preisveränderungen
            # weshalb manchmal die 3te order nicht ausgeführt wird
            # wegen insufficent margin
            rest_to_bet = rest_to_bet - rest_to_bet * 0.03
            # current eth futures price
            current_eth_price_info = client.futures_mark_price(
                symbol='ETHUSDT')
            current_eth_price = float(current_eth_price_info["markPrice"])
            print("current_eth_price: {}".format(
                current_eth_price), file=f)
            # get affordable eth to buy
            rest_eths_to_set = round(rest_to_bet/current_eth_price, 3)
            print("eths_to_set: {}".format(rest_eths_to_set), file=f)
            rest_eths_to_set = rest_eths_to_set * leverage_amount - 0.002
            rest_eths_to_set = round(rest_eths_to_set, 3)
            print("eths_to_set {}er hebel: {}".format(
                leverage_amount, rest_eths_to_set), file=f)

            # und dann alle usdt auf LONG ETH gesetzt werden
            future_order('ETHUSDT', "LONG",
                         rest_eths_to_set, Client.SIDE_BUY)
            print("future_order3 LONG eth acquired {}".format(
                rest_eths_to_set), file=f)

        else:
            # alle LONG ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "LONG",
                         sum_money_longshort_eth_1, Client.SIDE_SELL)
            print("future_order1 LONG eth dissolved {}".format(
                sum_money_longshort_eth_1), file=f)
            time.sleep(5)
            # alle LONG ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "LONG",
                         sum_money_longshort_eth_2, Client.SIDE_SELL)
            print("future_order2 LONG eth dissolved {}".format(
                sum_money_longshort_eth_2), file=f)
            time.sleep(5)
            # alle LONG ETH orders verkauft werden (Check ob alle verkauft)
            future_order('ETHUSDT', "LONG",
                         sum_money_longshort_eth_3, Client.SIDE_SELL)
            print("future_order3 LONG eth dissolved {}".format(
                sum_money_longshort_eth_3), file=f)

            # get money in tha bank
            # get my current account balance
            futures_acc_balance_list = client.futures_account_balance()
            for x in range(len(futures_acc_balance_list)):
                if futures_acc_balance_list[x]["asset"] == 'USDT':
                    usdt_acc_balance = futures_acc_balance_list[x]["balance"]
                    usdt_acc_balance = float(usdt_acc_balance)
                    print("usdt_acc_balance: ", usdt_acc_balance, file=f)
            # current eth futures price
            current_eth_price_info = client.futures_mark_price(
                symbol='ETHUSDT')
            current_eth_price = float(current_eth_price_info["markPrice"])
            print("current_eth_price: {}".format(
                current_eth_price), file=f)
            # get affordable eth to buy
            eths_to_set = round(usdt_acc_balance/current_eth_price, 3)
            print("eths_to_set: {}".format(eths_to_set), file=f)
            eths_to_set = eths_to_set * leverage_amount - 0.002
            print("eths_to_set {}er hebel: {}".format(
                leverage_amount, eths_to_set), file=f)

            eths_to_set_1 = round(eths_to_set / 3, 3)
            eths_to_set_2 = round(eths_to_set / 3, 3)
            # the third is not needed! see below
            eths_to_set_3 = round(
                eths_to_set - eths_to_set_1 - eths_to_set_2, 3)

            time.sleep(1)

            # und dann alle usdt auf LONG ETH gesetzt werden
            future_order('ETHUSDT', "LONG",
                         eths_to_set_1, Client.SIDE_BUY)
            print("future_order1 LONG eth acquired {}".format(
                eths_to_set_1), file=f)
            time.sleep(5)
            # und dann alle usdt auf LONG ETH gesetzt werden
            future_order('ETHUSDT', "LONG",
                         eths_to_set_2, Client.SIDE_BUY)
            print("future_order2 LONG eth acquired {}".format(
                eths_to_set_2), file=f)
            time.sleep(5)

            # did not execute before because price rose and
            # to less money in acc for calc eths to set
            # get rest money in tha bank
            futures_acc_balance_list = client.futures_account_balance()
            for x in range(len(futures_acc_balance_list)):
                if futures_acc_balance_list[x]["asset"] == 'USDT':
                    rest_to_bet = futures_acc_balance_list[x]["withdrawAvailable"]
                    rest_to_bet = float(rest_to_bet)
                    print("usdt_rest_to_bet: ", rest_to_bet, file=f)

            # manchmal gibt es usdt preisveränderungen oder eth preisveränderungen
            # weshalb manchmal die 3te order nicht ausgeführt wird
            # wegen insufficent margin
            rest_to_bet = rest_to_bet - rest_to_bet * 0.03
            # current eth futures price
            current_eth_price_info = client.futures_mark_price(
                symbol='ETHUSDT')
            current_eth_price = float(current_eth_price_info["markPrice"])
            print("current_eth_price: {}".format(
                current_eth_price), file=f)
            # get affordable eth to buy
            rest_eths_to_set = round(rest_to_bet/current_eth_price, 3)
            print("eths_to_set: {}".format(rest_eths_to_set), file=f)
            rest_eths_to_set = rest_eths_to_set * leverage_amount - 0.002
            rest_eths_to_set = round(rest_eths_to_set, 3)
            print("eths_to_set {}er hebel: {}".format(
                leverage_amount, rest_eths_to_set), file=f)

            # und dann alle usdt auf LONG ETH gesetzt werden
            future_order('ETHUSDT', "LONG",
                         rest_eths_to_set, Client.SIDE_BUY)
            print("future_order3 LONG eth acquired {}".format(
                rest_eths_to_set), file=f)

    else:
        print("sth went wrong, algoprediction has weird value: ",
              algoprediction, file=f)

    # # Prepare data for DB
    # Hier der Preis der oben bei binance angeeigt wird
    active_futures_list = client.futures_position_information(
        symbol='ETHUSDT')

    entryprice_eth_list = []
    for x in range(len(active_futures_list)):
        if active_futures_list[x]['symbol'] == 'ETHUSDT' and (float(active_futures_list[x]["positionAmt"]) > 0.001 or float(active_futures_list[x]["positionAmt"]) < -0.001):
            entryprice_eth_list.append(
                float(active_futures_list[x]["entryPrice"]))

    mean_entryprice_eth_list = statistics.mean(entryprice_eth_list)

    # bring together with binance bot and add binance eth price into column!
    timenow = datetime.datetime.now()
    current_eth_price_in_dollar = most_current_day["market/price_usd_close"].iloc[0]

    # get info after bet
    # get info if im currently short or long on eth
    after_bet_money_longshort_eth_list = []
    for x in range(len(active_futures_list)):
        if active_futures_list[x]['symbol'] == 'ETHUSDT' and (float(active_futures_list[x]["positionAmt"]) > 0.001 or float(active_futures_list[x]["positionAmt"]) < -0.001):
            after_bet_short_or_long = active_futures_list[x]["positionSide"]
            after_bet_money_longshort_eth_list.append(
                float(active_futures_list[x]["positionAmt"]))

    sum_after_bet_money_longshort_eth_list = sum(
        after_bet_money_longshort_eth_list)

    pred_proba = float(most_current_day_pred_prob[0])
    money_active = sum_money_longshort_eth

    predicted_df = pd.DataFrame([[most_current_day_pred_prob[0], algoprediction,
                                  timenow, current_eth_price_in_dollar, timeofbuy_longshort_order, mean_entryprice_eth_list,
                                  eths_to_set_1, eths_to_set_2, eths_to_set_3,
                                  sum_money_longshort_eth_1, sum_money_longshort_eth_2, sum_money_longshort_eth_3,
                                  after_bet_short_or_long, sum_unRealizedProfit_list, sum_after_bet_money_longshort_eth_list,
                                  ]], index=dateofprediction, columns=[
                                "most_current_day_pred_prob", "algoprediction",
                                "exact_time_of_prediction", "current_eth_price_in_dollar", "timeofbuy_longshort_order", "binance_mean_entryprice_eth",
                                "eths_to_set_1_acquired", "eths_to_set_2_acquired", "eths_to_set_3_acquired",
                                "eths_to_set_1_dissolved", "eths_to_set_2_dissolved", "eths_to_set_3_dissolved",
                                "after_bet_short_or_long", "sum_unRealizedProfit_list_of_last_day", "sum_after_bet_money_longshort_eth_list",
                                ])

    print("pred_proba", pred_proba, file=f)
    print("algoprediction", algoprediction, file=f)
    print("current_short_or_long", current_short_or_long, file=f)
    print("money_active", money_active, file=f)
    print("timenow", timenow, file=f)
    print("current_eth_price_in_dollar",
          current_eth_price_in_dollar, file=f)
    print("mean_entryprice_eth_list", mean_entryprice_eth_list, file=f)
    print("after_bet_short_or_long", after_bet_short_or_long, file=f)
    print("sum_unRealizedProfit_list_of_last_day",
          sum_unRealizedProfit_list, file=f)
    print("sum_after_bet_money_longshort_eth_list",
          sum_after_bet_money_longshort_eth_list, file=f)

    sendmail_result(pred_proba,
                    algoprediction, current_short_or_long, money_active,
                    timenow, current_eth_price_in_dollar, mean_entryprice_eth_list,
                    after_bet_short_or_long, sum_after_bet_money_longshort_eth_list,
                    sum_unRealizedProfit_list,
                    eths_to_set_1, eths_to_set_2, eths_to_set_3,
                    sum_money_longshort_eth_1, sum_money_longshort_eth_2, sum_money_longshort_eth_3)

    # # Store DF of the one prediction in database
    engine = create_engine("mysql://root:root@127.0.0.1/eth_predictions")

    predicted_df.to_sql(name="eth_preds",
                        con=engine, index=False,
                        if_exists='append',
                        )

f.close()
