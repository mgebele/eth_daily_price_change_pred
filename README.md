# Eth daily price change prediction 
> this project has the goal of predicting the rise or fall of the price of the cryptocurrency eth in the next 24 hours based on different metrics.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Project Organization](#project-organization)

## General Information
- The raw data used for the machine learning (ml) model consists of onchain- and exchange-metrics from Glassnode for the by market cap two biggest cryptocurrencies btc and eth. Furthermore, there has been experimented with including general financial data (currency rates, commodities, stock indices) from the yfinance api and sentiment data from twitter.
In the feature engineering step, the most useful features have been selected as well as calculated to new features to improve the predictive power of eth-price change.
These steps resulted in different combinations represented by different datasets, which were tested on different ml-models including hyperparameter tuning on the two most promising models xgboost and random forest.
Next to the basic model performance like precision and accuracy of this binary classification target there has been implemented a performance metric calculating the change in price for every daily prediction of a rise or fall in the eth-price. 
If the model predicted a rise in price for the next 24 hours, followed by a real increase of the eth-price of f. I. 5 % would deliver 5 $, assuming we take a bet of 100 $, every day the model has a sufficient prediction probability (0-100%).
The contrary case of a followed eth-price change of -5 % would result in a loss of 5 $. This performance metric has also been compared to the performance of a simple buy and hold strategy for eth. 
The data has been trained on the years 2015 until 2019 and tested on the year 2020. 
Lastly the project consists of a trading bot which was deployed on an external server getting daily at 00:00 UTC the newest data, running the model-prediction and based on the result executing long buy and sell orders on the eth-perp asset via the Binance-api. On every execution, the information about the daily model prediction has been sent out via email. 

## Technologies Used
- Python 3 in VSCode with the Juypter extension  

## Features
most important features after the random forest feature selection method:
- addresses/new_non_zero_count
- addresses/active_count
- addresses/sending_count
- addresses/receiving_count
- addresses/count
- addresses/non_zero_count
- addresses/min_point_zero_1_count
- addresses/min_point_1_count
- addresses/min_1_count
- addresses/min_10_count
- addresses/min_100_count
- addresses/min_1k_count
- addresses/min_10k_count
- addresses/min_32_count
- blockchain/block_height
- blockchain/block_count
- blockchain/block_interval_mean
- blockchain/block_interval_median
- blockchain/block_size_sum
- blockchain/block_size_mean
- fees/volume_sum
- fees/volume_mean
- fees/volume_median
- fees/gas_used_sum
- fees/gas_used_mean
- fees/gas_used_median
- fees/gas_price_mean
- fees/gas_price_median
- fees/gas_limit_tx_mean
- fees/gas_limit_tx_median
- fees/fee_ratio_multiple
- Closed Price USD
- market/marketcap_realized_usd
- market/mvrv
- market/price_drawdown_relative
- market/marketcap_usd
- market/price_usd_close_cummax
- transactions/count
- transactions/rate
- transactions/transfers_count
- transactions/transfers_rate
- transactions/transfers_volume_sum
- transactions/transfers_volume_mean
- transactions/transfers_volume_median
- transactions/transfers_volume_to_exchanges_sum
- transactions/transfers_volume_from_exchanges_sum
- transactions/transfers_volume_exchanges_net
- transactions/transfers_to_exchanges_count
- transactions/transfers_from_exchanges_count
- transactions/transfers_volume_to_exchanges_mean
- transactions/transfers_volume_from_exchanges_mean
- transactions/contract_calls_internal_count
- distribution/balance_1pct_holders
- distribution/gini
- distribution/herfindahl
- distribution/supply_contracts
- distribution/balance_exchanges
- indicators/sopr
- indicators/net_unrealized_profit_loss
- indicators/unrealized_profit
- indicators/unrealized_loss
- indicators/cdd
- indicators/liveliness
- indicators/average_dormancy
- indicators/asol
- indicators/msol
- indicators/nvt
- indicators/nvts
- indicators/velocity
- mining/difficulty_latest
- mining/hash_rate_mean
- mining/thermocap
- mining/marketcap_thermocap_ratio
- mining/revenue_sum
- mining/revenue_from_fees
- supply/current
- supply/profit_relative
- supply/profit_sum
- supply/loss_sum
- supply/active_24h
- supply/active_1d_1w
- supply/active_1w_1m
- supply/active_1m_3m
- supply/active_3m_6m
- supply/active_6m_12m
- supply/active_1y_2y
- supply/active_2y_3y
- supply/active_3y_5y
- supply/active_5y_7y
- supply/active_7y_10y
- supply/active_more_10y
- supply/issued
- supply/inflation_rate
- Closed Price USD
- Daily Return in Percent
- Daily Log Return in Percent
- Log Price in USD
- percentage_daily_return_bef_shift
- price_usd_close_percent_of_maxtilnow

## Setup
For running these project python3 needs to be installed as well as the packages specified in the requieremts.txt. Lastly Jupyter Notebook or Jupyter Lab can be used to inspect and execute the code. The most current version of the model is the "dailydata_featureengineering_xgboost_ethbtcfinance.ipynb" located in "Pr BTC\ml_model\final_models\". This runs on execution with the btc and eth onchain- and exchange-data.

## Room for Improvement
Room for improvement and To-Dos:
- Testing of different kind of ml-models like deep neural nets, reinforcement learning and models using evolutionary-algorithms
- Changing the target from eth-price-change to other cryptocurrency coins for which the model performance increases. 
- Including more statistical analyses based on price moving patterns. 
- Changing prediction to higher or lower timeframe. Also hourly price changes have been tested, but there are less features for it.
- include google trends data 
- analyze data from specific twitter user.
- adding of metrics which better reflect the overall state of the world economy.
- include stock price changes of companies which have influence on the btc&eth-price due to big amount of btc&eth-holdings/interest.   
- ...

## Project Organization
│<br />
├── README.md                               <- The top-level README for developers using the project.<br />
│<br />
├── requirements.txt                        <- Containing the python packages incl. versions used within this project.<br />
│<br />
├── automated_trading_bot                   <- Includes the code and the results of the deployed code which predicts daily based on the new data<br />
│<br />
├── calculated_datasets                     <- The datasets calculated from the ipynbs in ml_model -> dataset_generation <br />
│<br />
├── data                                    <- Raw data from glassnode and yahoofinance used for calculating the datasets<br />
│<br />
└── scripts                                 <- Python scripts used within this project.<br />
&nbsp;&nbsp;&nbsp;│<br />
&nbsp;&nbsp;&nbsp;├── dataset_generation                  <- Scripts to generate the "calculated dataset". <br />
&nbsp;&nbsp;&nbsp;│<br />
&nbsp;&nbsp;&nbsp;├── final_models                        <- Ipynbs which contain the best and most current versions of the ml-model to predict the target most accurately.<br />
&nbsp;&nbsp;&nbsp;│       <br />
&nbsp;&nbsp;&nbsp;├── model_development                   <- Scripts used within the development of the final model in combination with the best features.<br />
&nbsp;&nbsp;&nbsp;│<br />
&nbsp;&nbsp;&nbsp;└── model_output                        <- Xlsx files containing the performance of different models, model parameters and features. <br />

