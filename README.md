# Ethereum Daily Price Change Prediction

This project aims to predict the daily price movement (rise or fall) of Ethereum (ETH) using a variety of metrics from on-chain data, exchanges, and broader financial indicators.

## Table of Contents
- [General Info](#general-info)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Project Status](#project-status)
- [Room for Improvement](#room-for-improvement)
- [Disclaimer](#disclaimer)
- [Project Organization](#project-organization)

## General Info
This project leverages machine learning to predict daily price changes of Ethereum (ETH) by analyzing on-chain and exchange metrics from Glassnode for the two largest cryptocurrencies by market cap, Bitcoin (BTC) and Ethereum (ETH). The analysis is enriched with general financial data including currency rates, commodities, and stock indices from the Yahoo Finance API, as well as sentiment data sourced from Twitter.

During the feature engineering phase, key indicators were identified and transformed into predictive signals to enhance the model's ability to forecast ETH price movements. Various combinations of these features were tested across different datasets using machine learning models such as XGBoost and Random Forest, with comprehensive hyperparameter tuning applied to optimize performance.

A specialized performance metric was developed to quantify the financial impact of each prediction. This metric assesses the outcome of predicting a price increase or decrease over the next 24 hours. For example, if a prediction of a 5% price increase comes true, and assuming a hypothetical bet of $100, a gain of $5 is realized. Conversely, a false prediction leading to a 5% decrease would result in a $5 loss. This approach allows us to directly compare the predictive model's performance with that of a simple buy-and-hold strategy.

The model was trained using data from three years and tested on the fourth year to validate its effectiveness over different market conditions. Furthermore, the project includes a trading bot deployed on an external server. This bot retrieves the latest data daily at 00:00 UTC, runs the prediction model, and executes trading orders (long or short) on the ETH perpetual futures market via the Binance API based on the prediction outcomes. Each trade's details and prediction results are communicated via automated daily emails.

## Features
Selected features from Glassnode and additional computed metrics include:
- Blockchain activity: block height, block count, transaction rates, fees
- Address metrics: active, sending, and receiving counts, transfer volumes
- Financial indicators: MVRV, SOPR, NVT
- Market data: realized and current market cap, price drawdowns
- Mining data: hash rate, mining difficulty, thermocap

## Setup
To run this project:
1. Ensure Python 3 is installed.
2. Install required packages from `requirements.txt`.
3. Use Jupyter Notebook to run the model notebooks.

## Usage
- The primary notebook `dailydata_featureengineering_xgboost_ethbtcfinance.ipynb` should be run to perform the latest predictions.
- Automated scripts in the `scripts` folder facilitate data handling and model training.




## Results

The following table compares the implemented performance metric against a simple buy-and-hold strategy, showcasing the annual rewards:

| Year | Buy-and-Hold Total Return | Buy-and-Hold Daily Return | Model Total Return | Model Daily Return |
|------|---------------------------|---------------------------|--------------------|--------------------|
| 2017 | +8942%                    | +24.50%                   | +208%              | +0.57%             |
| 2018 | -83%                      | -0.23%                    | +183%              | +0.50%             |
| 2019 | -8%                       | -0.02%                    | +95%               | +0.26%             |
| 2020 | +564%                     | +1.55%                    | +168%              | +0.46%             |

These results highlight the more balanced and consistently positive performance of the model across all four years, despite the model performing worse than the buy-and-hold strategy in 2020. Notably, in 2018, despite a significant drop in ETH price, the model's short buy orders effectively capitalized on market conditions, demonstrating the utility of active trading strategies during downturns.

## Project Status
The project is currently in a stable phase with ongoing tests for enhancements in model accuracy and feature engineering.

## Room for Improvement
Future updates may include:
- Expanding the model to other cryptocurrencies.
- Integrating additional economic indicators and data sources like Google Trends.
- Testing alternative ML approaches like neural networks.## Disclaimer

## Disclaimer
This project is for educational and developmental purposes only. The algorithms, strategies, and analyses provided herein are not recommendations or endorsements for specific investments or trading strategies. The data and predictions generated by this project should not be interpreted as financial advice or a guide to making financial decisions. The accuracy of the models and data interpretation cannot be guaranteed. Users of this information should understand that they are using any and all predictions at their own risk. Neither the creators nor contributors of this project are responsible for any financial losses or damages resulting from the use of this project.

## Project Organization
```plaintext
├── README.md
├── requirements.txt
├── automated_trading_bot
├── calculated_datasets
├── data
└── scripts
    ├── dataset_generation
    ├── final_models
    ├── model_development
    └── model_output
