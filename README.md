Project Overview

This project builds a predictive supply chain risk system that forecasts customer backlog levels 7 days into the future and identifies customers at risk of stockout.

Instead of simply predicting numbers, the system converts forecasts into operational decisions by:

Estimating future backlog quantities

Calculating coverage days (backlog relative to demand)

Classifying customers into risk levels (LOW / MEDIUM / HIGH / CRITICAL)

Identifying operational drivers of risk

Recommending corrective actions

The final output is an interactive Streamlit dashboard that functions as a mini supply-chain control tower.

Business Objective

The goal of this project is to answer a key operational question:

“Will this customer’s current backlog sustain them for the next week, and if not, what action should we take?”

The system helps determine:

Whether a customer’s backlog is sufficient

If replenishment orders should be generated

If logistics bottlenecks (e.g., truck availability) are driving risk

Whether depot or plant issues are affecting stability

Modeling Approach
Target Definition

The model predicts future backlog 7 days ahead:

df["future_backlog_qty"] = (
    df.groupby("customer_code")["backlog_qty"]
    .shift(-7)
)

This trains the model to forecast where backlog will be one week from today based on current operational conditions.

Feature Engineering

To incorporate time-awareness, the model uses:

Lag Features

backlog_lag_1

backlog_lag_7

dispatch_lag_1

dispatch_lag_7

Rolling Trend Features

7-day rolling mean of backlog

7-day rolling mean of dispatch

Operational Drivers

Truck availability index

Dispatch delay days

Depot status

Plant status

Available funds

Daily demand target

This allows the model to understand trend, momentum, and operational constraints.

Risk Classification Logic

After predicting future backlog, the system computes:

Cover Days = Predicted Backlog / Daily Target

Risk levels are assigned based on coverage:

LOW → sufficient coverage

MEDIUM → moderate buffer

HIGH → unstable

CRITICAL → imminent stockout

This converts raw forecasts into actionable risk signals.

Driver Identification

The dashboard includes rule-based driver detection such as:

Low truck availability

Depot out of stock

Plant shutdown

High dispatch delays

Each critical case is accompanied by a recommended action.

Streamlit Dashboard Features

Interactive filtering by region, product, date, and risk

Critical customer alert cards

Risk distribution visualizations

Trend analysis over time

Customer severity ranking

Driver breakdown analysis

Downloadable prediction results

The dashboard is designed for operations teams and decision-makers.

Tech Stack

Python

Pandas

XGBoost

Scikit-learn

Streamlit

Plotly

Model Performance

On chronological test split:

R² ≈ 0.95+

RMSE and MAE evaluated on future backlog prediction

The model generalizes well across customers and operational conditions.

Key Insight

This project demonstrates how predictive modeling can be translated into operational intelligence.

It moves beyond regression accuracy and focuses on:

Forward-looking supply stability

Early risk detection

Decision support automation

Access the app using; https://backlogpy.streamlit.app/


Future Improvements

SHAP explainability integration

Order quantity optimization using available funds

LSTM-based sequence modeling comparison

Automated alerting system

Real-time deployment pipeline
