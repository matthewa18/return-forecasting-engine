# Volatility-Based Return Forecasting & Classification Engine

This project implements a modeling pipeline to analyze how stock-level risk characteristics relate to future returns. The current version estimates the relationship between idiosyncratic volatility and next-month returns using linear regression, and applies supervised classification to segment return outcomes. The long-term goal is to develop a flexible return engine that supports signal-based ranking and evaluation across multiple financial factors.

## Contents

- `process_dsf.py` — core script for loading, forecasting, and modeling
- `1stmonthly_returns.csv` — monthly returns by security (PERMNO)
- `1stidiosyncratic_volatility.csv` — 30-day rolling return volatility (monthly average)
- `3D Return Surface Under Varying Idiosyncratic Risk Levels.png` — visualization of forecast vs. volatility
- `forecasted_returns.csv` — predicted next-month returns
- `classification_report.txt` — model evaluation metrics
- `confusion_matrix.png` — visual performance summary

## Methodology

1. Aggregate daily return data to monthly frequency per security
2. Compute idiosyncratic volatility from rolling standard deviation of returns
3. Merge volatility and return datasets by `PERMNO` and `Month`
4. Define `RET_lead` as next-month return via time-shift
5. Fit linear regression:
6. Generate forecasts and label future returns by percentile thresholding
7. Train a Random Forest classifier on signal inputs
8. Evaluate model via classification metrics and confusion matrix

## Interpretation

The forecast surface visualizes the relationship between:
- Idiosyncratic volatility
- Realized return
- Forecasted return

Initial results indicate a negative relationship between specific risk and expected return. Classification performance is strongest in the neutral class, with weak but consistent structure observed in the tails.

## Long-Term Objective

This repository is the first step toward a modular return classification engine that will:

- Integrate multiple signals (e.g., momentum, size, beta)
- Use gradient boosting and probability-based ranking
- Score securities within return buckets and test performance spreads
- Support out-of-sample evaluation and portfolio grouping logic

## Future Work

- Add support for multi-factor feature sets
- Implement XGBoost with probability calibration
- Rank securities by predicted return class probability
- Evaluate grouped return spreads based on model output

---

Author: Matthew Aldridge  
Barrett, The Honors College — Arizona State University
