# Volatility-Based Return Forecasting & Classification Engine

This project is the foundation of a return classification engine designed to analyze how stock-level risk characteristics relate to future returns. The initial implementation models the relationship between idiosyncratic volatility and next-month returns using linear regression. The long-term objective is to expand this into a fully modular engine capable of return classification and signal-based ranking across multiple financial factors.

## Contents

- `process_dsf.py` – Core modeling script for reading, merging, forecasting, and visualizing
- `1stmonthly_returns.csv` – Monthly average returns by security (PERMNO)
- `1stidiosyncratic_volatility.csv` – Monthly average of rolling 30-day return volatility (idiosyncratic risk proxy)
- `3D Return Surface Under Varying Idiosyncratic Risk Levels.png` – Visual output of actual vs. forecasted return over varying volatility

## Methodology

1. Daily return data is aggregated into monthly average returns per security.
2. Idiosyncratic volatility is computed as a 30-day rolling standard deviation of returns, then monthly-averaged.
3. Datasets are merged on security ID (`PERMNO`) and `Month`.
4. The model targets `RET_lead`, the next-month return, by shifting historical return data.
5. A linear regression is performed:  
   `RET_lead ~ Idiosyncratic Volatility`
6. Forecasted returns are generated, and a 3D surface plot is created to visualize the model’s behavior across risk conditions.

## Interpretation

The 3D visualization shows the relationship between:
- Idiosyncratic volatility (risk input)
- Actual current returns
- Forecasted next-month returns

Early results suggest a negative relationship: higher specific risk is associated with lower expected return under this specification.

## Long-Term Objective

This repository is the first step toward a complete return classification engine that will:

- Integrate multi-factor inputs (e.g., beta, momentum, size)
- Use machine learning models to classify securities by return potential
- Score and rank securities within decile bands or performance buckets
- Support model validation, out-of-sample testing, and backtesting integration

## Future Work

- Add support for multiple predictors
- Implement classification algorithms (logistic regression, tree-based models)
- Quantify predictive accuracy on holdout sets
- Add labeling logic for return class targets (e.g., overperform/neutral/underperform)
- Introduce a feature pipeline for real-time scoring

## Author

Matthew Aldridge  
Barrett, The Honors College  
Arizona State University
