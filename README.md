# Volatility-Based Return Forecasting & Classification Engine

This project uses Python to forecast and classify next-month stock returns based on idiosyncratic volatility, momentum, and lagged returns. It starts with a linear regression to estimate the relationship between volatility and forward returns (β ≈ –0.08), followed by a supervised classification model using XGBoost to label stocks as "buy" or "don't buy" relative to return thresholds.

The classifier achieves ~49% accuracy. Evaluation includes classification metrics (precision, recall, confusion matrix) and return performance analysis of model-selected signals. A 3D risk-return surface visualizes the relationship between volatility, forecasted return, and realized return.

### Key Features:
- Custom-labeled return classes from time-shifted returns
- Feature engineering: 12-month momentum (excl. t-1), 2-month lagged returns
- Linear regression modeling and XGBoost classification
- 3D visualization of volatility and return structure
- Full evaluation of model accuracy and predictive performance

**Author:** Matthew Aldridge — Barrett, The Honors College, ASU

