# ML-Based Equity Return Classifier
By: Matthew Aldridge | Honors Mathematics & Finance @ ASU

This project builds a model to classify the directional performance of individual stocks on a one-month forward horizon. It uses idiosyncratic volatility and lagged returns as predictors. The process begins with a regression analysis to estimate the relationship between volatility and future return, followed by a binary classifier that labels stocks as “buy” or “don’t buy” based on relative forward performance.

The classifier achieves 61% accuracy on a balanced dataset (50/50 class distribution), reflecting an 11% lift over baseline. Model output is used to construct a long-short equity strategy, which is evaluated using Sharpe ratio and cumulative return.

**Live App:** [Streamlit Deployment](https://matthewa18-return-forecasting-engine-app-v7agn0.streamlit.app/)

---

## Methodology

### Data and Feature Engineering
- Monthly returns per stock 
- 30-day rolling idiosyncratic volatility 
- Forward 1-month return, shifted to serve as prediction target
- Binary target defined using the median forward return per month

### Regression Step
- Linear regression of forward return on idiosyncratic volatility
- Beta coefficient ≈ –0.07 
- Used for both interpretation and feature augmentation

### Classification Step
- Random Forest classifier (scikit-learn)
- Features:
  - 2-month lagged return
  - Idiosyncratic volatility
- Target: outperform (1) vs. underperform (0) relative to cross-sectional median

### Model Evaluation
- Accuracy: 61% (vs. 50% baseline)
- Additional metrics: precision, recall, F1 score, confusion matrix

---

## Strategy Backtest

- Rank stocks by classifier output each month
- Long top 20%, short bottom 20%, equally weighted
- Monthly rebalancing over full period
- Sharpe ratio: 2.45 (strategy vs. market)

---

## Visualizations

- Confusion matrix (classification performance)
- Feature importance (Random Forest output)
- 3D surface: forecasted return vs. volatility vs. realized return

---

## Deployment

- Web interface built using Streamlit
- Application includes model outputs, performance visualizations, and backtest results
- Fully interactive and publicly hosted

---

## Tools and Libraries

- Python 3  
- pandas, numpy  
- scikit-learn  
- matplotlib, plotly  
- Streamlit  
- Git



