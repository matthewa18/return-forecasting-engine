# Volatility-Based Return Forecasting & Classification Engine

This project implements a return forecasting and classification pipeline using idiosyncratic volatility and lagged returns to predict the direction of next-month stock returns. The workflow begins with a linear regression to estimate the relationship between volatility and forward returns (β ≈ –0.07), followed by a binary classifier trained using scikit-learn to label stocks as "buy" or "don’t buy" relative to the median return.

The final classifier achieves 61% accuracy on a 50/50 split, providing ~11% absolute lift over baseline. Evaluation includes precision, recall, confusion matrix, and directional classification metrics. A 3D plot is used to visualize the relationship between realized return, volatility, and model forecast.

---

### Key Features
- Label construction via time-shifted forward returns (above/below median)
- Feature engineering:
  - 2-month lagged return
  - Idiosyncratic volatility (30-day rolling std dev)
  - Linear regression-based return forecast
- Binary classification using `RandomForestClassifier` (scikit-learn)
- 3D visualization of return-volatility interaction
- Classification evaluation with standard performance metrics

---

**Author:** Matthew Aldridge  
**Institution:** Barrett, The Honors College, ASU

