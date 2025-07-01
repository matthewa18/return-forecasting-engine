import streamlit as st
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Quant Strategy Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Machine Learning-Based Equity Return Classifier")
st.markdown("""
This app combines machine learning and financial theory to classify and forecast stock returns based on
idiosyncratic volatility and lagged returns. Long-short strategy performance is evaluated against the market.
""")

# -----------------------------
# NAVIGATION TABS
# -----------------------------
tabs = st.tabs([
    "Strategy Backtests",
    "Model Performance",
    "Feature Insights",
    "3D Return Surface",
    "Project Summary"
])

# -----------------------------
# TAB 1: STRATEGY BACKTESTS
# -----------------------------
with tabs[0]:
    st.header("Strategy Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ML Strategy vs Market")
        try:
            st.image("ML-backtestvisual.png", caption="ML Strategy", use_column_width=True)
        except Exception as e:
            st.error(f"ML-backtestvisual.png not found: {e}")
        st.metric("Sharpe Ratio (ML Strategy)", "2.45")

    with col2:
        st.subheader("Volatility-Only Strategy vs Market")
        try:
            st.image("Volatilityonly-backtestvisual.png", caption="Volatility Strategy", use_column_width=True)
        except Exception as e:
            st.error(f"Volatilityonly-backtestvisual.png not found: {e}")
        st.metric("Sharpe Ratio (Volatility Strategy)", "0.36")

# -----------------------------
# TAB 2: CLASSIFICATION RESULTS
# -----------------------------
with tabs[1]:
    st.header("Model Classification Results")
    st.markdown("Trained a Random Forest classifier on idiosyncratic volatility and 2-month lagged returns.")

    st.subheader("Confusion Matrix")
    try:
        st.image("Final Confusion Matrix.png", caption="Confusion Matrix", use_column_width=True)
    except Exception as e:
        st.error(f"Final Confusion Matrix.png not found: {e}")

    st.subheader("Classification Report")
    st.code("""
              precision    recall  f1-score   support

           0       0.70      0.79      0.74       470
           1       0.43      0.31      0.36       237

    accuracy                           0.63       707
   macro avg       0.56      0.55      0.55       707
weighted avg       0.61      0.63      0.61       707
    """)

# -----------------------------
# TAB 3: FEATURE IMPORTANCE
# -----------------------------
with tabs[2]:
    st.header("Feature Importance")
    st.markdown("Random Forest model trained with:")
    st.code("['Idiosyncratic Volatility', 'LAG_2']")

    st.subheader("Estimated Feature Importances")
    st.bar_chart(pd.Series(
        [0.68, 0.32],
        index=["Idiosyncratic Volatility", "LAG_2"]
    ))

# -----------------------------
# TAB 4: 3D RISK-RETURN SURFACE
# -----------------------------
with tabs[3]:
    st.header("3D Risk-Return Surface Visualization")
    st.markdown("Linear model forecasts vs actual returns across varying volatility levels.")
    try:
        st.image("3D Return Surface Under Varying Idiosyncratic Risk Levels.png", caption="3D Surface", use_column_width=True)
    except Exception as e:
        st.error(f"3D Return Surface image not found: {e}")

# -----------------------------
# TAB 5: PROJECT SUMMARY
# -----------------------------
with tabs[4]:
    st.header("Project Summary")
    st.markdown("""
**Objective:**  
Build a machine learning pipeline to classify next-month returns using volatility and lag features,
then backtest a trading strategy based on ranked signal confidence.

**Models Used:**  
- Linear Regression for baseline forecasting  
- Random Forest Classifier for return class prediction  

**Strategy:**  
- Long top 20% of predicted winners, short bottom 20%  
- Portfolio rebalanced monthly, benchmarked against equal-weight market return  

**Results:**  
- ML strategy outperformed market and baseline in cumulative return and Sharpe ratio  
- Feature importance reveals volatility as primary predictive signal  
- Classification accuracy: 61% vs 50% baseline  

**Built with:**  
Python, Streamlit
    """)
