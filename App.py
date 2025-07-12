import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from PIL import Image
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="ClassifierX", layout="wide")

# --- GLOBAL LIGHT STYLING ---
st.markdown("""
    <style>
        html, body {
            background-color: #ffffff !important;
            color: #111111 !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
        }
        .main, .block-container {
            background-color: #ffffff !important;
            color: #111111 !important;
            font-size: 16px;
        }
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: #111111 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- PLOTLY STYLE FIX ---
pio.templates.default = "plotly_white"

# --- HEADER ---
st.title("Machine Learning-Based Equity Return Classifier")
st.markdown("""
This application forecasts equity returns using a supervised learning model based on idiosyncratic volatility and historical returns.
A simulated trading strategy is backtested to compare performance against the market.
""")

# --- EXPANDABLE INTRO ---
with st.expander("What This App Does"):
    st.markdown("""
    - Trains a model to predict whether a stock will outperform or underperform next month.
    - Uses volatility and return data to classify risk-adjusted opportunities.
    - Constructs a long-short portfolio: buys top-ranked predictions, sells bottom-ranked.
    - Tracks portfolio performance versus market returns over time.
    """)

# --- TABS ---
tabs = st.tabs([
    "Strategy Backtest",
    "Model Performance",
    "Feature Insights",
    "3D Return Surface",
    "Quant for Dummies",
    "Project Summary"
])

# ========== STRATEGY BACKTEST ==========
with tabs[0]:
    st.header("Strategy Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ML Strategy vs Market")
        img1 = Image.open("ML-backtestvisual.png")
        st.image(img1, use_container_width=True)
        st.metric(label="Sharpe Ratio (ML Strategy)", value="2.45")

    with col2:
        st.subheader("Volatility-Only Strategy vs Market")
        img2 = Image.open("Volatilityonly-backtestvisual.png")
        st.image(img2, use_container_width=True)
        st.metric(label="Sharpe Ratio (Volatility Strategy)", value="0.36")

# ========== MODEL PERFORMANCE ==========
with tabs[1]:
    st.header("Model Classification Results")
    st.markdown("""
    This section shows how well the model predicted next-month return direction, whether a stock would outperform (label = 1) or underperform (label = 0).

    The **confusion matrix** below compares the model's predictions to actual outcomes:

    - Rows = actual return class  
    - Columns = predicted return class  
    - Top-left = correctly predicted underperformers  
    - Bottom-right = correctly predicted outperformers  
    - Off-diagonals = misclassifications  

    Higher values on the diagonal indicate stronger model performance.
    """)

    st.subheader("Confusion Matrix")
    img_cm = Image.open("Final Confusion Matrix.png")
    st.image(img_cm)

    st.subheader("Classification Report")
    img_report = Image.open("Classification Report Results.png")
    st.image(img_report, use_container_width=True)

# ========== FEATURE INSIGHTS ==========
with tabs[2]:
    st.header("Feature Importance")
    st.markdown("Random Forest model was trained using the following features:")

    st.markdown("""
    <pre style="background-color:#f8f9fa; color:#111111; padding:10px; border-radius:5px;
                font-family:'Courier New', monospace; font-size:14px; white-space: pre; overflow-x: auto;">
['Idiosyncratic Volatility', 'LAG_2']
    </pre>
    """, unsafe_allow_html=True)

    importances = [0.67, 0.33]
    features = ["Idiosyncratic Volatility", "LAG_2"]

    fig = px.bar(
        x=features,
        y=importances,
        labels={"x": "Feature", "y": "Importance"},
        title="Estimated Feature Importances",
        text=[f"{x:.2f}" for x in importances]
    )

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', size=14),
        title_font=dict(color='black'),
        xaxis=dict(
            title_font=dict(color='black'),
            tickfont=dict(color='black'),
            color='black'
        ),
        yaxis=dict(
            title_font=dict(color='black'),
            tickfont=dict(color='black'),
            color='black'
        ),
        legend=dict(font=dict(color='black'))
    )

    st.plotly_chart(fig, use_container_width=True)

# ========== 3D RETURN SURFACE ==========
with tabs[3]:
    st.header("3D Return Surface")
    st.markdown("Visual representation of actual and forecasted returns as a function of idiosyncratic risk.")

    try:
        img3d = Image.open("3D Return Surface Under Varying Idiosyncratic Risk Levels.png")
        st.image(img3d, use_container_width=True)
    except:
        st.error("Error: 3D Return Surface image not found. Please check file name and location.")

# ========== QUANT FOR DUMMIES ==========
with tabs[4]:
    st.header("Quant for Dummies")
    st.markdown("""

    **1. What's the goal?**  
    To predict whether a stock will perform better or worse than average next month.

    **2. What does the model look at?**  
    Two things:
    - How volatile (risky) a stock is
    - How it performed in the recent past

    **3. What kind of model is this?**  
    It’s a type of **machine learning classifier**. Think of it like a pattern detector:  
    - It looks at thousands of past examples  
    - It learns which combinations of volatility and returns tend to lead to gains or losses  
    - It doesn’t memorize, rather it generalizes, like a smart assistant that gets better with experience

    **4. What happens after it makes predictions?**  
    - The model ranks stocks from most likely to outperform to most likely to underperform  
    - Then we simulate a strategy: buy the top 20%, short the bottom 20%, and hold for one month

    **5. Why should we care?**  
    If this model consistently outperforms the market, even slightly, that signal could be turned into a real investment strategy.
    """)

# ========== PROJECT SUMMARY ==========
with tabs[5]:
    st.header("Project Summary")
    st.markdown("""
    **Objective**  
    Build a machine learning pipeline to classify next-month stock returns based on volatility and lagged return data.

    **Model**  
    - Linear Regression for baseline forecasting  
    - Random Forest Classifier for directional prediction

    **Strategy**  
    - Long top 20% of predicted winners  
    - Short bottom 20%  
    - Monthly rebalancing  
    - Benchmarked against equal-weighted market return

    **Results**  
    - ML strategy outperformed baseline in cumulative return  
    - Sharpe Ratio: 2.45  
    - Classification accuracy: 61%  
    - Volatility shown as primary predictive signal

    **Technologies**  
    Python, Streamlit, Scikit-learn, Plotly, Matplotlib
    """)

