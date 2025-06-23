import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
idio_vol = pd.read_csv("1stidiosyncratic_volatility.csv")
monthly_returns = pd.read_csv("1stmonthly_returns.csv")

merged_df = pd.merge(idio_vol, monthly_returns, on=["PERMNO", "Month"], how="inner")
merged_df["Month"] = pd.to_datetime(merged_df["Month"])
merged_df["RET"] = merged_df["RET_x"]
merged_df["RET_lead"] = merged_df.groupby("PERMNO")["RET"].shift(-1)
merged_df["LAG_2"] = merged_df.groupby("PERMNO")["RET"].shift(2)

# Define binary return class
q50 = merged_df["RET_lead"].quantile(0.5)
merged_df["Return_Class"] = (merged_df["RET_lead"] > q50).astype(int)

# Drop missing data
merged_df = merged_df.dropna(subset=["Idiosyncratic Volatility", "LAG_2", "Return_Class", "RET_lead"])

# Define features and target
X = merged_df[["Idiosyncratic Volatility", "LAG_2"]]
y = merged_df["Return_Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict class probabilities
merged_df["ML_Score"] = model.predict_proba(X)[:, 1]  # Prob of class 1

# Backtest logic
returns = []
for month in sorted(merged_df["Month"].unique()):
    data = merged_df[merged_df["Month"] == month].copy()
    if data.empty:
        continue
    data = data.sort_values("ML_Score", ascending=False)
    data["Rank"] = data["ML_Score"].rank(ascending=False)

    top = data[data["Rank"] <= data["Rank"].quantile(0.2)]
    bottom = data[data["Rank"] >= data["Rank"].quantile(0.8)]

    strat_ret = top["RET_lead"].mean() - bottom["RET_lead"].mean()
    market_ret = data["RET_lead"].mean()

    returns.append({"Month": month, "StrategyReturn": strat_ret, "MarketReturn": market_ret})

# Compile backtest results
bt_df = pd.DataFrame(returns)
bt_df["Cumulative_Strategy"] = (1 + bt_df["StrategyReturn"]).cumprod()
bt_df["Cumulative_Market"] = (1 + bt_df["MarketReturn"]).cumprod()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(bt_df["Month"], bt_df["Cumulative_Strategy"], label="ML Strategy")
plt.plot(bt_df["Month"], bt_df["Cumulative_Market"], label="Market")
plt.title("Cumulative Returns: ML Model vs Market")
plt.xlabel("Month")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
