import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates

# --- Load data ---
idio_vol = pd.read_csv("1stidiosyncratic_volatility.csv")
monthly_returns = pd.read_csv("1stmonthly_returns.csv")

merged_df = pd.merge(idio_vol, monthly_returns, on=["PERMNO", "Month"], how="inner")
merged_df["Month"] = pd.to_datetime(merged_df["Month"])
merged_df["RET"] = merged_df["RET_x"]
merged_df["RET_lead"] = merged_df.groupby("PERMNO")["RET"].shift(-1)
merged_df = merged_df.dropna(subset=["Idiosyncratic Volatility", "RET_lead"])

# --- Backtest: Rolling Linear Regression ---
results = []
months = sorted(merged_df["Month"].unique())
window = 36

for i in range(window, len(months) - 1):
    train_months = months[i - window:i]
    test_month = months[i + 1]

    train = merged_df[merged_df["Month"].isin(train_months)]
    test = merged_df[merged_df["Month"] == test_month].copy()
    train = train.dropna(subset=["Idiosyncratic Volatility", "RET_lead"])

    if train.empty or test.empty:
        continue

    model = LinearRegression()
    model.fit(train[["Idiosyncratic Volatility"]], train["RET_lead"])

    test["Forecast"] = model.predict(test[["Idiosyncratic Volatility"]])
    test["Rank"] = test["Forecast"].rank(ascending=False)

    top = test[test["Rank"] <= test["Rank"].quantile(0.2)]
    bottom = test[test["Rank"] >= test["Rank"].quantile(0.8)]

    strat_return = top["RET_lead"].mean() - bottom["RET_lead"].mean()
    market_return = test["RET_lead"].mean()

    results.append({
        "Month": test_month,
        "StrategyReturn": strat_return,
        "MarketReturn": market_return
    })

bt_vol = pd.DataFrame(results)
bt_vol = bt_vol[bt_vol["Month"] >= pd.to_datetime("1989-01-01")]
bt_vol["Cumulative_Strategy"] = (1 + bt_vol["StrategyReturn"]).cumprod()
bt_vol["Cumulative_Market"] = (1 + bt_vol["MarketReturn"]).cumprod()

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(bt_vol["Month"], bt_vol["Cumulative_Strategy"], label="Volatility-Only Strategy")
plt.plot(bt_vol["Month"], bt_vol["Cumulative_Market"], label="Market", linestyle="--")
plt.title("Cumulative Returns: Volatility Model vs Market")
plt.xlabel("Month")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)

# Set x-axis to 1-year ticks
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.show()

