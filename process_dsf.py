import pandas as pd

# Define the file path
file_path = "dsf.csv"

# Read only 5 rows to test if the file loads
df_sample = pd.read_csv(file_path, nrows=3600)
print(df_sample.info())
print(df_sample.describe())

# Convert DATE column to datetime and format it with dashes
df_sample["Date"] = pd.to_datetime(df_sample["DATE"], format="%Y%m%d")

df_sample.drop(columns=["DATE"], inplace=True)

df_sample = df_sample.sort_values(by=["PERMNO", "Date"])

df_sample["RET"] = pd.to_numeric(df_sample["RET"], errors="coerce")

df_sample["Month"] = df_sample["Date"].dt.to_period('M')

df_sample["Idiosyncratic Volatility"] = df_sample.groupby("PERMNO")["RET"].rolling(window=30).std().reset_index(0, drop=True)

monthly_returns = df_sample.groupby(["PERMNO", "Month"])["RET"].mean().reset_index()

print(df_sample)

print(monthly_returns)

df_sample.to_csv("1stidiosyncratic_volatility.csv", index=False)
monthly_returns.to_csv("1stmonthly_returns.csv", index=False)
