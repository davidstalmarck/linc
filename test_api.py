import hackathon_linc as lh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lh.init("")


orders = lh.get_all_orders()
balance = lh.get_balance()
tickers = lh.get_all_tickers()

historical_data = lh.get_historical_data(100, "STOCK2")

# Convert data to DataFrame
df = pd.DataFrame(historical_data)

# Convert time to datetime
df["gmtTime"] = pd.to_datetime(df["gmtTime"])

# Compute volume difference
df["volumeDifference"] = df["askVolume"] - df["bidVolume"]

# Plot Ask and Bid Prices
fig, ax = plt.subplots(2, 1, figsize=(10, 6))  # Create two subplots stacked vertically
ax[0].plot(df["askMedian"], label="Ask Price", marker="o", linestyle="-")
ax[0].plot(df["bidMedian"], label="Bid Price", marker="o", linestyle="-")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Price")
ax[0].set_title("Ask and Bid Prices Over Time")
ax[0].legend()
ax[0].grid()

# Plot Volume Difference
ax[1].plot(
    df["volumeDifference"],
    label="Ask-Bid Volume Difference",
    marker="s",
    linestyle="-",
    color="red",
)
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Volume Difference")
ax[1].set_title("Difference in Ask and Bid Volumes Over Time")
ax[1].axhline(0, color="black", linestyle="--", linewidth=1)
ax[1].legend()
ax[1].grid()
plt.show()

# Compute volume difference
df["volumeDifference"] = df["askVolume"] - df["bidVolume"]
# Compute average between average of ask and bid
df["avgPrice"] = (df["askMedian"] + df["bidMedian"]) / 2


# Extract the two signals
price = (
    df["avgPrice"] - df["avgPrice"].mean()
)  # Remove mean for better correlation analysis
volume_diff = df["volumeDifference"] - df["volumeDifference"].mean()  # Remove mean
spread = df["spreadMedian"] - df["spreadMedian"].mean()  # Remove mean

# Plot cross-correlation function
plt.figure(figsize=(10, 5))
plt.plot(price, label="Average Price")
plt.plot(10 * spread, label="Spread")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Average Price and Spread Over Time")
plt.show()

# Compute cross-correlation
lags = np.arange(-len(df) + 1, len(df))
cross_corr = np.correlate(spread, price, mode="full")

# Plot cross-correlation function
plt.figure(figsize=(10, 5))
plt.stem(lags, cross_corr)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Lag")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation Between Spread and Average Price")
plt.grid()
plt.show()
