import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtest_strategy import backtest_strategy, plot_results
from gold_cross import GoldenCrossStrategy

def test_gold_cross(stock_df, stock_name, initial_cash=100000):
    """Test the Golden Cross strategy for a single stock and return percentage change."""
    stock_df = stock_df.copy()  # Avoid modifying the original DataFrame
    trade_signal = GoldenCrossStrategy(stock_df, short_window=20, long_window=50).run()  # Run strategy

    stock_df, final_balance = backtest_strategy(trade_signal, "signal")  # Backtest on signal

    # Calculate percentage change from the initial capital
    percent_change = ((final_balance - initial_cash) / initial_cash) * 100

    print(f"{stock_name} - Golden Cross Strategy Percentage Change: {percent_change:.2f}%")
    plot_results(stock_df, title=f"Golden Cross Strategy - {stock_name}")

    return percent_change  # Return percentage change for aggregation

if __name__ == "__main__":
    # Load the historical stock data
    df = pd.read_csv("datasets/stockPrices_hourly.csv", skipinitialspace=True)

    df.rename(columns={
    'askMedian': 'ask_price',
    'bidMedian': 'bid_price',
    'askVolume': 'ask_volume',
    'bidVolume': 'bid_volume'
    }, inplace=True)

    # Convert relevant columns to float/int
    df['bid_price'] = df['bid_price'].astype(float)
    df['ask_price'] = df['ask_price'].astype(float)
    df['bid_volume'] = df['bid_volume'].astype(int)
    df['ask_volume'] = df['ask_volume'].astype(int)

    # Store percentage changes for all stocks
    percentage_changes = []

    for stock, stock_df in df.groupby("symbol"):  # Process each stock separately
        print(f"\nTesting Golden Cross Strategy for: {stock}")
        percent_change = test_gold_cross(stock_df, stock)
        percentage_changes.append(percent_change)

    # Compute and print the average percentage change across all stocks
    avg_percent_change = sum(percentage_changes) / len(percentage_changes)
    print(f"\nAverage Percentage Change Across All Stocks: {avg_percent_change:.2f}%")
