import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

# Import shared logic here:
from arma_strategies import (
    train_arima,
    basic_long_short_signal,
    threshold_signal,
    magnitude_signal,
)


def load_and_prepare_data(csv_file, stock_name="STOCK1"):
    df = pd.read_csv(csv_file)
    df["gmtTime"] = pd.to_datetime(df["gmtTime"])
    df.sort_values("gmtTime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[df["gmtTime"].dt.year != 2020]

    df["avgPrice"] = (df["askMedian"] + df["bidMedian"]) / 2.0
    df = df[df["symbol"] == stock_name].copy()
    df["returns"] = df["avgPrice"].pct_change()

    return df


def backtest_ar_strategy(
    df,
    p=1,
    q=0,
    window_size=50,
    strategy="basic",
    threshold=0.0005,
    scale_factor=0.001,
    max_shares=50,
):
    """
    Rolling AR backtest. (Simplified snippet.)
    """
    df = df.copy()
    df.reset_index(drop=True, inplace=True)

    initial_capital = 100_000.0
    capital = initial_capital
    position = 0
    last_position_dir = 0
    share_price_yesterday = 0.0

    equity_curve = []
    dates = []

    for t in range(window_size, len(df)):
        train_slice = df.iloc[t - window_size : t].dropna(subset=["returns"])
        if len(train_slice) < 2:
            equity_curve.append(capital)
            dates.append(df["gmtTime"].iloc[t])
            continue

        # Instead of duplicating the code, we call train_arima from our shared module
        model = train_arima(train_slice, p, q, stock_name=df["symbol"].iloc[t])

        forecast = model.forecast(steps=1).iloc[0]
        current_price = df["avgPrice"].iloc[t]

        # Update capital if we had a position from last time
        if t > 0 and share_price_yesterday > 0:
            daily_pnl = position * (current_price - share_price_yesterday)
            capital += daily_pnl

        # Decide signal based on strategy
        if strategy == "basic":
            signal = basic_long_short_signal(forecast, last_position_dir)
            if signal == "BUY":
                position = 10
                last_position_dir = 1
            elif signal == "SELL":
                position = -10
                last_position_dir = -1
        elif strategy == "threshold":
            signal = threshold_signal(forecast, last_position_dir, threshold)
            if signal == "BUY":
                position = 10
                last_position_dir = 1
            elif signal == "SELL":
                position = -10
                last_position_dir = -1
        elif strategy == "magnitude":
            pos_target = magnitude_signal(forecast, scale_factor, max_shares)
            position = pos_target
            last_position_dir = np.sign(position)

        share_price_yesterday = current_price

        equity_curve.append(capital)
        dates.append(df["gmtTime"].iloc[t])

    result_df = pd.DataFrame({"gmtTime": dates, "equity": equity_curve})
    return result_df


def evaluate_performance(equity_df):
    """
    Takes a DataFrame with 'gmtTime' and 'equity' columns,
    returns final stats like total return, daily Sharpe, etc.
    """
    # Calculate daily returns from equity
    equity_df["eq_returns"] = equity_df["equity"].pct_change().fillna(0.0)
    total_return = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0]) - 1

    # Simple daily Sharpe ratio = mean(returns)/std(returns)
    daily_ret = equity_df["eq_returns"]
    if daily_ret.std() != 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(24 * 365)  # if daily data
    else:
        sharpe = 0

    return {
        "Final Equity": equity_df["equity"].iloc[-1],
        "Total Return": total_return,
        "Sharpe": sharpe,
    }


def main():
    csv_file = "datasets/stockPrices_hourly.csv"
    stock_name = "STOCK1"

    df = load_and_prepare_data(csv_file, stock_name=stock_name)

    # We'll run each strategy, then evaluate the final equity curve
    strategies = ["basic", "threshold", "magnitude"]
    results = {}

    for strat in strategies:
        print(f"Backtesting strategy={strat}...")
        eq_df = backtest_ar_strategy(
            df,
            p=1,
            q=0,
            window_size=50,
            strategy=strat,
            threshold=0.0005,  # for threshold
            scale_factor=0.001,  # for magnitude
            max_shares=50,
        )
        perf = evaluate_performance(eq_df)
        results[strat] = perf

        print(
            f"Strategy={strat} -> Final Equity: {perf['Final Equity']:.2f}, "
            f"Total Return: {perf['Total Return']:.2%}, "
            f"Sharpe: {perf['Sharpe']:.2f}"
        )

    print("\nComparison of Strategies:")
    for strat, perf in results.items():
        print(f"{strat:10} -> {perf}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
