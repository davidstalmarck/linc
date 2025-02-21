from matplotlib import pyplot as plt


def backtest_strategy(df, signal_col):
    """
    df: DataFrame with a 'Close' column and a strategy signal in 'signal_col'.
    signal_col: name of the column containing the trade signal.

    We assume:
      1) signal = +1 (long), -1 (short), or 0 (no position)
      2) Price changes from day t to day t+1 are realized if we had the position at day t.
    """
    # Daily returns
    df["Return"] = df["Close"].pct_change()

    # Strategy daily returns = signal at time t * next day return
    # Shift the signal by 1 if you assume you can only trade at the close of day t
    df["StrategyReturn"] = df[signal_col].shift(1) * df["Return"]

    # Fill initial NAs with 0
    df["StrategyReturn"] = df["StrategyReturn"].fillna(0)

    # Compute cumulative returns
    df["CumMarketReturn"] = (1 + df["Return"]).cumprod()
    df["CumStrategyReturn"] = (1 + df["StrategyReturn"]).cumprod()

    total_return = df["CumStrategyReturn"].iloc[-1] - 1
    return df, total_return


def plot_results(df, title="Strategy Backtest"):
    plt.figure(figsize=(10, 6))
    df["CumMarketReturn"].plot(label="Market (Buy & Hold)", alpha=0.7)
    df["CumStrategyReturn"].plot(label="Strategy", alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.show()
