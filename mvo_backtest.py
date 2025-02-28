import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from mvo_strategies import load_and_prepare_data, compute_returns_wide, mvo_optimize


def rolling_mvo_backtest(
    returns_wide: pd.DataFrame,
    window_size: int = 50,
    target_return: float = 0.001,  # e.g. 0.1% per period
):
    """
    Rolling backtest: at each step t from window_size to the end:
    - Take the last `window_size` rows as "in-sample" data
    - Solve MVO to get weights (including a CASH column)
    - Compute 1-step realized return at step t+1 using those weights
    - Accumulate PnL
    """

    # We'll store an equity curve
    initial_capital = 100_000.0
    capital = initial_capital
    equity_curve = []
    dates = []

    # Make sure we have no all-NaN rows
    returns_wide.dropna(how="all", inplace=True)
    returns_wide.sort_index(inplace=True)

    # Ensure we have a "CASH" column that is 0.0 for all rows
    # (If you want to do this in compute_returns_wide, that's fine, too.)
    if "CASH" not in returns_wide.columns:
        returns_wide["CASH"] = 0.0

    # We treat each row as a "step."
    all_dates = returns_wide.index
    symbols = returns_wide.columns

    # Start with equal weights among the real assets + cash
    n = len(symbols)
    w_prev = pd.Series(index=symbols, data=1.0 / n)  # equal weighting
    all_weights = pd.DataFrame(index=all_dates, columns=symbols)
    all_weights.iloc[0] = w_prev

    # We'll hold the portfolio from time t to t+1
    for i in range(window_size, len(returns_wide) - 1):
        # in-sample slice: [i-window_size, i)
        train_slice = returns_wide.iloc[i - window_size : i]

        # Solve MVO
        # Here, we do a simple long-only MVO with no forced min cash
        # If you want to force min cash, see next section
        try:
            w_opt = mvo_optimize(
                train_slice, target_return=target_return, long_only=True
            )
            print("MVO Weights at time", all_dates[i], ":\n", w_opt)
        except RuntimeError as e:
            # If it's infeasible or fails, reuse previous weights
            w_opt = w_prev
            print(f"Error: {e}")

        # realized returns at step i+1
        returns_next = returns_wide.iloc[i + 1]  # row for next time
        if returns_next.isna().any():
            # Skip if there's any NaN in next returns
            equity_curve.append(capital)
            all_weights.iloc[i + 1] = w_opt
            dates.append(all_dates[i + 1])
            continue

        port_ret = np.dot(w_opt.values, returns_next.values)

        w_prev = w_opt
        # update capital
        capital *= 1.0 + port_ret

        equity_curve.append(capital)
        all_weights.iloc[i + 1] = w_opt
        dates.append(all_dates[i + 1])

    # Build a DataFrame of results
    results_df = pd.DataFrame({"gmtTime": dates, "equity": equity_curve})
    results_df.set_index("gmtTime", inplace=True)
    return results_df, all_weights


def evaluate_backtest(equity_df: pd.DataFrame):
    """
    Compute some performance metrics: total return, Sharpe, etc.
    """
    equity_df["returns"] = equity_df["equity"].pct_change().fillna(0.0)
    total_return = equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1.0
    ret_series = equity_df["returns"]
    ann_factor = 365 * 24  # e.g. hourly data
    if ret_series.std() != 0:
        sharpe = (ret_series.mean() / ret_series.std()) * np.sqrt(ann_factor)
    else:
        sharpe = 0

    return {
        "Final Equity": equity_df["equity"].iloc[-1],
        "Total Return": total_return,
        "Sharpe": sharpe,
    }


def main():
    csv_file = "datasets/Historical_Data.csv"
    df_raw = load_and_prepare_data(csv_file)

    # The function compute_returns_wide might also insert a "CASH" column if you prefer
    returns_wide = compute_returns_wide(df_raw)

    # For example: rolling 100-step window, target return of 0.1% per step
    window_size = 100
    target_return = 0.001

    results_df, all_weights = rolling_mvo_backtest(
        returns_wide, window_size, target_return
    )
    perf = evaluate_backtest(results_df)

    print("Backtest Results:")
    print("Final Equity: {:.2f}".format(perf["Final Equity"]))
    print("Total Return: {:.2%}".format(perf["Total Return"]))
    print("Sharpe Ratio: {:.2f}".format(perf["Sharpe"]))

    # Plot equity curve
    plt.plot(results_df["equity"], label="Equity Curve")
    plt.title("MVO Backtest with Cash")
    plt.legend()
    plt.show()

    # If you want, you can also plot how much 'CASH' is held over time
    # all_weights["CASH"].plot()
    # plt.title("CASH allocation over time")
    # plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
