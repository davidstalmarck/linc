import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from generate_data import generate_random_walk_price_series
from backtest_strategy import backtest_strategy, plot_results
from moving_average import moving_average_crossover


# Set random seed for reproducibility
# np.random.seed(42)


def test_moving_average_crossover(df):
    df_ma = df.copy()
    df_ma = moving_average_crossover(df_ma, short_window=20, long_window=50)
    df_ma, total_return_ma = backtest_strategy(df_ma, "MA_Signal")
    print(f"Moving Average Strategy Total Return: {total_return_ma:.2%}")
    plot_results(df_ma, title="Moving Average Crossover Strategy")


if __name__ == "__main__":
    df = generate_random_walk_price_series()
    print(df)
    # test_moving_average_crossover(df)
