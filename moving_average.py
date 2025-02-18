from backtest_strategy import backtest_strategy
from backtest_strategy import plot_results
from generate_data import generate_random_walk_price_series
def moving_average_crossover(df, short_window=20, long_window=50):
    """
    Generate signals when the short moving average crosses the long moving average.
    Here we go long if short_MA > long_MA, else flat.
    """
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    
    # Signal +1 if short_sma > long_sma, else 0
    df['MA_Signal'] = 0
    df.loc[df['SMA_short'] > df['SMA_long'], 'MA_Signal'] = 1
    
    return df

