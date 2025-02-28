import pandas as pd
import numpy as np
from scipy.optimize import minimize


def load_and_prepare_data(csv_file: str) -> pd.DataFrame:
    """
    Loads CSV data into a DataFrame, ensures 'gmtTime' is datetime,
    sorts by date, and returns the raw DataFrame (all stocks).
    """
    df = pd.read_csv(csv_file)
    df["gmtTime"] = pd.to_datetime(df["gmtTime"])
    df.sort_values("gmtTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def compute_returns_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw DataFrame with multiple stocks into a wide returns DataFrame:
      - Each column is a symbol
      - Each row is a timestamp
      - Values are daily/hourly returns
    Assumes df has columns: 'gmtTime', 'askMedian', 'bidMedian', 'symbol'.
    """
    df.drop_duplicates(subset=["gmtTime", "symbol"], inplace=True)

    df["midPrice"] = (df["askMedian"] + df["bidMedian"]) / 2.0

    # We pivot by 'gmtTime' x 'symbol'
    # First, compute returns in the long form
    df.sort_values(["symbol", "gmtTime"], inplace=True)
    df["returns"] = df.groupby("symbol")["midPrice"].pct_change()

    # Pivot to wide (index=gmtTime, columns=symbol, values=returns)
    returns_wide = df.pivot(index="gmtTime", columns="symbol", values="returns")
    returns_wide.dropna(how="all", inplace=True)  # drop any row that's all NaN

    return returns_wide


def mvo_optimize(
    returns_wide: pd.DataFrame, target_return: float, long_only: bool = True
) -> pd.Series:
    """
    Solves a mean-variance optimization to minimize portfolio variance
    subject to:
       1) sum of weights = 1
       2) w^T mu >= target_return
       3) If long_only=True, then w_i >= 0
    Returns a Pandas Series of weights (indexed by symbol).

    returns_wide: columns = symbols, rows = time points
    target_return: desired min average return
    long_only: if True, w_i >= 0
    """
    # Drop any rows with NaN
    df = returns_wide.dropna(axis=0, how="any")
    # Mean returns vector
    mu = df.mean().values
    # Covariance matrix
    Sigma = df.cov().values
    symbols = df.columns
    n = len(symbols)

    def portfolio_variance(weights):
        return float(weights.T @ Sigma @ weights)

    # Constraints
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # sum of w = 1
        {
            "type": "ineq",
            "fun": lambda w: np.dot(w, mu) - target_return,
        },  # w^T mu >= target_return
    ]

    # Bounds
    if long_only:
        bounds = [(0.0, 1.0)] * n
    else:
        # No explicit constraints on negative weights
        # or do something else if you allow partial shorting.
        bounds = [(None, None)] * n

    # Initial guess
    x0 = np.ones(n) / n

    result = minimize(
        fun=portfolio_variance, x0=x0, bounds=bounds, constraints=cons, method="SLSQP"
    )

    if not result.success:
        raise RuntimeError(f"MVO optimization failed: {result.message}")

    # Return weights as a Pandas Series
    w_opt = result.x
    return pd.Series(w_opt, index=symbols, name="weight")
