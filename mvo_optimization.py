import pandas as pd
import numpy as np
from scipy.optimize import minimize


def compute_returns_from_csv(csv_file: str) -> pd.DataFrame:
    """
    Reads CSV data, computes midpoint price, and calculates returns for each stock.
    Returns a pivoted dataframe where each column is a symbol's return time series.
    """
    df = pd.read_csv(csv_file)

    # Compute midpoint price
    df["midPrice"] = (df["askMedian"] + df["bidMedian"]) / 2.0

    # Sort by time to be consistent
    df.sort_values(by=["gmtTime", "symbol"], inplace=True)

    # Calculate returns by symbol
    df["return"] = df.groupby("symbol")["midPrice"].pct_change()

    # Pivot so each column is a symbol’s returns; index = gmtTime
    df["gmtTime"] = pd.to_datetime(df["gmtTime"])
    returns_wide = df.pivot(index="gmtTime", columns="symbol", values="return")

    # Drop rows with all NaN (e.g. first row for each symbol)
    returns_wide.dropna(how="all", inplace=True)

    return returns_wide


def mvo_weights(returns_wide: pd.DataFrame, target_return: float) -> pd.Series:
    """
    Compute the portfolio weights via Mean-Variance Optimization (MVO).

    Minimizes portfolio variance subject to:
      1) Sum of weights = 1
      2) w^T mu >= target_return
      3) w >= 0 (long-only)

    Parameters
    ----------
    returns_wide : pd.DataFrame
        Wide DataFrame of returns (each column is a stock's return).
    target_return : float
        Desired minimum average return (in the same period units as your returns).

    Returns
    -------
    w_series : pd.Series
        The MVO weights as a Pandas Series indexed by symbol.
    """

    # Drop any rows that contain NaNs across stocks
    returns_wide = returns_wide.dropna()

    # Estimate average returns (mean) and covariance of returns
    mu = returns_wide.mean()  # average return per stock
    cov_matrix = returns_wide.cov()  # covariance matrix

    # Convert to numpy for optimization
    mu_vals = mu.values
    cov_vals = cov_matrix.values

    n = len(mu_vals)

    # MVO: minimize w^T Sigma w
    def portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
        return float(weights.T @ cov @ weights)

    # Constraints:
    #  1) sum of weights = 1
    #  2) portfolio return >= target_return => w^T mu - target_return >= 0
    #  3) w >= 0 for all w
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: np.dot(w, mu_vals) - target_return},
    ]
    bounds = [(0.0, 1.0) for _ in range(n)]  # long-only

    # Initial guess: equally weight all assets
    x0 = np.ones(n) / n

    # Solve using SLSQP
    result = minimize(
        fun=portfolio_variance,
        x0=x0,
        args=(cov_vals,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError("MVO optimization failed: " + result.message)

    w_opt = result.x
    # Convert to a Pandas Series
    w_series = pd.Series(data=w_opt, index=mu.index, name="weight")
    return w_series


def main():
    csv_file = "datasets/stockPrices_hourly.csv"
    returns_wide = compute_returns_from_csv(csv_file)

    # Suppose we want a 1% average return (0.01) in the same period your returns represent.
    # If your returns are hourly, that might be quite high or low in practice.
    # Adjust as needed for your scale.
    target_return = 0.0

    weights = mvo_weights(returns_wide, target_return)

    print(
        "Mean-Variance Optimization Weights (Target Return = {:.2%}):".format(
            target_return
        )
    )
    print(weights)
    print("\nPortfolio Return:", (weights * returns_wide.mean()).sum())
    print(
        "Portfolio Variance:",
        weights.values.T @ returns_wide.cov().values @ weights.values,
    )
    print("\nSum of weights:", weights.sum())


if __name__ == "__main__":
    main()
