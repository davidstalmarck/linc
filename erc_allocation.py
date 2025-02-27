import pandas as pd
import numpy as np
from scipy.optimize import minimize


def compute_returns_from_csv(csv_file: str) -> pd.DataFrame:
    """
    Reads CSV data, computes midpoint price from 'askMedian' and 'bidMedian',
    calculates returns for each stock, and returns a pivoted dataframe
    where each column is a symbol's return time series.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing training data.

    Returns
    -------
    returns_wide : pd.DataFrame
        A pivoted dataframe of returns, each column is a stock symbol.
    """
    df = pd.read_csv(csv_file)

    # Compute midpoint price
    df["midPrice"] = (df["askMedian"] + df["bidMedian"]) / 2.0

    # Sort by time, just to be consistent
    df.sort_values(by=["gmtTime", "symbol"], inplace=True)

    # Group by symbol to compute returns
    #   SHIFT(1) so we can do (P_t - P_(t-1)) / P_(t-1)
    #   We'll store it in a new column
    df["return"] = df.groupby("symbol")["midPrice"].pct_change()

    # Pivot so each column is a symbol’s return over time, index = gmtTime
    #   For robust merging, we ensure that 'gmtTime' is a proper datetime index
    df["gmtTime"] = pd.to_datetime(df["gmtTime"])
    returns_wide = df.pivot(index="gmtTime", columns="symbol", values="return")

    # Drop rows with all NaN (e.g. the first row for each symbol)
    returns_wide.dropna(how="all", inplace=True)

    return returns_wide


def _risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Compute the individual risk contribution (RC) of each asset.

    RC_i = w_i * (Sigma * w)_i
    """
    # (Sigma * w)
    portfolio_var_terms = cov @ weights
    # elementwise multiplication w_i * (Sigma w)_i
    return weights * portfolio_var_terms


def _erc_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Objective function that tries to minimize the squared difference
    between each pair of assets' risk contributions.
    Essentially sum( (RC_i - RC_j)^2 ), for i < j.

    We want all risk contributions to be equal => this sum of squares should be small.
    """
    rc = _risk_contribution(weights, cov)
    # We can measure how far each pair is from each other
    # A simpler approach is variance of RC: var(RC).
    # Minimizing var(RC) also tries to make them equal.
    return np.var(rc)


def erc_weights(returns_wide: pd.DataFrame, max_iter=1000, tol=1e-8) -> pd.Series:
    """
    Given a wide DataFrame of returns, compute the ERC weights
    that allocate equal risk to each asset.

    Parameters
    ----------
    returns_wide : pd.DataFrame
        Each column is a symbol's return time series.
    max_iter : int
        Maximum number of iterations in the solver.
    tol : float
        Tolerance for termination.

    Returns
    -------
    w_series : pd.Series
        The ERC weights as a Pandas Series indexed by symbol.
    """

    # Drop any rows with NaN (if partial data is missing, you'll lose some rows)
    returns_wide = returns_wide.dropna()

    # Estimate covariance matrix
    cov_matrix = returns_wide.cov()

    n = len(cov_matrix)
    # Initial guess: 1/n for each asset
    x0 = np.ones(n) / n

    # Constraints:
    # 1) sum of weights = 1
    # 2) weights >= 0  (Long-only)
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # sum of weights = 1
    )
    bounds = tuple((0.0, 1.0) for _ in range(n))  # Nonnegative, up to 100%

    # Use SciPy minimize
    result = minimize(
        fun=_erc_objective,
        x0=x0,
        args=(cov_matrix.values,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": tol},
    )

    if not result.success:
        raise RuntimeError("ERC optimization failed: " + result.message)

    # The optimized weights
    w_opt = result.x

    # Create a nice Pandas Series with stock names
    stock_names = cov_matrix.columns
    w_series = pd.Series(data=w_opt, index=stock_names, name="weight")

    return w_series


def main():
    # Example usage
    csv_file = "datasets/stockPrices_hourly.csv"
    returns_wide = compute_returns_from_csv(csv_file)
    weights = erc_weights(returns_wide)
    print("Equal Risk Contribution Weights:")
    print(weights)
    print("\nSum of weights:", weights.sum())
    print("Individual Risk Contributions:")
    # For demonstration, compute each symbol's risk contribution
    cov_matrix = returns_wide.dropna().cov()
    rc = _risk_contribution(weights.values, cov_matrix.values)
    for symbol, rc_val in zip(weights.index, rc):
        print(f"{symbol}: {rc_val:.6f}")


if __name__ == "__main__":
    main()
