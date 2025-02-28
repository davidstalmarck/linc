import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def compute_returns_from_csv(csv_file: str) -> pd.DataFrame:
    """
    Reads CSV data, computes midpoint price, and calculates returns for each stock.
    Returns a pivoted DataFrame where each column is a symbol's return time series.
    """
    df = pd.read_csv(csv_file)

    df.drop_duplicates(subset=["gmtTime", "symbol"], inplace=True)

    # Compute midpoint price
    df["midPrice"] = (df["askMedian"] + df["bidMedian"]) / 2.0

    # Filter to get only rows for INDEX1
    df_index1 = df[df["symbol"] == "INDEX1"].copy()

    # Sort by time (good practice before plotting time series)
    df_index1.sort_values("gmtTime", inplace=True)

    # Sort by time to be consistent
    df.sort_values(by=["gmtTime", "symbol"], inplace=True)

    # Calculate returns by symbol
    df["return"] = df.groupby("symbol")["midPrice"].pct_change()

    # Pivot so each column is a symbol’s returns; index = gmtTime
    df["gmtTime"] = pd.to_datetime(df["gmtTime"])
    returns_wide = df.pivot(index="gmtTime", columns="symbol", values="return")

    # Drop rows with all NaN (first row for each symbol, or missing data)
    returns_wide.dropna(how="all", inplace=True)

    return returns_wide


def solve_min_var_for_target(
    returns_wide: pd.DataFrame, target_return: float
) -> np.ndarray:
    """
    Solve the mean-variance optimization problem to obtain weights that:
      - Minimize w^T Sigma w  (variance)
      - Subject to: sum(w)=1, w^T mu >= target_return, w>=0

    Returns:
        w_opt : 1D numpy array of weights
    """
    # Drop any row that has NaNs across columns
    returns_wide = returns_wide.dropna()

    # Compute mean & covariance
    mu = returns_wide.mean().values
    Sigma = returns_wide.cov().values
    n = len(mu)

    # Objective: portfolio variance = w^T Sigma w
    def portfolio_variance(weights):
        return float(weights.T @ Sigma @ weights)

    # Constraints:
    # 1) sum of weights = 1
    # 2) w^T mu >= target_return
    # 3) w >= 0
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: np.dot(w, mu) - target_return},
    ]
    bnds = [(0.0, 1.0) for _ in range(n)]

    # Initial guess: equally weight all assets
    x0 = np.ones(n) / n

    result = minimize(
        fun=portfolio_variance, x0=x0, method="SLSQP", bounds=bnds, constraints=cons
    )

    if not result.success:
        # Could raise an error or return None. We'll raise to indicate infeasible or solver issue.
        raise RuntimeError(
            f"MVO optimization failed for target={target_return}: {result.message}"
        )

    return result.x


def plot_efficient_frontier(returns_wide: pd.DataFrame, num_points=20):
    """
    Build and plot the efficient frontier by solving the min-var problem for
    a range of target returns.
    """
    # Drop NaNs
    returns_wide = returns_wide.dropna()
    mu_series = returns_wide.mean()  # average return per stock
    cov_matrix = returns_wide.cov()  # covariance

    # We'll pick target returns from the min mean return to the max mean return.
    # Alternatively, pick your own range based on domain knowledge.
    mu_min = mu_series.min()
    mu_max = mu_series.max()

    # For storing solutions
    frontier_risk = []
    frontier_return = []

    # Generate a sequence of target returns between mu_min and mu_max
    # If your data can be negative, you may offset or clamp at zero. Up to you.
    targets = np.linspace(mu_min, mu_max, num_points)

    for t_return in targets:
        try:
            w_opt = solve_min_var_for_target(returns_wide, t_return)
            # Evaluate the portfolio's realized mean return
            port_ret = np.dot(w_opt, mu_series.values)
            # Evaluate the variance => stdev
            port_var = w_opt.T @ cov_matrix.values @ w_opt
            port_std = np.sqrt(port_var)

            frontier_risk.append(port_std)
            frontier_return.append(port_ret)
        except RuntimeError:
            # In case some target_return is infeasible, skip it
            pass

    # Plot the efficient frontier
    plt.figure(figsize=(8, 5))
    plt.plot(frontier_risk, frontier_return, "o--", label="Efficient Frontier")
    plt.xlabel("Portfolio Risk (Std. Dev.)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier (Mean-Variance Optimization)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Example usage
    csv_file = "datasets/Historical_Data.csv"
    returns_wide = compute_returns_from_csv(csv_file)

    # Plot efficient frontier for these stocks
    plot_efficient_frontier(returns_wide, num_points=50)


if __name__ == "__main__":
    main()
