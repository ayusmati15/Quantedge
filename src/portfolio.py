import numpy as np
from scipy.optimize import minimize
from config import *

# Portfolio performance
def portfolio_performance(weights, mean_returns, covariance):
    ret = np.sum(mean_returns * weights) * 252
    vol = np.sqrt(weights.T @ (covariance * 252) @ weights)
    return ret, vol

# Sharpe Ratio
def sharpe_ratio(weights, mean_returns, covariance):
    ret, vol = portfolio_performance(weights, mean_returns, covariance)
    return -(ret - RISK_FREE_RATE) / vol

# Portfolio volatility
def portfolio_volatility(weights, covariance):
    return np.sqrt(weights.T @ (covariance * 252) @ weights)

# Maximum Sharpe Portfolio
def max_sharpe_portfolio(mean_returns, covariance):

    n = len(mean_returns)

    bounds = tuple((0,1) for _ in range(n))

    constraints = ({
        "type":"eq",
        "fun":lambda w: np.sum(w)-1
    })

    initial = np.ones(n)/n

    result = minimize(
        sharpe_ratio,
        initial,
        args=(mean_returns,covariance),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x

# Minimum Variance Portfolio
def minimum_variance_portfolio(mean_returns, covariance):

    n = len(mean_returns)

    bounds = tuple((0,1) for _ in range(n))

    constraints = ({
        "type":"eq",
        "fun":lambda w: np.sum(w)-1
    })

    initial = np.ones(n)/n

    result = minimize(
        portfolio_volatility,
        initial,
        args=(covariance,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x

# Efficient Frontier
def efficient_frontier(num_portfolios, mean_returns, covariance):

    portfolios=[]

    for _ in range(num_portfolios):

        w=np.random.random(len(mean_returns))
        w/=w.sum()

        ret,vol=portfolio_performance(
            w,
            mean_returns,
            covariance
        )

        sr=(ret-RISK_FREE_RATE)/vol

        portfolios.append([ret,vol,sr,w])

    return portfolios

# Position sizing
def position_size(capital, confidence):

    return capital*MAX_RISK_PER_TRADE*confidence

# Portfolio allocation
def allocate_portfolio(signals, confidence, capital):

    allocation=[]

    for s,c in zip(signals,confidence):

        if s=="HOLD":
            allocation.append(0)

        else:
            allocation.append(position_size(capital,c))

    return allocation

# Rebalance
def rebalance(weights):

    weights=np.array(weights)

    return weights/weights.sum()

# Complete pipeline
def optimize_portfolio(mean_returns,covariance):

    sharpe=max_sharpe_portfolio(
        mean_returns,
        covariance
    )

    minimum=minimum_variance_portfolio(
        mean_returns,
        covariance
    )

    frontier=efficient_frontier(
        5000,
        mean_returns,
        covariance
    )

    return {
        "max_sharpe":sharpe,
        "min_variance":minimum,
        "frontier":frontier
    }

if __name__=="__main__":
    print("Portfolio Module Loaded")
