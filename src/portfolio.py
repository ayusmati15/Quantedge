import numpy as np
def portfolio_performance(
    weights,
    mean_returns,
    covariance
):

    portfolio_return = np.sum(
        mean_returns * weights
    ) * 252

    portfolio_volatility = np.sqrt(
        np.dot(
            weights.T,
            np.dot(
                covariance * 252,
                weights
            )
        )
    )

    return (
        portfolio_return,
        portfolio_volatility
    )


def sharpe_ratio(
    portfolio_return,
    portfolio_volatility,
    risk_free_rate=0.05
):

    return (
        portfolio_return
        - risk_free_rate
    ) / portfolio_volatility


def generate_random_portfolios(
    num_portfolios,
    mean_returns,
    covariance
):

    results = []

    weights_record = []

    num_assets = len(mean_returns)

    for _ in range(num_portfolios):

        weights = np.random.random(
            num_assets
        )

        weights /= np.sum(weights)

        portfolio_return, portfolio_volatility = (
            portfolio_performance(
                weights,
                mean_returns,
                covariance
            )
        )

        sr = sharpe_ratio(
            portfolio_return,
            portfolio_volatility
        )

        results.append(
            [
                portfolio_return,
                portfolio_volatility,
                sr
            ]
        )

        weights_record.append(weights)

    return (
        np.array(results),
        weights_record
    )