import numpy as np
def calculate_var(returns, confidence=0.95):
    percentile = (
        1 - confidence
    ) * 100
    var = np.percentile(
        returns,
        percentile
    )
    return var
def calculate_cvar(
    returns,
    confidence=0.95
):
    var = calculate_var(
        returns,
        confidence
    )
    cvar = returns[
        returns <= var
    ].mean()
    return cvar

def calculate_drawdown(prices):
    cumulative_max = np.maximum.accumulate(
        prices
    )

    drawdown = (
        prices - cumulative_max
    ) / cumulative_max

    return drawdown