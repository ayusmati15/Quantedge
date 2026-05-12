import numpy as np


def brownian_motion(T, N):

    dt = T / N

    dW = np.random.normal(
        0,
        np.sqrt(dt),
        size=N
    )

    W = np.insert(
        np.cumsum(dW),
        0,
        0
    )

    return W


def simulate_gbm(
    S0,
    mu,
    sigma,
    T,
    N
):

    dt = T / N

    dW = np.random.normal(
        0,
        np.sqrt(dt),
        size=N
    )

    W = np.insert(
        np.cumsum(dW),
        0,
        0
    )

    t = np.linspace(
        0,
        T,
        N + 1
    )

    S = S0 * np.exp(
        (mu - 0.5 * sigma**2) * t
        + sigma * W
    )

    return t, S

def simulate_multiple_paths(
    S0,
    mu,
    sigma,
    T,
    N,
    num_paths
):

    dt = T / N

    Z = np.random.normal(
        0,
        1,
        (num_paths, N)
    )

    dW = np.sqrt(dt) * Z

    W = np.cumsum(dW, axis=1)

    W = np.hstack(
        [
            np.zeros((num_paths, 1)),
            W
        ]
    )

    t = np.linspace(
        0,
        T,
        N + 1
    )

    S = S0 * np.exp(
        (mu - 0.5 * sigma**2) * t
        + sigma * W
    )

    return t, S
