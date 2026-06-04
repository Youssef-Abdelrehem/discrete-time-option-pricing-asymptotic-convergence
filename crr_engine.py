import numpy as np
from scipy.stats import norm


def bsm_price(S0, K, T, r, sigma, option_type="put"):
    """Black-Scholes-Merton closed-form price."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def crr_price(S0, K, T, r, sigma, N, option_type="put", exercise="european"):
    """
    CRR binomial tree pricer for European and American options.

    Uses a single 1D array updated in-place during backward induction —
    O(N) memory, O(N^2) time.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    q = (np.exp(r * dt) - d) / (u - d)

    # Terminal asset prices: S0 * u^j * d^(N-j), j = 0..N
    j = np.arange(N + 1)
    S_T = S0 * (u ** j) * (d ** (N - j))

    # Terminal payoffs
    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # Backward induction
    for step in range(N - 1, -1, -1):
        # Continuation value (risk-neutral expectation, discounted)
        V = disc * (q * V[1:] + (1 - q) * V[:-1])

        if exercise == "american":
            # Asset prices at this node level
            j_step = np.arange(step + 1)
            S_step = S0 * (u ** j_step) * (d ** (step - j_step))
            if option_type == "call":
                intrinsic = np.maximum(S_step - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_step, 0.0)
            V = np.maximum(V, intrinsic)

    return float(V[0])


def jr_price(S0, K, T, r, sigma, N, option_type="put", exercise="european"):
    """
    Jarrow-Rudd binomial tree pricer for European and American options.

    JR parametrization: equal risk-neutral probabilities (q = 1/2) with
    the drift correction embedded in the jump sizes instead of in q.
    The tree is recombining but NOT symmetric: u*d != 1.
    """
    dt = T / N
    drift = (r - 0.5 * sigma**2) * dt
    u = np.exp(drift + sigma * np.sqrt(dt))
    d = np.exp(drift - sigma * np.sqrt(dt))
    disc = np.exp(-r * dt)
    q = 0.5  # equal probabilities by construction

    # Terminal asset prices: S0 * u^j * d^(N-j), j = 0..N
    j = np.arange(N + 1)
    S_T = S0 * (u ** j) * (d ** (N - j))

    # Terminal payoffs
    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # Backward induction — identical logic to CRR, different u/d/q
    for step in range(N - 1, -1, -1):
        V = disc * (q * V[1:] + (1 - q) * V[:-1])

        if exercise == "american":
            j_step = np.arange(step + 1)
            S_step = S0 * (u ** j_step) * (d ** (step - j_step))
            if option_type == "call":
                intrinsic = np.maximum(S_step - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_step, 0.0)
            V = np.maximum(V, intrinsic)

    return float(V[0])


def richardson_price(S0, K, T, r, sigma, N, option_type="put", exercise="european"):
    """Richardson extrapolation: 2·CRR(2N) − CRR(N).  Error O(1/N) vs O(1/√N)."""
    return (2 * crr_price(S0, K, T, r, sigma, 2 * N, option_type, exercise)
              - crr_price(S0, K, T, r, sigma,     N, option_type, exercise))
