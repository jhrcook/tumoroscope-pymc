"""Generate mock data for 'Tumoroscope' input."""

import numpy as np

from .tumoroscope_data import TumoroscopeData


def generate_random_data() -> TumoroscopeData:
    """Simple data generation function."""
    K, S, M = 3, 5, 7
    return TumoroscopeData(
        K=K,
        S=S,
        M=M,
        F=np.ones(K) / K,
        cell_counts=np.random.randint(1, 20, size=S),
        C=np.random.beta(2, 2, size=(M, K)),
        D_obs=np.random.randint(2, 20, size=(M, S)),
        A_obs=np.random.randint(2, 20, size=(M, S)),
    )
