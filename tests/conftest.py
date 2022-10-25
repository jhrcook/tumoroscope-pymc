from typing import Callable

import numpy as np
import pytest

from tumoroscope import TumoroscopeData


@pytest.fixture
def make_random_data() -> Callable[[int, int, int], TumoroscopeData]:
    def _make_random_data(K: int, S: int, M: int) -> TumoroscopeData:
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

    return _make_random_data


@pytest.fixture
def random_data(
    make_random_data: Callable[[int, int, int], TumoroscopeData]
) -> TumoroscopeData:
    return make_random_data(3, 5, 7)
