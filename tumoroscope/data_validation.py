"""'Tumoroscope' input data validation."""

import numpy as np

from .tumoroscope_data import TumoroscopeData


def validate_tumoroscope_data(data: TumoroscopeData) -> None:
    """Validate 'Tumoroscope' inport data."""
    assert data.K > 0 and data.S > 0 and data.M > 0

    assert np.isclose(data.F.sum(), 1.0)
    assert data.F.ndim == 1
    assert data.F.shape[0] == data.K

    assert data.cell_counts.shape == (data.S,)
    assert np.all(data.cell_counts > 0)

    assert np.all(data.C >= 0.0) and np.all(data.C <= 1.0)
    assert data.C.shape == (data.M, data.K)

    if data.D_obs is not None:
        assert data.D_obs.shape == (data.M, data.S)

    if data.A_obs is not None:
        assert data.A_obs.shape == (data.M, data.S)

    assert data.zeta_s > 0.0
    assert data.F_0 > 0.0
    assert data.l > 0.0
    assert data.r > 0.0 and data.p > 0.0
