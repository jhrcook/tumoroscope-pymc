"""'Tumoroscope' input data validation."""

import numpy as np

from .tumoroscope_data import TumoroscopeData


class TumoroscopeDataValidationError(BaseException):
    """Tumoroscope data validation error."""

    ...


def validate_tumoroscope_data(data: TumoroscopeData) -> None:
    """Validate 'Tumoroscope' input data.

    Args:
        data (TumoroscopeData): Input data for building the 'Tumoroscope' model.

    Raises:
        TumoroscopeDataValidationError: Raised if various shape or value requirements
        are not met.
    """
    if not (data.K > 0 and data.S > 0 and data.M > 0):
        raise TumoroscopeDataValidationError("K, S, and M must be greater than 0.")

    if not np.isclose(data.F.sum(), 1.0):
        raise TumoroscopeDataValidationError("F must sum to 1.")

    if not data.F.ndim == 1:
        raise TumoroscopeDataValidationError("F must be only 1 dimension.")

    if not data.F.shape[0] == data.K:
        raise TumoroscopeDataValidationError("F must have length of K.")

    if not data.cell_counts.shape == (data.S,):
        raise TumoroscopeDataValidationError(
            "Cell counts array must have dimension (S,)."
        )

    if np.any(data.cell_counts == 0):
        raise TumoroscopeDataValidationError(
            "Cell counts must be greater than 0 in all spots"
        )

    if not (np.all(data.C >= 0.0) and np.all(data.C <= 1.0)):
        raise TumoroscopeDataValidationError("All values for C must be between [0, 1].")

    if not data.C.shape == (data.M, data.K):
        raise TumoroscopeDataValidationError("C must have shape (M,K).")

    if data.D_obs is not None and data.D_obs.shape != (data.M, data.S):
        raise TumoroscopeDataValidationError("D_obs must have shape (M,S).")

    if data.A_obs is not None and data.A_obs.shape != (data.M, data.S):
        raise TumoroscopeDataValidationError("A_obs must have shape (M,S).")

    _positives = np.array([data.zeta_s, data.F_0, data.l, data.r, data.p])
    if np.any(_positives <= 0):
        raise TumoroscopeDataValidationError(
            "zeta_s, F_0, l, r, and p must be greater than 0."
        )
