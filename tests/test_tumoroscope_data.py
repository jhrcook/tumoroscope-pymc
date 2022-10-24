from dataclasses import asdict
from typing import Callable

import numpy as np
import pytest

from tumoroscope.tumoroscope_data import TumoroscopeData, TumoroscopeDataValidationError

random_nonzero_data_args = pytest.mark.parametrize(
    "rand_data_args", ((2, 3, 5), (1, 5, 7), (3, 1, 5), (1, 1, 1))
)


@pytest.mark.parametrize(
    "rand_data_args",
    (
        (0, 3, 5),
        (3, 0, 5),
        (1, 3, 0),
        (0, 3, 0),
        (0, 0, 5),
        (0, 3, 0),
        (-1, 3, 5),
        (3, -1, 5),
        (1, 3, -1),
    ),
)
def test_validate_nonzero_dimension_constants(
    make_random_data: Callable[[int, int, int], TumoroscopeData],
    rand_data_args: tuple[int, int, int],
) -> None:
    data = make_random_data(*[abs(x) for x in rand_data_args])
    data.K, data.S, data.M = rand_data_args
    with pytest.raises(TumoroscopeDataValidationError) as err:
        data.validate()
    assert "K, S, and M must be greater than 0" in str(err.value)


@pytest.mark.parametrize("param", (("zeta_s", "F_0", "l", "r", "p")))
@pytest.mark.parametrize("val", (0, -1))
def test_nonzero_hyperparameter_constants(
    random_data: TumoroscopeData, param: str, val: float
) -> None:
    _info = asdict(random_data)
    _info[param] = val
    newdata = TumoroscopeData(**_info)
    with pytest.raises(TumoroscopeDataValidationError) as err:
        newdata.validate()
    assert "zeta_s, F_0, l, r, and p must be greater than 0." in str(err.value)


def test_F_not_sum_to_zero(random_data: TumoroscopeData) -> None:
    random_data.F = np.ones(random_data.K)
    with pytest.raises(TumoroscopeDataValidationError) as err:
        random_data.validate()
    msg = str(err.value)
    assert "F" in msg and "sum to 1" in msg


@pytest.mark.parametrize("n_off", (-5, -1, 1, 5))
def test_cell_counts_wrong_dimensions(random_data: TumoroscopeData, n_off: int) -> None:
    random_data.cell_counts = np.ones(random_data.S + n_off)
    with pytest.raises(TumoroscopeDataValidationError) as err:
        random_data.validate()
    assert "cell counts" in str(err.value).lower()


@pytest.mark.parametrize("spot_s", (0, 1, 5, 10, -1))
@pytest.mark.parametrize("value", (0, -1, -10))
def test_cell_counts_nonpositive(
    make_random_data: Callable[[int, int, int], TumoroscopeData],
    spot_s: int,
    value: int,
) -> None:
    data = make_random_data(3, 19, 7)
    data.cell_counts[spot_s] = value
    with pytest.raises(TumoroscopeDataValidationError) as err:
        data.validate()
    assert "Cell counts must be greater than 0" in str(err.value)


@random_nonzero_data_args
@pytest.mark.parametrize("M_off", (-1, 1))
@pytest.mark.parametrize("K_off", (-1, 1))
def test_validate_zygosity_dimensions(
    make_random_data: Callable[[int, int, int], TumoroscopeData],
    rand_data_args: tuple[int, int, int],
    M_off: int,
    K_off: int,
) -> None:
    data = make_random_data(*rand_data_args)
    data.C = np.random.uniform(0, 1, (data.M + M_off, data.K + K_off))
    with pytest.raises(TumoroscopeDataValidationError) as err:
        data.validate()
    assert "C must have shape (M,K)." in str(err.value)


def test_validate_zygosity_negative(random_data: TumoroscopeData) -> None:
    random_data.C = np.ones((random_data.M, random_data.K)) * -1
    with pytest.raises(TumoroscopeDataValidationError) as err:
        random_data.validate()
    assert "All values for C must be between [0, 1]." in str(err.value)


@pytest.mark.parametrize("param", ("D_obs", "A_obs"))
@pytest.mark.parametrize("value", (-1, -5, -1.0, -5.0))
@pytest.mark.parametrize("pos", ((0, 0), (1, 1), (5, 5)))
def test_counts_data_negative(
    make_random_data: Callable[[int, int, int], TumoroscopeData],
    param: str,
    value: float | int,
    pos: tuple[int, int],
) -> None:
    _info = asdict(make_random_data(5, 13, 23))
    _info[param][pos[0], pos[1]] = value
    data = TumoroscopeData(**_info)
    with pytest.raises(TumoroscopeDataValidationError) as err:
        data.validate()
    err_msg = str(err.value)
    assert param in err_msg
    assert "must be non-negative" in err_msg
