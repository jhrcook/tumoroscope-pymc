import pytest

from tumoroscope.mock_data import generate_random_data, generate_simulated_data


@pytest.mark.parametrize("K", (1, 3, 7))
@pytest.mark.parametrize("S", (1, 3, 7))
@pytest.mark.parametrize("M", (1, 3, 7))
def test_generate_random_data_is_valid(K: int, S: int, M: int) -> None:
    for _ in range(10):
        data = generate_random_data(K=K, S=S, M=M)
        data.validate()


@pytest.mark.parametrize("K", (2, 5))
@pytest.mark.parametrize("S", (5, 25))
@pytest.mark.parametrize("M", (50, 100))
@pytest.mark.parametrize("total_read_rate", (0.5, 1.0))
@pytest.mark.parametrize("cells_per_spot_range", ((1, 5), (1, 10)))
@pytest.mark.parametrize("zygosity_beta_dist", ((2, 2), (4, 7)))
def test_generate_simulated_data(
    K: int,
    S: int,
    M: int,
    total_read_rate: float,
    cells_per_spot_range: tuple[int, int],
    zygosity_beta_dist: tuple[float, float],
) -> None:
    for _ in range(10):
        data = generate_simulated_data(
            K=K,
            S=S,
            M=M,
            total_read_rate=total_read_rate,
            cells_per_spot_range=cells_per_spot_range,
            zygosity_beta_dist=zygosity_beta_dist,
        )
        data.sim_data.validate()
