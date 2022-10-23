import arviz as az
import numpy as np
import pymc as pm
import pytest

from tumoroscope import TumoroscopeData, build_tumoroscope_model


@pytest.fixture
def random_data() -> TumoroscopeData:
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


@pytest.mark.parametrize("fixed", (False, True))
def test_build_tumoroscope_model(random_data: TumoroscopeData, fixed: bool) -> None:
    model = build_tumoroscope_model(random_data, fixed=fixed)
    assert isinstance(model, pm.Model)
    assert ("N" in [v.name for v in model.free_RVs]) != fixed


@pytest.mark.parametrize("fixed", (False, True))
def test_tumoroscope_model_sampling(random_data: TumoroscopeData, fixed: bool) -> None:
    with build_tumoroscope_model(random_data, fixed=fixed):
        prior_pred = pm.sample_prior_predictive(samples=100)
    assert isinstance(prior_pred, az.InferenceData)
    if not fixed:
        prior_pred.prior["N"].shape == (1, 100, 5)

    assert prior_pred.prior["Z"].shape == (1, 100, 5, 3)
    assert prior_pred.prior["H"].shape == (1, 100, 5, 3)
    for v in ["A", "D"]:
        assert prior_pred.prior_predictive[v].shape == (1, 100, 7, 5)
