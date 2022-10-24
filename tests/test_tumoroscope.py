import arviz as az
import pymc as pm
import pytest

from tumoroscope import TumoroscopeData, build_tumoroscope_model


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
