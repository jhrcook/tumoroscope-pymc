"""The 'Tumoroscope' model definition."""

import aesara.tensor as at
import numpy as np
import pymc as pm

from .tumoroscope_data import TumoroscopeData


def _prefixed_index(n: int, prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in np.arange(n)]


def _make_tumoroscope_model_coords(data: TumoroscopeData) -> dict[str, list[str]]:
    coords = {
        "clone": _prefixed_index(data.K, "c"),
        "spot": _prefixed_index(data.S, "s"),
        "position": _prefixed_index(data.M, "p"),
    }
    return coords


def build_tumoroscope_model(
    data: TumoroscopeData, fixed: bool = False, *, validate_data: bool = True
) -> pm.Model:
    """Build the 'Tumoroscope' model.

    Args:
        data (TumoroscopeData): Input data.
        validate_data (bool, optional): Whether to validate the data confirms to
        certain structural and value requirements. Defaults to `True`.

    Returns:
        pm.Model: PyMC model.
    """
    if validate_data:
        data.validate()

    coords = _make_tumoroscope_model_coords(data)
    with pm.Model(coords=coords) as model:
        ζ_s = pm.ConstantData("ζ_s", data.zeta_s)
        ℓ = pm.ConstantData("ℓ", data.l)
        F_0 = pm.ConstantData("F0", data.F_0)
        F = pm.ConstantData("F", data.F, dims="clone")
        if not fixed:
            Λ = pm.ConstantData("Λ", data.cell_counts, dims="spot")
        r = pm.ConstantData("r", data.r)
        p = pm.ConstantData("p", data.p)
        C = pm.ConstantData("C", data.C, dims=("position", "clone"))

        F_prime = pm.Deterministic("F_prime", ℓ * at.ceil(20 * F) / 20, dims="clone")

        Π = pm.Beta("Π", alpha=ζ_s / data.K, beta=1, dims=("spot", "clone"))
        Z = pm.Bernoulli("Z", p=Π, dims=("spot", "clone"))
        G = pm.Gamma(
            "G", (F_prime[None, :] ** Z) * (F_0 ** (1 - Z)), 1, dims=("spot", "clone")
        )
        H = pm.Deterministic("H", G / G.sum(axis=1)[:, None], dims=("spot", "clone"))

        if fixed:
            N = pm.ConstantData("N", data.cell_counts, dims="spot")
        else:
            N = pm.Poisson("N", Λ, dims="spot")
        Φ = pm.Gamma("Φ", r, p, dims=("position", "clone"))

        D = pm.Poisson(
            "D", N * H.dot(Φ.T).T, dims=("position", "spot"), observed=data.D_obs
        )
        _A_num = H.dot((Φ * C).T).T
        _A_denom = H.dot(Φ.T).T
        A_prob = pm.Deterministic(
            "A_prob", _A_num / _A_denom, dims=("position", "spot")
        )
        pm.Binomial("A", D, A_prob, dims=("position", "spot"), observed=data.A_obs)
    return model
