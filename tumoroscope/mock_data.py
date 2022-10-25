"""Generate mock data for 'Tumoroscope' input."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .tumoroscope_data import TumoroscopeData


def generate_random_data(K: int, S: int, M: int) -> TumoroscopeData:
    """Generate simple, random data for Tumoroscope.

    Args:
        K (int): Number of clones.
        S (int): Number of spots.
        M (int): Number of mutation positions.

    Returns:
        TumoroscopeData: Random data with no meaning, but passes validation.
    """
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


@dataclass
class SimulatedTumoroscopeData:
    """Simulated data for Tumoroscope."""

    sim_data: TumoroscopeData
    true_labels: pd.DataFrame
    clone_mutations: np.ndarray
    clone_zygosity: np.ndarray


def _make_mock_mutation_data(K: int, M: int) -> tuple[np.ndarray, int]:
    clone_mutations = np.random.binomial(1, 0.136, size=(M, K))
    # Drop positions without any mutations.
    clone_mutations = clone_mutations[clone_mutations.sum(axis=1) > 0.0, :]
    clone_mutations = clone_mutations[clone_mutations.mean(axis=1) < 1.0, :]
    return clone_mutations, clone_mutations.shape[0]


def generate_simulated_data(
    K: int,
    S: int,
    M: int,
    total_read_rate: float = 1.7,
    cells_per_spot_range: tuple[int, int] = (1, 5),
    zygosity_beta_dist: tuple[float, float] = (3, 1),
    random_seed: int | None = None,
    **kwargs: Any
) -> SimulatedTumoroscopeData:
    """Generate simulated data for Tumoroscope.

    This simulated data generating process is not perfect, but is useful for simple
    experimentation. For more rigorous testing, I would recommend implementing your own
    simulation procedure. Contributions are welcome.

    Args:
        K (int): Number of clones.
        S (int): Number of RNA transcriptomic spots.
        M (int): Number of mutation positions. This value may change depending on the
        random sampling process.
        total_read_rate (float, optional): Rate for the random Poisson distribution used
        to generate the number of reads per spot per cell. Defaults to 1.7.
        cells_per_spot_range (tuple[int, int], optional): Range to use for randomly
        generating the number of cells per spot. Defaults to (1, 5).
        zygosity_beta_dist (tuple[float, float], optional): Parameters for the beta
        distribution used to randomly assign zygosity of mutations. Defaults to (3, 1).
        random_seed (int | None, optional): Random seed to set 'numpy's random number
        generator. Defaults to None.

    Returns:
        SimulatedTumoroscopeData: Input and underlying true data for Tumoroscope.
    """
    # Set random state.
    if random_seed is not None:
        np.random.seed(random_seed)

    # Assign mutations across positions and clones.
    clone_mutations, M = _make_mock_mutation_data(K=K, M=M)

    # Number of cells per spot.
    cell_counts = np.random.randint(*cells_per_spot_range, size=S)
    # Assign cell IDs.
    _cell_labels = []
    for spot_s in range(S):
        for cell_i in range(cell_counts[spot_s]):
            _cell_labels.append((spot_s, cell_i, np.random.randint(K)))

    cell_labels = pd.DataFrame(_cell_labels, columns=["spot", "cell", "clone"]).assign(
        clone=lambda d: pd.Categorical(d["clone"], categories=list(range(K)))
    )
    clone_proportions = cell_labels.groupby("clone")["cell"].count().values
    clone_proportions = clone_proportions / clone_proportions.sum()

    # Assign zygosity at each position for each clone.
    _zygosity_fractions = np.random.beta(*zygosity_beta_dist, size=K)
    zygosity = np.random.choice(_zygosity_fractions, size=(M, K)) * clone_mutations

    # Build alternate and total read count matrices by random sampling using known
    # underlying cell ID.
    alt_read_counts = np.zeros((M, S, K))
    tot_read_counts = np.zeros((M, S, K))

    clone_cell_counts_per_spot = (
        cell_labels.groupby(["spot", "clone"])["cell"]
        .count()
        .reset_index()
        .rename(columns={"cell": "n_cells"})
    )

    for _, row in clone_cell_counts_per_spot.iterrows():
        clone_tot_reads = np.zeros(M)
        clone_alt_reads = np.zeros(M)
        for _ in range(row["n_cells"]):
            cell_tot_reads = np.random.poisson(total_read_rate, size=M)
            cell_alt_reads = np.random.binomial(
                cell_tot_reads, zygosity[:, row["clone"]]
            )
            clone_tot_reads = clone_tot_reads + cell_tot_reads
            clone_alt_reads = clone_alt_reads + cell_alt_reads
        tot_read_counts[:, row["spot"], row["clone"]] = clone_tot_reads
        alt_read_counts[:, row["spot"], row["clone"]] = clone_alt_reads

    alt_read_counts = alt_read_counts.sum(axis=2)
    tot_read_counts = tot_read_counts.sum(axis=2)

    sim_data = TumoroscopeData(
        K=K,
        S=S,
        M=M,
        F=clone_proportions,
        cell_counts=cell_counts,
        C=zygosity,
        D_obs=tot_read_counts,
        A_obs=alt_read_counts,
        **kwargs
    )
    return SimulatedTumoroscopeData(
        sim_data=sim_data,
        true_labels=cell_labels,
        clone_mutations=clone_mutations,
        clone_zygosity=zygosity,
    )
