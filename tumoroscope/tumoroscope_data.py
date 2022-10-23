"""Data structure for input into 'Tumoroscope' model."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TumoroscopeData:
    """'Tumoroscope' model data."""

    K: int  # number of clones
    S: int  # number of spots
    M: int  # number of mutation positions
    F: np.ndarray  # Prevelance of clones from bulk-DNA seq.
    cell_counts: np.ndarray  # Number of cell counted per spot
    C: np.ndarray  # Zygosity per position and clone
    D_obs: np.ndarray | None  # Read count per position per spot
    A_obs: np.ndarray | None  # Alternated reads per position per spot
    zeta_s: float = 1.0  # Pi hyper-parameter
    F_0: float = 1.0  # "pseudo-frequency" for lower bound on clone proportion
    l: float = 100  # Scaling factor to discretize F
    r: float = 0.1  # shape parameter for Gamma over Phi
    p: float = 1.0  # rate parameter for Gamma over Phi
