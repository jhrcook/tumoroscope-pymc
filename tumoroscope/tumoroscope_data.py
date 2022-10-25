"""Data structure for input into 'Tumoroscope' model."""

from dataclasses import dataclass

import numpy as np

from .utilities import ErrorMessage


class TumoroscopeDataValidationError(BaseException):
    """Tumoroscope data validation error."""

    ...


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

    def validate(self) -> None:
        """Validate 'Tumoroscope' input data.

        Args:
            data (TumoroscopeData): Input data for building the 'Tumoroscope' model.

        Raises:
            TumoroscopeDataValidationError: Raised if various shape or value
            requirements are not met.
        """
        msg = ErrorMessage()
        if not (self.K > 0 and self.S > 0 and self.M > 0):
            msg("K, S, and M must be greater than 0.")

        if not np.isclose(self.F.sum(), 1.0):
            msg("F must sum to 1.")

        if not self.F.ndim == 1:
            msg("F must be only 1 dimension.")

        if not self.F.shape[0] == self.K:
            msg("F must have length of K.")

        if not self.cell_counts.shape == (self.S,):
            msg("Cell counts array must have dimension (S,).")

        if np.any(self.cell_counts <= 0):
            msg("Cell counts must be greater than 0 in all spots")

        if not (np.all(self.C >= 0.0) and np.all(self.C <= 1.0)):
            msg("All values for C must be between [0, 1].")

        if not self.C.shape == (self.M, self.K):
            msg("C must have shape (M,K).")

        if self.D_obs is not None:
            if self.D_obs.shape != (self.M, self.S):
                msg("D_obs must have shape (M,S).")
            if np.any(self.D_obs < 0):
                msg("D_obs must be non-negative.")

        if self.A_obs is not None:
            if self.A_obs.shape != (self.M, self.S):
                msg("A_obs must have shape (M,S).")
            if np.any(self.A_obs < 0):
                msg("A_obs must be non-negative.")

        _positives = np.array([self.zeta_s, self.F_0, self.l, self.r, self.p])
        if np.any(_positives <= 0):
            msg("zeta_s, F_0, l, r, and p must be greater than 0.")

        if msg.final_message is not None:
            raise TumoroscopeDataValidationError(msg.final_message)

    def __str__(self) -> str:
        s = "Tumoroscope Data\n"
        s += f" Data sizes:\n   K: {self.K}  S: {self.S}  M: {self.M}\n"
        s += (
            f" Hyperparameters:\n   zeta_s: {self.zeta_s}  F_0: {self.F_0}  l: {self.l}"
            + f"  r: {self.r}  p: {self.p}\n"
        )
        s += " Counts data:\n"
        if self.D_obs is None:
            s += "   D: (None)"
        else:
            s += "   D: provided"

        if self.A_obs is None:
            s += "  A: (None)"
        else:
            s += "  A: provided"

        return s

    def __repr__(self) -> str:
        return str(self)
