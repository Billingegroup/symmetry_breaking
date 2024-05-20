from abc import ABC, abstractmethod

import numpy as np
from diffpy.structure import Lattice


class BaseOperator(ABC):
    def __init__(self, lattice: Lattice = None) -> None:
        self._lattice = lattice

    @property
    def lattice(self) -> Lattice:
        """
        The lattice of the target unit cell. Default to be empty.
        """
        return self._lattice

    @abstractmethod
    def apply(self, atoms_xyz: np.ndarray) -> np.ndarray:
        """
        Apply the symmetry operation to the given structure. This method should be
        overridden in derived classes.

        Properties:
        -----------
        atoms : np.ndarray
            The atom sites in Cartesian coordinate, which is a N by 3 numpy
            array.

        Returns:
        --------
        atoms_transformed : np.ndarray
            The atom sites in Cartesian coordinate after applying the symmetry
            operation.
        """
        pass  # pylint: disable=unnecessary-pass
