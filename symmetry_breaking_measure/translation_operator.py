from typing import List, Union

import numpy as np
from diffpy.structure import Lattice

from symmetry_breaking_measure import BaseOperator


class TranslationOperator(BaseOperator):
    """
    Define a 3d translation operation on atoms in Cartesian coordinate.

    Properties:
    -----------
    direction : np.ndarray
        The unit vector in Cartesian coordinate which specifies the direction
        of the translation.
    displacement : float
        A float number which specifies the length of movement along the
        direction. The size of the displacement should not be greater than the
        lattice parameter, i.e. the length of the unit cell.
    lattice : Lattice
        The lattice of the target unit cell. Default to be None.
    pbc: bool
        If pbc is True, the atoms affter transformation will be translated back
        to the default unit cell according to the periodic boundary condition.
        pbc can be True only when lattice is provided. Default to be False.

    Method:
    -------
    apply:
        Apply the translation operation to the given structure.
    """

    def __init__(
        self,
        direction: Union[List[float], np.ndarray],
        displacement: float,
        lattice: Lattice = None,
    ) -> None:
        super().__init__(lattice)
        self._direction = np.array(direction) / np.linalg.norm(
            np.array(direction)
        )
        self._displacement = displacement

    @property
    def direction(self) -> np.ndarray:
        """
        The unit vector in Cartesian coordinate which specifies the direction
        of the translation.
        """
        return self._direction

    @property
    def displacement(self) -> float:
        """
        A float number which specifies the length of movement along the
        direction. The size of the displacement should not be greater than the
        lattice parameter, i.e. the length of the unit cell.
        """
        return self._displacement

    def apply(self, atoms_xyz: np.ndarray) -> np.ndarray:
        """
        Apply the translation operation to the given structure.

        Properties:
        -----------
        atoms : np.ndarray
            The atom sites in Cartesian coordinate, which is a N by 3 numpy
            array.

        Returns:
        --------
        atoms_transformed : np.ndarray
            The atom sites in Cartesian coordinate after applying the
            translation operation.
        """
        atoms_xyz_transformed = (
            atoms_xyz + self._direction * self._displacement
        )
        return atoms_xyz_transformed
