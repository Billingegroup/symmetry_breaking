from typing import List, Union

import numpy as np
from diffpy.structure import Lattice

from symmetry_breaking_measure.base_operator import BaseOperator


class InversionOperator(BaseOperator):
    """
    Define a 3d inversion operation on atoms in Cartesian coordinate.

    Properties:
    -----------
    origin : np.ndarray
        A point in Cartesian coordinate that the inversion is with respect to.
    lattice : Lattice
        The lattice of the target unit cell. Default to be None.
    pbc: bool
        If pbc is True, the atoms affter transformation will be translated back
        to the default unit cell according to the periodic boundary condition.
        pbc can be True only when lattice is provided. Default to be False.

    Method:
    -------
    apply:
        Apply the inversion operation to the given structure.
    """

    def __init__(
        self,
        origin: Union[List[float], np.ndarray],
        lattice: Lattice = None,
    ) -> None:
        super().__init__(lattice)
        self._origin = np.array(origin)

    @property
    def origin(self) -> np.ndarray:
        """
        A point in Cartesian coordinate that the inversion is with respect to.
        """
        return self._origin

    @property
    def lattice(self) -> Lattice:
        """
        The lattice of the target unit cell. Default to be empty.
        """
        return self._lattice

    def apply(self, atoms_xyz: np.ndarray) -> np.ndarray:
        """
        Apply the inversion operation to the given structure.

        Properties:
        -----------
        atoms : np.ndarray
            The atom sites in Cartesian coordinate, which is a N by 3 numpy
            array.

        Returns:
        --------
        atoms_transformed : np.ndarray
            The atom sites in Cartesian coordinate after applying the
            reflection operation.
        """
        num_of_atoms = atoms_xyz.shape[0]
        origin = np.tile(self._origin.reshape(1, -1), (num_of_atoms, 1))
        atoms_xyz_transformed = 2 * origin - atoms_xyz
        return atoms_xyz_transformed
