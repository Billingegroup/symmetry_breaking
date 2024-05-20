from typing import List, Union

import numpy as np
from diffpy.structure import Lattice
from scipy.spatial.transform import Rotation as R

from symmetry_breaking_measure import BaseOperator


class RotationOperator(BaseOperator):
    """
    Define a 3d counter-clockwise rotation operation on atoms in Cartesian
    coordinate.

    Properties:
    -----------
    axis : np.ndarray
        The unit vector in Cartesian coordinate which is the axis of the 3d
        counter-clockwise rotation.
    origin : np.ndarray
        A point in Cartesian coordinate that the rotation axis passes through.
    lattice : Lattice
        The lattice of the target unit cell. Default to be None.
    pbc: bool
        If pbc is True, the atoms affter transformation will be translated back
        to the default unit cell according to the periodic boundary condition.
        pbc can be True only when lattice is provided. Default to be False.

    Method:
    -------
    apply:
        Apply the counter-clockwise rotation operation to the given structure.
    """

    def __init__(
        self,
        axis: Union[List[float], np.ndarray],
        origin: Union[List[float], np.ndarray],
        lattice: Lattice = None,
    ) -> None:
        super().__init__(lattice=lattice)
        self._axis = np.array(axis) / np.linalg.norm(np.array(axis))
        self._origin = np.array(origin)

    @property
    def axis(self) -> np.ndarray:
        """
        The unit vector in Cartesian coordinate which is the axis of the 3d
        counter-clockwise rotation.
        """
        return self._axis

    @property
    def origin(self) -> np.ndarray:
        """
        A point in Cartesian coordinate that the rotation axis passes through.
        """
        return self._origin

    def apply(  # pylint:disable=arguments-differ
        self, atoms_xyz: np.ndarray, angle: float
    ) -> np.ndarray:
        """
        Apply the counter-clockwise rotation operation to the given structure.

        Properties:
        -----------
        atoms : np.ndarray
            The atom sites in Cartesian coordinate, which is a N by 3 numpy
            array.
        angle : float
            The rotation angle in degree.

        Returns:
        --------
        atoms_transformed : np.ndarray
            The atom sites in Cartesian coordinate after applying the rotation
            operation.
        """
        rotation = R.from_rotvec((angle * np.pi / 180) * self._axis)
        atoms_centered = atoms_xyz - self._origin
        atoms_rotated = rotation.apply(atoms_centered)
        atoms_xyz_transformed = atoms_rotated + self._origin
        return atoms_xyz_transformed
