from typing import List, Union

import numpy as np
from diffpy.structure import Lattice

from symmetry_breaking_measure.base_operator import BaseOperator


class ReflectionOperator(BaseOperator):
    """
    Define a 3d reflection operation on atoms in Cartesian coordinate.

    Properties:
    -----------
    axis : np.ndarray
        The unit vector in Cartesian coordinate which is perpendicular to the
        reflection plane.
    origin : np.ndarray
        A point in Cartesian coordinate that the reflection plane passes
        through.
    lattice : Lattice
        The lattice of the target unit cell. Default to be None.
    pbc: bool
        If pbc is True, the atoms affter transformation will be translated back
        to the default unit cell according to the periodic boundary condition.
        pbc can be True only when lattice is provided. Default to be False.


    Method:
    -------
    apply:
        Apply the reflection operation to the given structure. The method
        first translate the structure such that the origin is translated to
        the (0,0,0) point. Then it performs appropriate rotations to make the
        normal vector of the reflection plane at the origin until it coincides
        with the +z-axis. This makes the reflection plane the z = 0 coordinate
        plane. After that reflect the object through the z = 0 coordinate
        plane. Finally, it performs the inverse of the combined rotation
        transformation and the inverse of the translation in previous steps.
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
        The unit vector in Cartesian coordinate which is perpendicular to the
        reflection plane.
        """
        return self._axis

    @property
    def origin(self) -> np.ndarray:
        """
        A point in Cartesian coordinate that the reflection plane passes
        through.
        """
        return self._origin

    def apply(self, atoms_xyz: np.ndarray) -> np.ndarray:
        """
        Apply the reflection operation to the given structure. The method
        first translate the structure such that the origin is translated to
        the (0,0,0) point. Then it performs appropriate rotations to make the
        normal vector of the reflection plane at the origin until it coincides
        with the +z-axis. This makes the reflection plane the z = 0 coordinate
        plane. After that reflect the object through the z = 0 coordinate
        plane. Finally, it performs the inverse of the combined rotation
        transformation and the inverse of the translation in previous steps.

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
        atoms_centered = atoms_xyz - self._origin
        reflection_matrix = np.identity(3) - 2 * np.outer(
            self._axis, self._axis
        )
        atoms_reflected = atoms_centered.dot(reflection_matrix)
        atoms_xyz_transformed = atoms_reflected + self._origin
        return atoms_xyz_transformed
