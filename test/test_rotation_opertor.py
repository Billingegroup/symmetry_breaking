import numpy as np
from diffpy.structure import Lattice

from symmetry_breaking_measure import RotationOperator


def test_apply_rotation_against_ground_truth() -> None:
    # GIVEN
    original_xyz = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    axis = np.array([0, 0, 1])
    origin = np.array([0.5, 0.5, 0])
    lattice = Lattice(a=1, b=1, c=1, alpha=90, beta=90, gamma=90)
    angle = 90
    expected_xyz = np.array(
        [
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
        ]
    )

    # WHEN
    rotation_op = RotationOperator(axis=axis, origin=origin, lattice=lattice)
    transformed_xyz = rotation_op.apply(original_xyz, angle)

    # THEN
    assert np.allclose(transformed_xyz, expected_xyz)