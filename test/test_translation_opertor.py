import numpy as np
from diffpy.structure import Lattice

from symmetry_breaking_measure import TranslationOperator


def test_apply_translation_against_ground_truth() -> None:
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
    direction = np.array([1, 1, 1])
    displacement = 1
    lattice = Lattice(a=1, b=1, c=1, alpha=90, beta=90, gamma=90)
    expected_xyz = np.array(
        [
            [0.57735027, 0.57735027, 0.57735027],
            [0.57735027, 0.57735027, 1.57735027],
            [0.57735027, 1.57735027, 0.57735027],
            [1.57735027, 0.57735027, 0.57735027],
            [0.57735027, 1.57735027, 1.57735027],
            [1.57735027, 0.57735027, 1.57735027],
            [1.57735027, 1.57735027, 0.57735027],
            [1.57735027, 1.57735027, 1.57735027],
        ]
    )

    # WHEN
    translation_op = TranslationOperator(
        direction=direction, displacement=displacement, lattice=lattice
    )
    transformed_xyz = translation_op.apply(original_xyz)

    # THEN
    assert np.allclose(transformed_xyz, expected_xyz)
