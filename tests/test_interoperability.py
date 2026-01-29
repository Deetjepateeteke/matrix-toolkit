#!usr/bin/python
# -*- coding: utf-8 -*-

"""
test_interoperability.py - Includes testing for the conversion of matrices.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import numpy as np
import pytest

from src import Matrix

raises = pytest.raises


test_cases = (
    (Matrix([[1]]), [[1]]),
    (Matrix.identity(3), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    (Matrix([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
    (Matrix.zeros((1, 5)), [[0, 0, 0, 0, 0]]),
    (Matrix.zeros((5, 1)), [[0], [0], [0], [0], [0]])
)


@pytest.mark.parametrize(("A", "B"), test_cases)
def test_to_list(A, B):
    assert A.to_list() == B


@pytest.mark.parametrize(("A", "B"), test_cases)
def test_to_numpy(A, B):
    A_np = A.to_numpy()  # Convert to numpy
    B_np = np.array(B)

    assert np.array_equal(A_np, B_np)

    A_np = np.array(A)  # Convert to numpy

    assert np.array_equal(A_np, B_np)


@pytest.mark.parametrize(("_", "B"), test_cases)
def test_from_numpy(_, B):
    B_np = np.array(B)  # Numpy array

    B_matrix = Matrix(B_np)  # Convert a numpy array to a Matrix

    assert B_matrix == Matrix(B)
