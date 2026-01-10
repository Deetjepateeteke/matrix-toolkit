#!usr/bin/python
# -*- coding: utf-8 -*-

"""
test_api.py - Includes testing for doing mathematical operations on matrices.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import pytest

from src import Matrix
from src.errors import DimensionError, NotSquareError


raises = pytest.raises


@pytest.mark.parametrize(("A", "B", "C"), [
    (
        Matrix([[1]]),
        Matrix([[2]]),
        Matrix([[3]])
    ),
    (
        Matrix([[3, 6, 0], [1, -3, 2], [-9, 3, 0]]),
        Matrix([[7, 2, -6], [-6, 0, 1], [-1, 3, 6]]),
        Matrix([[10, 8, -6], [-5, -3, 3], [-10, 6, 6]])
    ),

])
def test_adding(A, B, C):
    assert A + B == C
    assert B + A == C


@pytest.mark.parametrize(("A", "B"), [
    (
        Matrix([[1, 2, 3]]),
        Matrix([[1], [2], [3]])
    )
])
def test_invalid_adding(A, B):
    with raises(DimensionError):
        C = A + B

    with raises(DimensionError):
        C = B + A


@pytest.mark.parametrize(("A", "B", "C"), [
    (
        Matrix([[1]]),
        Matrix([[2]]),
        Matrix([[-1]])
    ),
    (
        Matrix([[3, 6, 0], [1, -3, 2], [-9, 3, 0]]),
        Matrix([[7, 2, -6], [-6, 0, 1], [-1, 3, 6]]),
        Matrix([[-4, 4, 6], [7, -3, 1], [-8, 0, -6]])
    ),

])
def test_subtracting(A, B, C):
    assert A - B == C
    assert A - C == B


@pytest.mark.parametrize(("A", "B"), [
    (
        Matrix([[1, 2, 3]]),
        Matrix([[1], [2], [3]])
    )
])
def test_invalid_subtracting(A, B):
    with raises(DimensionError):
        C = A - B

    with raises(DimensionError):
        C = B - A


@pytest.mark.parametrize(("n", "A", "B"), [
    (
        2,
        Matrix([[4, 6, 1], [-5, 3, -1], [0, 1, 4], [9, 5, 0]]),
        Matrix([[8, 12, 2], [-10, 6, -2], [0, 2, 8], [18, 10, 0]])
    )
])
def test_scalar_multiplication(n, A, B):
    assert n * A == B
    assert A * n == B


@pytest.mark.parametrize(("A", "B", "C"), [
    (
        Matrix([[2, 4, 5], [5, 2, 7]]),
        Matrix([[5, 3, 2, 5], [7, 2, 1, 3], [2, 3, 0, 1]]),
        Matrix([[48, 29, 8, 27], [53, 40, 12, 38]])
    )
])
def test_matrix_multiplication(A, B, C):
    assert A * B == C


@pytest.mark.parametrize(("A", "B"), [
    (
        Matrix([[2, 1, 4], [3, 4, 6]]),
        Matrix([[1, 2, 5], [5, 2, 1]])
    )
])
def test_invalid_multiplication(A, B):
    with raises(DimensionError):
        C = A * B


@pytest.mark.parametrize(("A", "n", "B"), [
    (Matrix([[2, 4], [3, -1]]), 0, Matrix([[1, 0], [0, 1]])),
    (Matrix([[2, 4], [3, -1]]), 1, Matrix([[2, 4], [3, -1]])),
    (Matrix([[2, 4], [3, -1]]), 2, Matrix([[16, 4], [3, 13]])),
    (Matrix([[2, 4], [3, -1]]), 3, Matrix([[44, 60], [45, -1]]))
])
def test_pow(A, n, B):
    assert A ** n == B


@pytest.mark.parametrize("A", [
    Matrix([[2, 1]]),
    Matrix([[5, 1], [0, 4], [3, 5]])
])
def test_invalid_pow(A):
    with raises(NotSquareError):
        B = A ** 2
