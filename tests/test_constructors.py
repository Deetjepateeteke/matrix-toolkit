#!usr/bin/python
# -*- coding: utf-8 -*-

"""
test_constructors.py - Includes testing for the creation of matrices.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import pytest

from src import Matrix
from src.errors import DimensionError, InvalidElementError

raises = pytest.raises


@pytest.mark.parametrize("data", [
    [],
    [[]],
    [[], []],
    [[0], []],
    [[], [0]],
    [[[0]]],
    [[0, 0], [0]],
    [[0], [0, 0]]
])
def test_invalid_dimensions(data):
    with raises(DimensionError):
        Matrix(data)


@pytest.mark.parametrize("data", [
    [["0"]],
    [["0", 0]],
    [[0, "0"]],
    [[True]],
    [[False, []]]
])
def test_invalid_elements(data):
    with raises(InvalidElementError):
        Matrix(data)


@pytest.mark.parametrize(("A", "B"), [
    (
        Matrix([[-2, 5, 1], [1, 0, 4]]),
        Matrix([[-2, 1], [5, 0], [1, 4]])
    )
])
def test_transpose(A, B):
    assert A.transpose() == B
    assert B.transpose() == A

    assert A.transpose().transpose() == A
    assert B.transpose().transpose() == B


@pytest.mark.parametrize(("A", "B"), [
    (
        Matrix.identity(1),
        Matrix([[1]])
    ),
    (
        Matrix.identity(2),
        Matrix([[1, 0], [0, 1]])
    ),
    (
        Matrix.identity(3),
        Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )
])
def test_identity(A, B):
    assert A == B


@pytest.mark.parametrize("n", [0, -1])
def test_invalid_identity(n):
    with raises(DimensionError):
        Matrix.identity(n)


@pytest.mark.parametrize(("A", "B"), [
    (
        Matrix.full((1, 1), 1),
        Matrix([[1]])
    ),
    (
        Matrix.full((3, 2), 5),
        Matrix([[5, 5], [5, 5], [5, 5]])
    ),
    (
        Matrix.full((2, 3), -1),
        Matrix([[-1, -1, -1], [-1, -1, -1]])
    )
])
def test_full(A, B):
    assert A == B


@pytest.mark.parametrize("dimensions", [(-1, 2), (2, -1), (0, 0)])
def test_invalid_full(dimensions):
    with raises(DimensionError):
        Matrix.full(dimensions, 1)


@pytest.mark.parametrize(("A", "B"), [
    (
        Matrix.zeros((1, 1)),
        Matrix([[0]])
    ),
    (
        Matrix.zeros((5, 2)),
        Matrix([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    )
])
def test_zeros(A, B):
    assert A == B
