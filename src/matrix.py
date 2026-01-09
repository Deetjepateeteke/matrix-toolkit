# -*- coding: utf-8 -*-

"""
matrix.py - Matrix datatype implementation.

This file contains the matrix-class that supports doing mathematical
operations on a 2-dimensional matrix.

Classes:
    Matrix: The matrix datatype.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from __future__ import annotations

from typing import NoReturn, Union

from .utils import _valid_dimensions
from .errors import DimensionError, InvalidOperationError

__all__ = ["Matrix"]


class Matrix:
    """ A 2-dimensional matrix datatype. """

    __slots__ = ["data", "dimensions"]

    def __init__(self, data: list[list[float]]):
        if _valid_dimensions(data):
            self.data = data
            self.dimensions = (len(self.data), len(self.data[0]))

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise InvalidOperationError(f"you can only add matrices, not {type(other).__name__}")

        if self.dimensions != other.dimensions:
            raise DimensionError(
                "to add matrices, they should be of the same dimensions, "
                f"got {self.dimensions} and {other.dimensions}"
            )

        return Matrix([
            [j1 + j2 for j1, j2 in zip(i1, i2)] for i1, i2 in zip(self.data, other.data)
        ])

    def __mul__(self, other):
        """ Scalar multiplication """
        if not isinstance(other, Matrix):
            return Matrix([
                [other * j for j in i] for i in self.data
            ])

        """ Matrix multiplication """
        if self.dimensions[1] != other.dimensions[0]:
            raise DimensionError(
                "to multiplate matrices, they should be of dimensions m x n and n x p, "
                f"got {self.dimensions} and {other.dimensions}"
            )

        m = self.dimensions[0]
        n = self.dimensions[1]
        p = other.dimensions[1]

        # The result after multiplication is a matrix of dimensions m x p.
        result = []

        for i in range(m):
            result.append([])

            for j in range(p):
                own_row = self.data[i]  # The left matrix' ith rown
                other_column = [other.data[c][j] for c in range(n)]  # The right matrix' jth column

                result[i].append(sum(e1 * e2 for e1, e2 in zip(own_row, other_column)))

        return Matrix(result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other) -> Union[Matrix, NoReturn]:
        # Scalar division
        if isinstance(other, (int, float)):
            return Matrix([
                [j / other for j in i] for i in self.data
            ])

        raise NotImplementedError

    def __floordiv__(self, other) -> NoReturn:
        raise NotImplementedError

    def __rtruediv__(self, other) -> NoReturn:
        raise NotImplementedError

    def __rfloordiv__(self, other) -> NoReturn:
        raise NotImplementedError

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise InvalidOperationError(f"you can only subtract matrices, not {type(other).__name__}")

        if self.dimensions != other.dimensions:
            raise DimensionError(
                "to subtract matrices, they should be of the same dimensions, "
                f"got {self.dimensions} and {other.dimensions}"
            )

        return Matrix([
            [j1 - j2 for j1, j2 in zip(i1, i2)] for i1, i2 in zip(self.data, other.data)
        ])

    def __eq__(self, other) -> bool:
        if not isinstance(other, Matrix):
            return False

        # Check for equal dimensions
        elif self.dimensions != other.dimensions:
            return False

        # See if the corresponding elements are equal
        for i1, i2 in zip(self.data, other.data):
            for j1, j2 in zip(i1, i2):

                if j1 != j2:
                    return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"<{self.__qualname__}<{str(self.data)}>"
