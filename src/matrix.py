# -*- coding: utf-8 -*-

"""
matrix.py - Matrix datatype implementation.

This file contains the matrix-class that supports doing mathematical
operations on a 2-dimensional matrix.

Classes:
    Matrix: The 2-dimensional matrix datatype.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from __future__ import annotations

from typing import NoReturn, Union

from .utils import _is_square, _valid_dimensions
from .errors import DimensionError, InvalidOperationError, NotSquareError

__all__ = ["Matrix"]


class Matrix:
    """ A 2-dimensional matrix datatype. """

    __slots__ = ["_data", "_dimensions"]

    def __init__(self, data: list[list[float]]):
        if _valid_dimensions(data):
            self._data = data
            self._dimensions = (len(data), len(data[0]))

    @classmethod
    def identity(cls, n: int) -> Matrix:
        """
        Create a n x n matrix that is filled with 1s along the main diagonal.

        Args:
            n (int): The dimensions (n, n) of the new matrix.

        Returns:
            Matrix: an n x n identity matrix
        """
        return Matrix([
            [1 if i == j else 0 for j in range(n)] for i in range(n)
        ])

    @classmethod
    def full(cls, dimensions: tuple[int], value: Union[int, float]) -> Matrix:
        """
        Create a m x n matrix that is filled with the given value.

        Args:
            dimensions (tuple[int]): The dimensions (m x n) of the new matrix.
            value (Union[int, float]): The value that the matrix should be filled with.

        Returns:
            Matrix: The filled matrix.
        """
        return Matrix([
            [value for j in range(dimensions[1])] for i in range(dimensions[0])
        ])

    @classmethod
    def zeros(cls, dimensions: tuple[int]) -> Matrix:
        """
        Create a m x n matrix that is filled with zeros.

        Args:
            dimensions (tuple[int]): The dimensions (m x n) of the new matrix.

        Returns:
            Matrix: The m x n zero matrix.
        """
        return cls.full(dimensions, 0)

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
    
    def __pow__(self, n: int):
        if not isinstance(n, int):
            raise InvalidOperationError(f"a matrix' power must be an integer, got {n!r}")

        # Check for a square matrix
        if not _is_square(self.dimensions):
            raise NotSquareError(f"to raise a matrix to a power, the matrix must be a squared matrix, got {self.dimensions}")
        
        # Negative powers
        if n < 0:
            raise NotImplementedError
        
        # Return the identity matrix when n is 0
        elif n == 0:
            return self.identity(self.dimensions[0])
        
        result = Matrix(self.data)  # make a copy of the matrix
        for _ in range(n-1):
            result *= self

        return result

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

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"{type(self).__name__}(dims={self.dimensions}, {self.data})"

    def transpose(self):
        """
        Returns the transposed of a matrix. The transposed
         matrix will be of dimensions n x m.

        returns:
            Matrix: the transposed matrix
        """
        n = self.dimensions[1]
        m = self.dimensions[0]

        return Matrix([
            [self.data[j][i] for j in range(m)] for i in range(n)
        ])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, _):
        raise AttributeError("Matrix.data cannot be modified")

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, _):
        raise AttributeError("Matrix.dimensions cannot be modified")
