# -*- coding: utf-8 -*-

"""
utils.py - Miscellaneous functions for matrixtoolkit.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from typing import Union, NoReturn

from .errors import DimensionError, InvalidElementError

__all__ = ["_is_square", "_valid_dimensions"]


def _valid_dimensions(matrix: list[list[float]]) -> Union[bool, NoReturn]:
    """ Checks for valid matrix dimensions. """

    # Check for a two-dimensional matrix
    if not len(matrix):
        raise DimensionError("a matrix should be two-dimensional")

    for i in range(len(matrix)):

        # Check for a two-dimensional matrix
        if not isinstance(matrix[i], (list, tuple, set)):
            raise DimensionError("a matrix should be two-dimensional")

        if not len(matrix[0]):
            raise DimensionError("a matrix cannot be empty")

        for j in range(len(matrix[i])):

            # Check for a two-dimensional matrix
            if isinstance(matrix[i][j], (list, tuple, set)):
                raise DimensionError("a matrix should be two-dimensional")

            # Check for non-numeric elements
            elif not isinstance(matrix[i][j], (int, float)) or isinstance(matrix[i][j], bool):
                raise InvalidElementError(elem=matrix[i][j], index=(i, j))

        # Check for dimension ambiguity
        if len(matrix[i]) != len(matrix[min(i+1, len(matrix)-1)]):
            raise DimensionError("there is ambiguity in the given matrix' dimesions")

    return True


def _is_square(dimensions: tuple[int]) -> bool:
    # Check if the matrix is a square matrix
    return dimensions[0] == dimensions[1]
