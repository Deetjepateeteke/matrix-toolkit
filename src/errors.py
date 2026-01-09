# -*- coding: utf-8 -*-

"""
errors.py - Custom matrixtoolkit errors.

Usage:
    >>> try:
    >>>     ...
    >>> except MatrixError:  # Catches all errors raised by matrixtoolkit
    >>>     ...

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from typing import Any


__all__ = [
    "DimensionError",
    "InvalidElementError",
    "InvalidOperationError",
    "MatrixError",
    "NotSquareError"
]

class MatrixError(Exception): ...  # The master matrixlib exception

class DimensionError(MatrixError):
    def __init__(self, msg: str):
        super().__init__(msg)

class NotSquareError(MatrixError):
    def __init__(self, msg: str):
        super().__init__(msg)

class InvalidElementError(MatrixError):
    def __init__(self, elem: Any, index: tuple[int]):
        msg = f"all matrix' elements should be numeric, not {type(elem)} at index ({index[0]}, {index[1]})"
        super().__init__(msg)

class InvalidOperationError(MatrixError):
    def __init__(self, msg: str):
        super().__init__(msg)
