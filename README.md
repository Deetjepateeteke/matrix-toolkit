# Matrix-toolkit

Matrix-toolkit is a Python add-on that introduces a 2-dimensional matrix datatype to the Python ecosystem, with native support for mathematical operations.

## Install

To clone the repository:

```bash
git clone https://github.com/Deetjepateeteke/matrix-toolkit.git
cd matrixtoolkit
```

## Features

- 2D matrix datatype
- Operator overloading (`+`, `-`, `*`, `/`, `@`, `**`, etc.)
- Matrix creation utilities (identity, zero, etc.)
- Pure Python, no external dependencies

## Usage

```python
from matrixtoolkit import Matrix

A = Matrix([[1, 0], [0, 1]])

A.dimensions  # (2, 2)

T = A.transpose()  # Compute a matrix' transpose
```

## Supported Operations

Matrix-toolkit has native support for mathematical operations.

| Operation | Description           |
|:---------:|:---------------------:|
| `A + B`   | Matrix addition       |
| `A - B`   | Matrix subtraction    |
| `n * A`   | Scalar multiplication |
| `A / n`   | Scalar division       |
| `A @ B`   | Matrix multiplication |
| `A ** n`  | Matrix power          |
| `A == B`  | Matrix equality       |
| `A != B`  | Matrix inequality     |

## Matrix Creation

To create matrices, use the following methods:

| Method                         | Description                                        |
|:-------------------------------|:---------------------------------------------------|
| Matrix(data)                   | Create a custom matrix.                            |
| Matrix.identity(n)             | Create a n x n identity matrix.                    |
| Matrix.zeros(dimensions)       | Create a m x n matrix filled with zeros.           |
| Matrix.full(value, dimensions) | Create a m x n matrix filled with the given value. |

## Links

- Source Code: https://github.com/Deetjepateeteke/matrix-toolkit/

## License

This project is licensed under the [MIT License](LICENSE)
