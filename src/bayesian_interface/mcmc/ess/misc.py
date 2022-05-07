import typing

import numpy as np


def check_dimension(
    array: np.ndarray, expected_dims: int | tuple[int, ...]
) -> typing.NoReturn:
    if isinstance(expected_dims, int):
        expected_dims = (expected_dims,)

    if array.ndim not in expected_dims:
        raise ValueError(
            f"Invalid array dimension. Expected one is {expected_dims}, Actual one is {array.ndim}"
        )
