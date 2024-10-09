"""
Module: easyAIs.utils.verifiers

This module provides utility functions for validating and verifying object properties and types. It includes checks for type conformity, length, indexability, iterability, and component types within compound objects.

Functions:
- `verify_type(obj: Any, t: Union[type, Tuple]) -> Any`: Ensures that the object is an instance of the specified type(s). Raises a TypeError if the check fails.
- `verify_len(obj: Sized, n: int) -> Any`: Confirms that the length of the object matches the expected length. Raises an IndexError if the length is incorrect.
- `verify_indexable(obj: object) -> Any`: Checks if the object supports indexing. Raises an IndexError if the object is not indexable.
- `verify_iterable(obj: Iterable) -> Any`: Validates if the object is iterable. Raises a TypeError if the object is not iterable.
- `verify_components_type(obj: Compound, etype: Union[type, Tuple[type, ...]]) -> Any`: Verifies that all components of a compound object are of the expected type(s). Raises a TypeError if any component is of an incorrect type.

Usage:
```python
from easyAIs.utils.verifiers import verify_type, verify_len, verify_indexable, verify_iterable, verify_components_type

obj = verify_type(some_object, int)
obj = verify_len(some_object, 5)
obj = verify_indexable(some_object)
obj = verify_iterable(some_object)
obj = verify_components_type(some_compound, (int, str))
"""

from typing import Any, Iterable, Sized, Tuple, Union, Protocol


class Compound(Protocol):
    """
    A protocol representing a compound object that must support
    indexing and length retrieval.

    Protocol methods:
        - __len__: Returns the length of the compound object.
        - __getitem__: Retrieves an item by index.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


def verify_type(obj: Any, t: Union[type, Tuple]) -> Any:
    """
    Verifies if the provided object is an instance of the specified type(s).

    Args:
        obj (Any): The object to check.
        t (Union[type, Tuple]): The expected type or types. If a tuple is provided,
                                the object must be an instance of any type in the tuple.

    Raises:
        TypeError: If the object is not of the expected type.

    Returns:
        Any: The original object if the type check passes.
    """
    if not isinstance(obj, t):
        raise TypeError(f"Expected {obj} to be {t}, got type {type(obj)}.")

    return obj


def verify_len(obj: Sized, n: int) -> Any:
    """
    Verifies that the length of the given object matches the expected length.

    Args:
        obj (Sized): The object whose length is to be verified.
        n (int): The expected length of the object.

    Raises:
        IndexError: If the object’s length does not match the expected length.

    Returns:
        Any: The original object if the length check passes.
    """
    if hasattr(obj, "__len__"):
        if len(obj) != n:
            raise IndexError(
                f"Expected {obj} to be {n} in length, got length {len(obj)}."
            )

    return obj


def verify_indexable(obj: object) -> Any:
    """
    Checks if the provided object supports indexing (i.e., has a __getitem__ method).

    Args:
        obj (Any): The object to check.

    Returns:
        Any: The original object if it is indexable.

    Raises:
        IndexError: If the object is not indexable.
    """
    if not hasattr(obj, "__getitem__") or not callable(getattr(obj, "__getitem__")):
        raise IndexError("Object is not indexable.")

    return obj


def verify_iterable(obj: Iterable) -> Any:
    """
    Verifies if the provided object is iterable.

    Args:
        obj (Iterable): The object to check.

    Returns:
        Any: The original object if it is iterable.

    Raises:
        TypeError: If the object is not iterable.
    """
    try:
        iter(obj)
    except TypeError:
        raise TypeError(f"The object of type {type(obj).__name__} is not iterable")

    return obj


def verify_components_type(obj: Compound, etype: Union[type, Tuple[type, ...]]) -> Any:
    """
    Checks if all components of a compound object are of the expected type(s).

    Args:
        obj (Compound): The compound object whose components are to be checked.
        etype (Union[type, Tuple[type, ...]]): The expected type or types of the components.

    Returns:
        Any: The original object if all components match the expected type(s).

    Raises:
        TypeError: If any component is not of the expected type(s).
    """
    for i in range(len(obj)):
        if not isinstance(obj[i], etype):
            raise TypeError(
                f"Expected all components of {obj} to be of type {etype}, "
                f"but component at position {i} is of type {type(obj[i])} with value {obj[i]}."
            )

    return obj
