"""
Module: easyAIs.utils.instances

This module provides utility functions for managing and interacting with object instances. It includes functionalities for searching instance names within reference chains and for creating deep copies of instances, thus facilitating effective instance handling and duplication.

Functions:
- `search_instnce_name(inst: object) -> str`: Searches for the name of an instance within its reference chains.
- `clone_instance(inst: object) -> object`: Creates and returns a deep copy of the given instance.

Usage:
```python
from easyAIs.utils.instances import search_instnce_name, clone_instance

name = search_instnce_name(my_instance)
cloned_instance = clone_instance(my_instance)
"""

from gc import get_referrers
from copy import deepcopy

def search_instnce_name(inst: object) -> str:
    for referrer in get_referrers(inst):
        if isinstance(referrer, dict):
            for name, val in referrer.items():
                if val is inst:
                    return name 

    return "[Imposible to locate instance name]"

def clone_instance(inst: object) -> object:
    return deepcopy(inst)
