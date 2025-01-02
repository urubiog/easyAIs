# easyAIs Style Guide

## Code Conventions

### 1. Class and Method Organization

Organize class definitions and methods in the following order:

1. **Class Docstring**: Provide a brief overview of the class.
2. **`__init__`**: Define the initializer method after the class docstring.
3. **Properties**: Include properties immediately after the `__init__` method.
4. **Special Methods (Dunder Methods)**: Include methods like `__repr__`, `__call__`, etc.
5. **Public Methods**: Methods intended for public use should follow the special methods.
6. **Internal Methods (Private)**: Private or internal methods should be placed at the end of the class, prefixed with double underscores (`__`).

#### Example:

```python
class Template(parent):
    """
    Docstring

    Explanation
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the class with necessary parameters.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        # Initialize attributes here
        ...

    @property
    def example_property(self) -> Type:
        """Description of the property."""
        ...

    @example_property.setter
    def example_property(self, value: Type) -> None:
        """Set the property value."""
        ...

    def __repr__(self) -> str:
        """
        Return a string representation of the instance.

        Returns:
            str: A string describing the instance.
        """
        ...

    def __call__(self, *args, **kwargs) -> ReturnType:
        """
        Perform an action when the instance is called.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            ReturnType: The result of the action.
        """
        ...

    def public_method(self, *args, **kwargs) -> ReturnType:
        """Description of a public method."""
        ...

    def another_public_method(self, *args, **kwargs) -> ReturnType:
        """Description of another public method."""
        ...

    def __private_method(self, *args, **kwargs) -> ReturnType:
        """Description of a private/internal method."""
        ...

    def __another_private_method(self, *args, **kwargs) -> ReturnType:
        """Description of another private/internal method."""
...
```

### 2. Docstrings

Each method, property, and class should be documented with a clear and concise docstring. Docstrings should include:

-   **Brief description** of the component's functionality.
-   **Explanation** (for complex structures).
-   **Arguments** (for methods and functions) specifying names and expected types.
-   **Returns** (for methods and functions) specifying the type of value returned.
-   **Exceptions** (if applicable) that the component may raise.

#### Example:

```python
def public_method(self, arg1: Type1, arg2: Type2) -> ReturnType:
    """
    Description of a public method.

    Args:
        arg1 (Type1): Description of the first argument.
        arg2 (Type2): Description of the second argument.

    Returns:
        ReturnType: Description of the return value.

    Raises:
        ExceptionType: Description of any exceptions raised.
    """
    ...
```

### 3. Naming Conventions

-   **Class Names**: Use CamelCase format.
-   **Method and Variable Names**: Use snake_case format.
-   **Constants**: Use uppercase with underscores.
-   **Dunder Methods**: Always preceded and followed by double underscores (`__init__`, `__repr__`, etc.).

#### Example:

```python
class MyTemplate(ABC):
    # Class names in CamelCase

    def __init__(self, *args, **kwargs) -> None:
        # Special method __init__

    def public_method(self, arg1: Type1, arg2: Type2) -> ReturnType:
        ...
        # Public method
```

### 4. Special Methods (Dunder Methods)

Special methods should be defined immediately after properties and before public methods.

#### Example:

```python
    def __repr__(self) -> str:
        """
        Return a string representation of the instance.

        Returns:
            str: A string describing the instance.
        """
        ...
```

### 5. Properties

Properties should be defined immediately after special methods and before public methods.

#### Example:

```python
@property
def example_property(self) -> Type:
    """Description of the property."""
    ...

@example_property.setter
def example_property(self, arg1: Type) -> None:
    if arg1:
        self._example_property = arg1
```

### 6. Public Methods

Public methods that will be used by class users should follow the properties.

#### Example:

```python
    def public_method(self, *args, **kwargs) -> ReturnType:
        """Description of a public method."""
        ...
```

### 7. Internal Methods (Private)

Internal methods should be placed at the end of the class, preceded by double underscores (`__`).

#### Example:

```python
def __private_method(self, *args, **kwargs) -> ReturnType:
    """Description of a private/internal method."""
    ...
```

### 8. Internal Helper Methods (Single Underscore Prefix)

Methods prefixed with a single underscore (`_`) are intended for internal use within the class but are not strictly private. These methods are used to support the core functionality of the class and should:

-   **Be Documented**: Each method should have a docstring that clearly describes its purpose, arguments, and return values.
-   **Not Be Part of the Public API**: They are meant for internal use within the class or module and should not be exposed as part of the public API.
-   **Be Well-Named**: Use descriptive names that indicate their purpose without being overly verbose.

#### Example:

```python
def _helper_method(self, param1: Type1, param2: Type2) -> ReturnType:
    """
    Perform an internal operation to support the class functionality.

    Args:
        param1 (Type1): Description of the first parameter.
        param2 (Type2): Description of the second parameter.

    Returns:
        ReturnType: Description of what the method returns.

    Raises:
        ExceptionType: Description of any exceptions the method might raise.
    """
    # Implementation of the internal helper method
    ...
```

## Additional Guidelines

-   **Import Statements**: Group standard library imports, third-party imports, and local imports separately, with a blank line between each group.
-   **Line Length**: Limit lines to 79 characters to enhance readability.
-   **Formatting**: Adhere to PEP 8 guidelines for formatting, including indentation, spacing, and blank lines.
-   **Error Handling**: Use exception handling judiciously to manage expected errors and avoid silent failures.
