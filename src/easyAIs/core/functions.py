"""
Module: easyAIs.core.functions

This module defines a set of base and example functions that demonstrate how to implement mathematical functions using metaclasses. It includes a framework for creating callable functions and their derivatives using static methods. 

Classes:
    - `Function` (metaclass):
        A metaclass that enables the classes to be called as functions. It delegates the call to the static method `__call__` of the class.

    - `SquareFunction`:
        An example class demonstrating the use of the `Function` metaclass. It implements a callable static method to compute the square of a number and another static method to compute its derivative.
        
        Methods:
            - `__call__(x: Union[int, float]) -> float`:
                Computes the square of the input `x`. This method is called when an instance of `SquareFunction` is invoked as if it were a function.
                
            - `deriv(x: Union[int, float]) -> float`:
                Computes the derivative of the function `x^2` at the point `x`. This method provides the gradient of the square function.

Notes:
    - The `Function` metaclass is designed to be extended for various mathematical operations. To create new functions, define a class with `Function` as its metaclass and implement the `__call__` and `deriv` static methods.
    - Ensure that the `__call__` method performs the primary computation, while the `deriv` method calculates the derivative of the function.
    - This module can be extended to include more complex functions and their derivatives by creating new classes following the `Function` metaclass structure.

Potential Extensions:
    - Add more example functions, such as trigonometric, exponential, or logarithmic functions.
    - Implement additional utility functions to handle more complex mathematical operations or optimizations.
    - Provide mechanisms for automatic differentiation to compute derivatives for a wider range of functions.
"""


class Function(type):
    def __call__(cls, *args, **kwargs):
        # When calling the class, delegate the call to the static __call__ method
        return cls.__call__(*args, **kwargs)


if __name__ == "__main__":
    # Example
    class SquareFunction(metaclass=Function):
        @staticmethod
        def __call__(x: int | float) -> float:
            """Devuelve el cuadrado de x cuando se llama a la clase."""
            return x**2

        @staticmethod
        def deriv(x: int | float) -> float:
            """Devuelve la derivada de la función x^2 en x, que es 2*x."""
            return 2 * x
