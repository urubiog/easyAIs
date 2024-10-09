from sys import path
from os.path import abspath, dirname, join

# Añadir el directorio src al sys.path
path.append(abspath(join(dirname(__file__), '..', '..', 'src')))

# test_activations.py

import pytest
from easyAIs.activations import (
    Step, Sigmoid, ReLU, LeakyReLU, Tanh, Softmax, PReLU, ELU, Softplus, NoneFunction
)
from math import exp, log

@pytest.fixture
def activation_values():
    return {
        "step": (Step, 1, 0, 0, 1),
        "sigmoid": (Sigmoid, 0.5),
        "relu": (ReLU, 0),
        "leaky_relu": (LeakyReLU, -1, -0.1),
        "tanh": (Tanh, 0),
        "softmax": (Softmax, [1, 2, 3], 2),
        "prelu": (PReLU, -1, 0.1),
        "elu": (ELU, -1, 1.0),
        "softplus": (Softplus, 0),
        "none": (NoneFunction, 0)
    }

# Pruebas para la clase Step
class TestStep:
    
    def test_step(self, activation_values):
        Step_class, test_value, threshold, _, expected = activation_values["step"]
        assert Step_class(test_value, threshold) == expected

    def test_step_deriv(self):
        assert Step.deriv(0) == 0

# Pruebas para la clase Sigmoid
class TestSigmoid:
    
    def test_sigmoid(self, activation_values):
        Sigmoid_class, test_value = activation_values["sigmoid"]
        expected = 1 / (1 + exp(-test_value))
        assert Sigmoid_class(test_value) == pytest.approx(expected, rel=1e-5)

    def test_sigmoid_deriv(self, activation_values):
        Sigmoid_class, test_value = activation_values["sigmoid"]
        sig = Sigmoid_class(test_value)
        expected_derivative = sig * (1 - sig)
        assert Sigmoid_class.deriv(test_value) == pytest.approx(expected_derivative, rel=1e-5)

# Pruebas para la clase ReLU
class TestReLU:
    
    def test_relu(self, activation_values):
        ReLU_class, test_value = activation_values["relu"]
        assert ReLU_class(test_value) == max(0, test_value)

    def test_relu_deriv(self, activation_values):
        ReLU_class, test_value = activation_values["relu"]
        assert ReLU_class.deriv(test_value) == (1 if test_value > 0 else 0)

# Pruebas para la clase LeakyReLU
class TestLeakyReLU:
    
    def test_leaky_relu(self, activation_values):
        LeakyReLU_class, test_value, expected = activation_values["leaky_relu"]
        assert LeakyReLU_class(test_value) == (test_value if test_value >= 0 else expected)

    def test_leaky_relu_deriv(self, activation_values):
        LeakyReLU_class, test_value, _ = activation_values["leaky_relu"]
        assert LeakyReLU_class.deriv(test_value) == (1 if test_value >= 0 else 0.1)

# Pruebas para la clase Tanh
class TestTanh:
    
    def test_tanh(self, activation_values):
        Tanh_class, test_value = activation_values["tanh"]
        expected = (exp(test_value) - exp(-test_value)) / (exp(test_value) + exp(-test_value))
        assert Tanh_class(test_value) == pytest.approx(expected, rel=1e-5)

    def test_tanh_deriv(self, activation_values):
        Tanh_class, test_value = activation_values["tanh"]
        tanh_x = Tanh_class(test_value)
        expected_derivative = 1 - tanh_x**2
        assert Tanh_class.deriv(test_value) == pytest.approx(expected_derivative, rel=1e-5)

# Pruebas para la clase Softmax
class TestSoftmax:
    
    def test_softmax(self, activation_values):
        Softmax_class, test_values, index = activation_values["softmax"]
        softmax_values = [exp(i) / sum(exp(j) for j in test_values) for i in test_values]
        assert Softmax_class(test_values, index) == [softmax_values[index]]

    def test_softmax_deriv(self):
        # El cálculo exacto del Jacobiano no se implementa aquí.
        assert Softmax.deriv([1, 2, 3]) == []

# Pruebas para la clase PReLU
class TestPReLU:
    
    def test_prelu(self, activation_values):
        PReLU_class, test_value, lp = activation_values["prelu"]
        assert PReLU_class(test_value, lp) == max(lp * test_value, test_value)

    def test_prelu_deriv(self, activation_values):
        PReLU_class, test_value, lp = activation_values["prelu"]
        assert PReLU_class.deriv(test_value, lp) == (lp if test_value < 0 else 1)

# Pruebas para la clase ELU
class TestELU:
    
    def test_elu(self, activation_values):
        ELU_class, test_value, alpha = activation_values["elu"]
        assert ELU_class(test_value, alpha) == (test_value if test_value > 0 else alpha * (exp(test_value) - 1))

    def test_elu_deriv(self, activation_values):
        ELU_class, test_value, alpha = activation_values["elu"]
        assert ELU_class.deriv(test_value, alpha) == (1 if test_value > 0 else alpha * exp(test_value))

# Pruebas para la clase Softplus
class TestSoftplus:
    
    def test_softplus(self, activation_values):
        Softplus_class, test_value = activation_values["softplus"]
        assert Softplus_class(test_value) == log(1 + exp(test_value))

    def test_softplus_deriv(self, activation_values):
        Softplus_class, test_value = activation_values["softplus"]
        assert Softplus_class.deriv(test_value) == Sigmoid(test_value)

# Pruebas para la clase NoneFunction
class TestNoneFunction:
    
    def test_none_function(self, activation_values):
        NoneFunction_class, test_value = activation_values["none"]
        assert NoneFunction_class(test_value) == float(test_value)

    def test_none_function_deriv(self, activation_values):
        assert NoneFunction.deriv(0) == 1

# Ejecutar las pruebas
if __name__ == "__main__":
    pytest.main()
