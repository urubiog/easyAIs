# test_neural_components.py
from sys import path
from os.path import abspath, dirname, join

# Añadir el directorio src al sys.path
path.append(abspath(join(dirname(__file__), '..', '..', 'src')))

import pytest
from easyAIs.core.neural_components import Node, Neuron, RecNeuron
from easyAIs.utils.verifiers import verify_type
from easyAIs.activations import NoneFunction
from typing import Callable

@pytest.fixture
def default_node():
    return Node(5)

@pytest.fixture
def default_neuron():
    return Neuron()

@pytest.fixture
def node_list():
    return [Node(1), Node(2)]

@pytest.fixture
def neuron_with_nodes(node_list):
    neuron = Neuron()
    neuron.inputnodes = node_list
    return neuron

@pytest.fixture
def dummy_activation() -> Callable[[float], float]:
    return lambda x: x  # Identity function as a dummy activation

# Pruebas para la clase Node
class TestNode:
    
    def test_node_initialization(self, default_node):
        assert isinstance(default_node, Node)
        assert default_node.output == 5

    def test_node_initialization_default(self):
        node = Node()
        assert isinstance(node, Node)
        assert node.output == 0

    def test_node_value_validation(self):
        with pytest.raises(TypeError):
            Node("invalid_value")

    def test_node_str_repr(self, default_node):
        assert str(default_node) == f"Node(): {default_node._id}"
        assert repr(default_node) == f"Node(): {default_node._id}"

# Pruebas para la clase Neuron
class TestNeuron:
    
    def test_neuron_initialization(self):
        neuron = Neuron()
        assert isinstance(neuron, Neuron)
        assert neuron.bias >= 0
        assert neuron.weights == []
        assert neuron.activation == NoneFunction

    def test_neuron_initialization_with_activation(self, dummy_activation):
        neuron = Neuron(activation=dummy_activation)
        assert neuron.activation == dummy_activation

    def test_neuron_set_inputnodes(self, node_list):
        neuron = Neuron()
        neuron.inputnodes = node_list
        assert neuron.inputnodes == node_list
        assert len(neuron.weights) == len(node_list)

    def test_neuron_set_inputnodes_invalid(self):
        neuron = Neuron()
        with pytest.raises(TypeError):
            neuron.inputnodes = [1, 2]

    def test_neuron_z_property(self, neuron_with_nodes):
        neuron = neuron_with_nodes
        neuron.weights = [0.5, 0.5]
        neuron.bias = 0.0
        assert neuron.z == (1 * 0.5 + 2 * 0.5)

    def test_neuron_output_property(self, neuron_with_nodes, dummy_activation):
        neuron = neuron_with_nodes
        neuron.weights = [0.5, 0.5]
        neuron.bias = 0.0
        assert neuron.output == dummy_activation(neuron.z)

    def test_neuron_iter(self, neuron_with_nodes):
        neuron = neuron_with_nodes
        iterator = iter(neuron)
        assert next(iterator) == neuron.inputnodes[0]
        assert next(iterator) == neuron.inputnodes[1]
        with pytest.raises(StopIteration):
            next(iterator)

    def test_neuron_len(self, neuron_with_nodes):
        neuron = neuron_with_nodes
        assert len(neuron) == 2

    def test_neuron_hash_eq(self, neuron_with_nodes):
        neuron1 = neuron_with_nodes
        neuron2 = Neuron()
        neuron2.inputnodes = neuron1.inputnodes
        neuron2.weights = neuron1.weights
        neuron2.bias = neuron1.bias
        assert neuron1 == neuron2
        assert hash(neuron1) == hash(neuron2)

# Pruebas para la clase RecNeuron
# class TestRecNeuron:
    
#     def test_rec_neuron_not_implemented(self, dummy_activation):
#         with pytest.raises(NotImplementedError):
#             RecNeuron(dummy_activation, 10)

#     def test_rec_neuron_iterations_validation(self):
#         with pytest.raises(ValueError):
#             RecNeuron(NoneFunction, -1)

#     @pytest.mark.parametrize("iterations", [1, 10, 100])
#     def test_rec_neuron_iterations_parametrized(self, iterations):
#         if iterations < 1:
#             with pytest.raises(ValueError):
#                 RecNeuron(NoneFunction, iterations)
#         else:
#             rec_neuron = RecNeuron(NoneFunction, iterations)
#             assert rec_neuron._iterations == iterations

# Ejecutar las pruebas
if __name__ == "__main__":
    pytest.main()


# Ejecutar las pruebas
if __name__ == "__main__":
    pytest.main()
