
from sys import path
from os.path import abspath, dirname, join

# Añadir el directorio src al sys.path
path.append(abspath(join(dirname(__file__), '..', '..', 'src')))

import pytest
from easyAIs.core.models import Model
from easyAIs.core.layer import Layer
from easyAIs.core.neural_components import Neuron, Node
from easyAIs.core.optimizers import Optimizer
from easyAIs.loss import LossFunction

class DummyLayer(Layer):
    def __init__(self, output):
        self._output = output

    def forward(self, input):
        return self._output

class DummyModel(Model):
    def __init__(self, layers):
        self._layers = layers

    def forward(self, input):
        for layer in self._layers:
            input = layer.forward(input)
        return input

    def train(self):
        pass

    def evaluate(self):
        return "evaluation"

    def save(self):
        return "model saved"

    def load(self):
        return "model loaded"

    def summary(self):
        return "model summary"

@pytest.fixture
def dummy_layers():
    return [DummyLayer([1, 2, 3]), DummyLayer([4, 5, 6])]

@pytest.fixture
def dummy_model(dummy_layers):
    return DummyModel(dummy_layers)

def test_model_forward(dummy_model):
    input_data = [0, 0, 0]
    output = dummy_model.forward(input_data)
    assert output == [4, 5, 6]

def test_model_evaluate(dummy_model):
    assert dummy_model.evaluate() == "evaluation"

def test_model_save(dummy_model):
    assert dummy_model.save() == "model saved"

def test_model_load(dummy_model):
    assert dummy_model.load() == "model loaded"

def test_model_summary(dummy_model):
    assert dummy_model.summary() == "model summary"

def test_model_equality(dummy_model, dummy_layers):
    another_model = DummyModel(dummy_layers)
    assert dummy_model == another_model

def test_model_inequality(dummy_model):
    different_layers = [DummyLayer([7, 8, 9])]
    another_model = DummyModel(different_layers)
    assert dummy_model != another_model

def test_model_hash(dummy_model, dummy_layers):
    another_model = DummyModel(dummy_layers)
    assert hash(dummy_model) == hash(another_model)

def test_model_call(dummy_model):
    input_data = [0, 0, 0]
    output = dummy_model(input_data)
    assert output == [4, 5, 6]