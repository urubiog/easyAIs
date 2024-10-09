from sys import path
from os.path import abspath, dirname, join

# Añadir el directorio src al sys.path
path.append(abspath(join(dirname(__file__), '..', '..', 'src')))

# test_loss.py 
import pytest
from math import log10, log
from easyAIs.loss import (
    MeanSquaredError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    HuberLoss,
    KLDivergence,
    loss_map,
    LossFunction  # Asegúrate de importar esta clase si existe en tu implementación
)

# Fixtures para las pruebas
@pytest.fixture
def mse_data():
    return {
        "tags": [1, 2, 3],
        "pred": [1.1, 2.1, 2.9]
    }

@pytest.fixture
def bce_data():
    return {
        "y_true": [1, 0],
        "y_pred": [0.8, 0.2]
    }

@pytest.fixture
def cce_data():
    return {
        "y_true": [1, 0, 0],
        "y_pred": [0.7, 0.2, 0.1]
    }

@pytest.fixture
def huber_data():
    return {
        "y_true": [1, 2, 3],
        "y_pred": [1.1, 2.1, 2.9],
        "delta": 1.0
    }

@pytest.fixture
def kl_data():
    return {
        "p": [0.4, 0.6],
        "q": [0.5, 0.5]
    }

# Pruebas para Mean Squared Error
def test_mean_squared_error(mse_data):
    loss = MeanSquaredError.__call__(**mse_data)
    expected_loss = sum([(t - p) ** 2 for t, p in zip(mse_data["tags"], mse_data["pred"])]) / len(mse_data["tags"])
    assert pytest.approx(loss, 0.001) == expected_loss

def test_mean_squared_error_deriv(mse_data):
    grad = MeanSquaredError.deriv(**mse_data)
    expected_grad = [(2 * (p - t)) / len(mse_data["tags"]) for t, p in zip(mse_data["tags"], mse_data["pred"])]
    assert grad == pytest.approx(expected_grad, 0.001)

# Pruebas para Binary Cross-Entropy
def test_binary_cross_entropy(bce_data):
    loss = BinaryCrossEntropy.__call__(**bce_data)
    epsilon = 1e-15
    y_pred = [max(epsilon, min(1 - epsilon, p)) for p in bce_data["y_pred"]]
    expected_loss = -sum(
        y_t * log10(y_p) + (1 - y_t) * log10(1 - y_p)
        for y_t, y_p in zip(bce_data["y_true"], y_pred)
    ) / len(bce_data["y_true"])
    assert pytest.approx(loss, 0.001) == expected_loss

def test_binary_cross_entropy_deriv(bce_data):
    grad = BinaryCrossEntropy.deriv(**bce_data)
    epsilon = 1e-15
    y_pred = [max(epsilon, min(1 - epsilon, p)) for p in bce_data["y_pred"]]
    expected_grad = [(y_p - y_t) / (y_p * (1 - y_p)) for y_t, y_p in zip(bce_data["y_true"], y_pred)]
    assert grad == pytest.approx(expected_grad, 0.001)

# Pruebas para Categorical Cross-Entropy
def test_categorical_cross_entropy(cce_data):
    loss = CategoricalCrossEntropy.__call__(**cce_data)
    expected_loss = -sum(true * log(pred) for true, pred in zip(cce_data["y_true"], cce_data["y_pred"]))
    assert pytest.approx(loss, 0.001) == expected_loss

def test_categorical_cross_entropy_deriv(cce_data):
    grad = CategoricalCrossEntropy.deriv(**cce_data)
    expected_grad = [-true / pred for true, pred in zip(cce_data["y_true"], cce_data["y_pred"])]
    assert grad == pytest.approx(expected_grad, 0.001)

# Pruebas para Huber Loss
def test_huber_loss(huber_data):
    loss = HuberLoss.__call__(**huber_data)
    delta = huber_data["delta"]
    expected_loss = sum(
        0.5 * abs(t - p)**2 if abs(t - p) <= delta
        else delta * (abs(t - p) - 0.5 * delta)
        for t, p in zip(huber_data["y_true"], huber_data["y_pred"])
    ) / len(huber_data["y_true"])
    assert pytest.approx(loss, 0.001) == expected_loss

def test_huber_loss_deriv(huber_data):
    grad = HuberLoss.deriv(**huber_data)
    delta = huber_data["delta"]
    expected_grad = [
        -error if abs(error) <= delta
        else -delta * (1 if error > 0 else -1)
        for error in (t - p for t, p in zip(huber_data["y_true"], huber_data["y_pred"]))
    ]
    assert grad == pytest.approx(expected_grad, 0.001)

# Pruebas para KL Divergence
def test_kl_divergence(kl_data):
    divergence = KLDivergence.__call__(**kl_data)
    epsilon = 1e-15
    q = [max(epsilon, min(1 - epsilon, prob)) for prob in kl_data["q"]]
    expected_divergence = sum(p_i * log(p_i / q_i) for p_i, q_i in zip(kl_data["p"], q) if p_i != 0)
    assert pytest.approx(divergence, 0.001) == expected_divergence

def test_kl_divergence_deriv(kl_data):
    grad = KLDivergence.deriv(**kl_data)
    epsilon = 1e-15
    q = [max(epsilon, min(1 - epsilon, prob)) for prob in kl_data["q"]]
    expected_grad = [-(p_i / q_i) for p_i, q_i in zip(kl_data["p"], q)]
    assert grad == pytest.approx(expected_grad, 0.001)

# Pruebas para el mapa de funciones de pérdida
def test_loss_map():
    for name, loss_class in loss_map.items():
        assert issubclass(loss_class, LossFunction), f"{name} should be a subclass of LossFunction"

if __name__ == "__main__":
    pytest.main()
