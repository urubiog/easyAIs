from sys import path
from os.path import abspath, dirname, join

# Añadir el directorio src al sys.path
path.append(abspath(join(dirname(__file__), "..", "..", "src")))

from easyAIs.activations import Sigmoid
from easyAIs.layers import Dense, Input
from easyAIs.arquitectures import MLP
from random import randint

iters = 40000
p, q = 2, 1

xor = lambda a, b: int((a and not b) or (not a and b))

X: list[list[int]] = [[randint(0, 1) for _ in range(p)] for _ in range(iters)]
Y: list[list[int]] = [[xor(X[i][0], X[i][1])] for i in range(0, len(X))]

model: MLP = MLP(
    Input(2),
    Dense(6, activation="sigmoid"),
    Dense(1, activation="sigmoid"),
)

model.train(X, Y, loss="binary-cross-entropy", learning_rate=0.1, epochs=2)

print(model.forward([0, 0]))
print(model.forward([0, 1]))
print(model.forward([1, 0]))
print(model.forward([1, 1]))
