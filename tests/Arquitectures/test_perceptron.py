from sys import path
from os.path import abspath, dirname, join

# Añadir el directorio src al sys.path
path.append(abspath(join(dirname(__file__), "..", "..", "src")))

from easyAIs.arquitectures import Perceptron
from random import randint

# NOT gate
X: list = [[randint(0, 1) for _ in range(1)] for _ in range(200)]

not_y: list = [int(not bool(X[i][0])) for i in range(len(X))]

not_g = Perceptron(1)

not_g.train(X, not_y, epochs=20)

print(
    f"""NOT:
[0]: {not_g([0])}
[1]: {not_g([1])}
"""
)

ENTRIES: int = 2
X: list = [[randint(0, 1) for _ in range(ENTRIES)] for _ in range(200)]

# AND gate
and_y: list = [int(bool(X[i][0]) and bool(X[i][1])) for i in range(0, len(X))]

and_g = Perceptron(ENTRIES)

and_g.train(X, and_y)

print(
    f"""AND:
[0, 0]: {and_g([0,0])}
[0, 1]: {and_g([0,1])}
[1, 0]: {and_g([1,0])}
[1, 1]: {and_g([1,1])}
"""
)

# The ideal config would be something like this
"""
and_g._weights = [1 , 1]
and_g._bias = -2

print(and_g([1,1])) # 1
print(and_g([0,0])) # 0
print(and_g([1,0])) # 0
"""

# OR gate
or_y: list = [int(bool(X[i][0]) or bool(X[i][1])) for i in range(0, len(X))]

or_g = Perceptron(ENTRIES)

or_g.train(X, or_y)

print(
    f"""OR:
[0, 0]: {or_g([0,0])}
[0, 1]: {or_g([0,1])}
[1, 0]: {or_g([1,0])}
[1, 1]: {or_g([1,1])}
"""
)

# NAND gate
nand_y: list = [
    int(not (bool(X[i][0]) and bool(X[i][1]))) for i in range(0, len(X))
]

nand_g = Perceptron(ENTRIES)

nand_g.train(X, nand_y)

print(
    f"""NAND:
[0, 0]: {nand_g([0,0])}
[0, 1]: {nand_g([0,1])}
[1, 0]: {nand_g([1,0])}
[1, 1]: {nand_g([1,1])}
"""
)

# XOR gate
xor_y: list = [int(bool(X[i][0]) ^ bool(X[i][0])) for i in range(0, len(X))]

xor_g = Perceptron(ENTRIES)

xor_g.train(X, xor_y)

print(
    f"""XOR:
[0, 0]: {xor_g([0,0])}
[0, 1]: {xor_g([0,1])}
[1, 0]: {xor_g([1,0])}
[1, 1]: {xor_g([1,1])}
"""
)
