from sys import path
from os.path import abspath, dirname, join

# Añadir el directorio src al sys.path
path.append(abspath(join(dirname(__file__), "..", "..", "src")))

from easyAIs.layers import Dense, Input
from easyAIs.preprocessing.datasets import mnist

train_images = mnist.train_images()
train_labels = mnist.train_labels()

train_images_vectorized = []

for i in range(len(train_images)):
    img = train_images[i]
    vector = []

    for row in img:
        vector = vector.__add__(row)
    
    train_images_vectorized.append(vector)

classes = len(set(train_labels))
train_labels_encoded = []

for i in range(len(train_labels)):
    label = train_labels[i]

    row = [0] * (classes - 1)
    row.insert(label, 1)

    train_labels_encoded.append(row)

from easyAIs.arquitectures import MLP

model = MLP(Input(784), Dense(40, activation="sigmoid"), Dense(10, activation="sigmoid"))

model.train(train_images_vectorized, train_labels_encoded, learning_rate=0.001, loss="cross-entropy", epochs=1, verbose=True)
