***Copyright (c) 2024 Uriel Rubio García | easyAIs. All Rights Reserved.***

## Building Logic Gates using the Perceptron
*by: [Uriel Rubio](https://github.com/urubiog)*

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://github.com/urubiog/easyAIs/blob/main/examples/"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

## Approach
A perceptron is a fundamental unit in neural networks that can be trained to perform binary classification tasks. In this context, we will explore how a single-layer perceptron can be trained to imitate basic logic gates, such as AND and OR.

### The Perceptron
A perceptron is a type of artificial neuron used in machine learning for binary classification tasks. It consists of:

- **Inputs**: Features or variables $x_1, x_2, \ldots, x_n$
- **Weights**: Parameters $w_1, w_2, \ldots, w_n$ associated with each input
- **Bias**: A constant $b$ added to the weighted sum
- **Activation Function**: Determines the output based on the weighted sum

<div>
<img src='/mnt/c/Users/Mi Pc/Downloads/perceptron.png' width="400"/>
</div>

The output $y$ of a perceptron is computed as:

$$y = H\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

where $H$ is the activation function, defined as a step function for binary classification:

$$H(t) = 
\begin{cases} 
1 & \text{if } t \geq 0 \\
0 & \text{if } t < 0 
\end{cases}$$

## Setup

```python
from easyAIs.arquitectures import Perceptron
from random import randint
```

## Content 

```python
# Let's define the model arquitecture
ENTRIES: int = 2  # Input nodes

# Generating the binary data
X: list[int] = [randint(0, 1) for _ in range(200)]
```
#### AND gate 

```python 
y: list[int] = [int(bool(X[i]) and bool(X[i + 1])) for i in range(0, len(X) - 1, 2)]

p = Perceptron(ENTRIES)

p.fit(X, y, verbose=True)

print(f"""AND:
[0, 0]: {and_g([0,0])}
[0, 1]: {and_g([0,1])}
[1, 0]: {and_g([1,0])}
[1, 1]: {and_g([1,1])}
""")
```

#### OR gate 

```python 
y: list = [int(bool(X[i]) or bool(X[i + 1])) for i in range(0, len(X) - 1, 2)]

p = Perceptron(ENTRIES)

p.fit(X, y)

print(f"""OR:
[0, 0]: {or_g([0,0])}
[0, 1]: {or_g([0,1])}
[1, 0]: {or_g([1,0])}
[1, 1]: {or_g([1,1])}
""")
```

#### NAND gate 

```python 
y: list = [int(not (bool(X[i]) and bool(X[i + 1]))) for i in range(0, len(X) - 1, 2)]

p = Perceptron(ENTRIES)

p.fit(X, nand_y)

print(f"""NAND:
[0, 0]: {nand_g([0,0])}
[0, 1]: {nand_g([0,1])}
[1, 0]: {nand_g([1,0])}
[1, 1]: {nand_g([1,1])}
""")
```

### Thanks 
Thanks for taking your time to review an example for the easyAIs framework. We hope this content was usefull :).

easyAIs's creator, Uriel Rubio.

