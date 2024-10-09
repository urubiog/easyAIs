# :information_source: Oficial documentation

[⇦ Repository](https://github.com/urubiog/easyAIs) \
[⇦ Source Code](https://github.com/urubiog/easyAIs/tree/main/src/easyAIs)

> 1. [Introduction](#introduction)
>    - [Releases](#releases)
>    - [Features and Functionality](#features-and-functionality)
>    - [Technologies Used](#technologies-used)
>    - [Installation](#installation)
>    - [Basic instructions](#basic-instructions)
>    - [Contributions](#contributions)
>    - [License](#license)

---

## Introduction
**`easyAIs`** is a [Python](https://python.org) framework designed to simplify the use of Deep Learning models and architectures, alongside integrated tools for data processing, loading, and visualization. This personal project aims to maintain a simple yet useful structure, avoiding the use of analytical sub-libraries like scikit-learn, and providing a friendly API for implementing artificial intelligence models.

> *For source code documentation, visit [Source overview](./Source/Overview.md)*.

### Releases

- **Version 0.0.1**: Initial release of easyAIs with basic functionality for deep learning models and data processing.
> *Watch out all the releases in [Releases](./Releases.md)*

### Features and Functionality
- **Deep Learning Models**: Streamlined implementation of common deep learning architectures.
- **Data Processing**: Tools for efficient data handling and manipulation.
- **Visualization**: Integrated support for plotting and visualizing data and model results.

In addition to these features, easyAIs emphasizes a user-friendly approach that simplifies the setup and execution of deep learning models. This design philosophy ensures that users, regardless of their experience level, can seamlessly integrate and utilize various functionalities without getting overwhelmed by technical complexities.

### Technologies Used
All technologies used in the project are listed in the [dependencies](./Dependencies.md) section. Visit the page for more information.
- [**Python**](./Dependencies.md#Python): Programming language.
- [**Plotly**](.Dependencies.md#Plotly): For data visualization.
- [**Pygrad**](.Dependencies.md#Pygrad): For gradient-based operations.
- [**Typing**](.Dependencies.md#Typing): For type hints and static type checking.
- [**Functools**](.Dependencies.md#Functools): For higher-order functions.
- [**Random**](.Dependencies.md#Random): For random number generation.

### Installation
To install `easyAIs`, simply use pip:

```bash
pip install easyAIs
```
> For more information visit [Installation Guide](./Installation.md)

### Basic Instructions
To use **easyAIs**, you can start by importing the necessary modules and creating instances of the available classes. Here’s a simple example:

```python
from easyAIs.arquitectures import Perceptron

# Create a model instance
model = Perceptron(2)

# Train the model
test_data = model.fit(data, params)

# Evaluate the model
results = model.evaluate(test_data)

print(results)
```
> Visit the [Examples](./Examples/Overview.md) section for more tutorials.


### Contributions

**We welcome contributions to** `easyAIs`**!** To contribute, please follow these guidelines:

- **Pull Requests**: Fork the repository, create a new branch for your changes, and submit a pull request with a detailed description of the modifications. Ensure that your changes are well-documented and include tests where applicable.
- **Coding Standards**: Follow the coding standards outlined in the style guide and ensure that your code is properly formatted.
- **Issues and Tags**: Check the issues section for any tasks or bugs that need attention. Use appropriate labels and provide detailed information when creating new issues.

### License

`easyAIs` is licensed under the [MIT License](https://mit-license.org/). You are free to use, modify, and distribute the code, but please include the license and copyright notice in all copies or substantial portions of the software.

---
Copyright (c) 2024 Uriel Rubio García. All Rights Reserved.

