from typing import List
import numpy as np
from enum import Enum

class ActivationFunctionMethod(Enum):
    LINEAR = "Linear",
    SIGMOID = "Sigmoid",
    RELU = "ReLU",
    TANH = "Tanh",
    SOFTMAX = "Softmax"

class ActivationFunction:
    global_fungsi_aktivasi : List[ActivationFunctionMethod]= list(ActivationFunctionMethod)

    def __init__(self, fungsi_aktivasi: list[str]):
        self.fungsi_aktivasi = fungsi_aktivasi

    def validate_input(self, fungsi_aktivasi):
        for i in fungsi_aktivasi:
            if i not in ActivationFunction.global_fungsi_aktivasi:
                raise ValueError("Fungsi aktivasi tidak valid")

    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return 1

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return 1 * (x > 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return (2 / (np.exp(x) - np.exp(-x))) ** 2

    def softmax(self, x, i):
        exps = np.exp(x[i])
        return exps / np.sum(np.exp(x), axis=0)

    def softmax_derivative(self, x):
        # TODO: Implementasi turunan softmax
        return x

    def get_activation_function(self, fungsi_aktivasi: ActivationFunctionMethod):
        match fungsi_aktivasi:
            case ActivationFunctionMethod.LINEAR:
                return self.linear
            case ActivationFunctionMethod.RELU:
                return self.relu
            case ActivationFunctionMethod.SIGMOID:
                return self.sigmoid
            case ActivationFunctionMethod.TANH:
                return self.tanh
            case ActivationFunctionMethod.SOFTMAX:
                return self.softmax

    def get_batch_activation_function(self):
        for i in range(len(self.fungsi_aktivasi)):
            yield self.get_activation_function(self.fungsi_aktivasi[i])

    def get_activation_derivative(self, fungsi_aktivasi: str):
        match fungsi_aktivasi:
            case ActivationFunctionMethod.LINEAR:
                return self.linear_derivative
            case ActivationFunctionMethod.RELU:
                return self.relu_derivative
            case ActivationFunctionMethod.SIGMOID:
                return self.sigmoid_derivative
            case ActivationFunctionMethod.TANH:
                return self.tanh_derivative
            case ActivationFunctionMethod.SOFTMAX:
                return self.softmax_derivative

    def get_batch_activation_derivative(self):
        for i in range(len(self.fungsi_aktivasi)):
            yield self.get_activation_derivative(self.fungsi_aktivasi[i])
