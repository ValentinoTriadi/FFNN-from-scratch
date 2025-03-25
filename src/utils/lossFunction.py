import numpy as np
import torch
import torch.nn as nn
from typing import List
from enum import Enum


class LossFunctionMethod(Enum):
    MSE = "MSE"
    BinaryCrossEntropy = "BinaryCrossEntropy"
    CategoricalCrossEntropy = "CategoricalCrossEntropy"


class LossFunction:

    global_fungsi_loss: List[str] = [method.value for method in LossFunctionMethod]

    def __init__(self, fungsi_loss: str):
        self.fungsi_loss = fungsi_loss

    def inject_class_num(self, class_num):
        self.class_num = class_num

    def get_lost_function(self):
        if self.fungsi_loss == LossFunctionMethod.MSE.value:
            return self.mse
        elif self.fungsi_loss == LossFunctionMethod.BinaryCrossEntropy.value:
            return self.binarycrossentropy
        elif self.fungsi_loss == LossFunctionMethod.CategoricalCrossEntropy.value:
            return self.categoricalcrossentropy
        else:
            raise ValueError("Fungsi loss tidak valid")

    def get_loss_derivative(self):
        if self.fungsi_loss == LossFunctionMethod.MSE.value:
            return self.mse_derivative
        elif self.fungsi_loss == LossFunctionMethod.BinaryCrossEntropy.value:
            return self.binary_cross_entropy_derivative
        elif self.fungsi_loss == LossFunctionMethod.CategoricalCrossEntropy.value:
            return self.categorical_cross_entropy_derivative
        else:
            raise ValueError("Fungsi loss tidak valid")


    def mse(self, y_pred, y_true):
        sse = np.sum((y_pred - y_true) ** 2)
        return sse / (y_pred.shape[0] * y_pred.shape[1])

    def binarycrossentropy(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            raise ValueError("Panjang input dengan target tidak sama")

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # ✅ Hindari log(0)

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


    def categoricalcrossentropy(self, y_pred, y_true):
        epsilon = 1e-15  
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))



    def mse_derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) 

    def binary_cross_entropy_derivative(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon)

    def categorical_cross_entropy_derivative(self, y_pred, y_true):
        return y_pred - y_true


