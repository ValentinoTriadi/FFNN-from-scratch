import numpy as np
from torch import nn, randn, tensor

global_fungsi_loss = ["MSE", "BinaryCrossEntropy", "CategoricalCrossEntropy"]


class LossFunction:
    def __init__(self, fungsi_loss: str):
        self.fungsi_loss = fungsi_loss

    def get_lost_function(self):
        if self.fungsi_loss == "MSE":
            return self.mse
        elif self.fungsi_loss == "BinaryCrossEntropy":
            return self.binarycrossentropy
        elif self.fungsi_loss == "CategoricalCrossEntropy":
            return self.categoricalcrossentropy
        else:
            raise ValueError("Fungsi loss tidak valid")

    def mse(self, y_pred, y_true):
        input = tensor(y_pred)
        target = tensor(y_true)

        loss = nn.MSELoss()
        mse_torch = loss(input, target)
        return mse_torch

    def binarycrossentropy(self, y_pred, y_true):
        # TODO: Implementasi Binary Cross Entropy
        pass

    def categoricalcrossentropy(self, y_pred, y_true):
        # TODO: Implementasi Categorical Cross Entropy
        pass
