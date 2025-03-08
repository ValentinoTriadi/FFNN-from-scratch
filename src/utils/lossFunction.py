import numpy as np

global_fungsi_loss = ["MSE", "BinaryCrossEntropy", "CategoricalCrossEntropy"]


class LossFunction:
    def __init__(self, fungsi_loss: str):
        self.fungsi_loss = fungsi_loss

    def hitung_loss(self, y_pred, y_true):
        if self.fungsi_loss == "MSE":
            return self.mse(y_pred, y_true)
        elif self.fungsi_loss == "BinaryCrossEntropy":
            return self.binarycrossentropy(y_pred, y_true)
        elif self.fungsi_loss == "CategoricalCrossEntropy":
            return self.categoricalcrossentropy(y_pred, y_true)
        else:
            raise ValueError("Fungsi loss tidak valid")

    def mse(self, y_pred, y_true):
        # TODO: Implementasi MSE
        pass

    def binarycrossentropy(self, y_pred, y_true):
        # TODO: Implementasi Binary Cross Entropy
        pass

    def categoricalcrossentropy(self, y_pred, y_true):
        # TODO: Implementasi Categorical Cross Entropy
        pass
