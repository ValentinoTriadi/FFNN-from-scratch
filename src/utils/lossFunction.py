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

    def mse(self, y_pred, y_true):
        sse = np.sum((y_pred - y_true) ** 2)
        return sse / (y_pred.shape[0] * y_pred.shape[1])

    def binarycrossentropy(self, y_pred, y_true):
        if y_true.shape[0] != y_true.shape[0]:
            raise ValueError("Panjang input dengan target tidak sama")

        if y_pred.shape[0] > 2:
            raise ValueError(
                "Fungsi ini hanya bisa digunakan untuk classification dengan jumlah class"
            )

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        y_test = y_true
        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(bce)

    def categoricalcrossentropy(self, y_pred, y_true):
        epsilon = 1e-3
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        cce = []
        for i in range(len(y_true)):
            temp = -np.sum(y_true[i] * np.log(y_pred[i]))
            cce.append(temp)
        return np.clip(np.mean(cce), epsilon, None)
