from typing import List
import numpy as np
from enum import Enum

class WeightInitiationMethod(Enum):
    ZERO = "Zero"
    UNIFORM = "Uniform"
    NORMAL = "Normal"

class WeightInitiation:
    global_inisialisasi_bobot: List[WeightInitiationMethod] = list(WeightInitiationMethod)

    def __init__(
        self,
        model: WeightInitiationMethod,
        jumlah_layer: int,
        jumlah_neuron: list[int],
    ):
        if model not in WeightInitiation.global_inisialisasi_bobot:
            raise ValueError("Metode inisialisasi bobot tidak valid")
        self.model = model
        self.jumlah_layer = jumlah_layer
        self.jumlah_neuron = jumlah_neuron

    def init_weights(
        self,
        input_count: int = 1,
        epoch: int = 1,
        low: float = -1.0,
        high: float = 1.0,
        mean: float = 0,
        std: float = 0,
        seed: int = 0,
    ):
        match(self.model):
            case WeightInitiationMethod.ZERO:
                return self.zero(input_count, epoch)
            case WeightInitiationMethod.UNIFORM:
                return self.uniform(input_count, epoch, low, high, seed)
            case WeightInitiationMethod.NORMAL:
                return self.normal(input_count, epoch, mean, std, seed)
            case _:
                raise ValueError("Metode inisialisasi bobot tidak valid")

    def zero(self, input_count: int, epoch: int):
        return [
            [
                np.zeros(
                    (
                        self.jumlah_neuron[i - 1] + 1,
                        self.jumlah_neuron[i],
                    )
                )
                for i in range(1, self.jumlah_layer + 1)
            ]
            for j in range(epoch)
        ]

    def uniform(self, input_count: int, epoch: int, low: float, high: float, seed: int):
        np.random.seed(seed)
        return [
            [
                np.random.uniform(
                    low,
                    high,
                    (
                        self.jumlah_neuron[i - 1] + 1,
                        self.jumlah_neuron[i],
                    ),
                )
                for i in range(1, self.jumlah_layer + 1)
            ]
            for j in range(epoch)
        ]

    def normal(self, input_count: int, epoch: int, mean: float, std: float, seed: int):
        np.random.seed(seed)
        return [
            [
                np.random.normal(
                    mean,
                    std,
                    (
                        self.jumlah_neuron[i - 1] + 1,
                        self.jumlah_neuron[i],
                    ),
                )
                for i in range(1, self.jumlah_layer + 1)
            ]
            for j in range(epoch)
        ]
