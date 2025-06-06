from typing import List
import numpy as np
from enum import Enum


class WeightInitiationMethod(Enum):
    ZERO = "zero"
    UNIFORM = "uniform"
    NORMAL = "normal"
    XAVIER_UNIFORM = "xavier-uniform"
    XAVIER_NORMAL = "xavier-normal"


class WeightInitiation:
    global_inisialisasi_bobot: List[str] = [
        method.value for method in WeightInitiationMethod
    ]

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
        epoch: int = 1,
        low: float = -1.0,
        high: float = 1.0,
        mean: float = 0,
        std: float = 0,
        seed: int = 0,
    ):
        match (self.model):
            case WeightInitiationMethod.ZERO.value:
                return self.zero()
            case WeightInitiationMethod.UNIFORM.value:
                return self.uniform(low, high, seed)
            case WeightInitiationMethod.NORMAL.value:
                return self.normal(mean, std, seed)
            case WeightInitiationMethod.XAVIER_UNIFORM.value:
                return self.xavier_uniform(seed)
            case WeightInitiationMethod.XAVIER_NORMAL.value:
                return self.xavier_normal(seed)
            case _:
                raise ValueError("Metode inisialisasi bobot tidak valid")

    def zero(self):
        return np.array(
            [
                np.zeros(
                    (
                        self.jumlah_neuron[i - 1] + 1,
                        self.jumlah_neuron[i],
                    )
                )
                for i in range(1, self.jumlah_layer + 1)
            ],
            dtype=object,
        )

    def uniform(self, low: float, high: float, seed: int):
        np.random.seed(seed)
        return np.array(
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
            ],
            dtype=object,
        )

    def normal(self, mean: float, std: float, seed: int):
        np.random.seed(seed)
        return np.array(
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
            ],
            dtype=object,
        )

    def xavier_uniform(self, seed: int):
        np.random.seed(seed)
        return np.array(
            [
                np.random.uniform(
                    -np.sqrt(6 / (self.jumlah_neuron[i - 1] + self.jumlah_neuron[i])),
                    np.sqrt(6 / (self.jumlah_neuron[i - 1] + self.jumlah_neuron[i])),
                    (
                        self.jumlah_neuron[i - 1] + 1,
                        self.jumlah_neuron[i],
                    ),
                )
                for i in range(1, self.jumlah_layer + 1)
            ],
            dtype=object,
        )

    def xavier_normal(self, seed: int):
        np.random.seed(seed)
        return np.array(
            [
                np.random.normal(
                    0,
                    np.sqrt(2 / (self.jumlah_neuron[i - 1] + self.jumlah_neuron[i])),
                    (
                        self.jumlah_neuron[i - 1] + 1,
                        self.jumlah_neuron[i],
                    ),
                )
                for i in range(1, self.jumlah_layer + 1)
            ],
            dtype=object,
        )
