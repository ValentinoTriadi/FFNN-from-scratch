import numpy as np

global_inisialisasi_bobot = ["zero", "uniform", "normal"]


class WeightInitiation:
    def __init__(
        self,
        model: str,
        jumlah_layer: int,
        jumlah_neuron: list[int],
    ):
        if model not in global_inisialisasi_bobot:
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
        if self.model == "zero":
            return self.zero(input_count, epoch)
        elif self.model == "uniform":
            return self.uniform(input_count, epoch, low=low, high=high, seed=seed)
        elif self.model == "normal":
            return self.normal(input_count, epoch, mean=mean, std=std, seed=seed)
        else:
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
